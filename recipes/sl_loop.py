#!/usr/bin/env python3
"""Minimal SFT loop against any Anvil endpoint — the Phase 1 GPU smoke.

One script, two ways to run the same four verbs:

  On the GPU host (in-process torch+PEFT, CUDA auto-detected):
      python recipes/sl_loop.py --base-model Qwen/Qwen2.5-0.5B-Instruct --steps 100

  From a laptop against `anvil serve` on that host:
      anvil serve --backend local --host 0.0.0.0 --port 8741   # on the GPU host
      python recipes/sl_loop.py --endpoint http://forge:8741   # on the laptop

  CPU sanity check with a tiny random model (no chat template → ChatML):
      python recipes/sl_loop.py \
          --base-model hf-internal-testing/tiny-random-gpt2 \
          --chat-template chatml --lr 1e-2 --rank 8 \
          --target-modules c_attn,c_proj,c_fc

Exit code is a smoke gate: 0 iff the CE loss decreased. Data is a built-in
8-example demo set, or `--data path.jsonl` with one chat per line:

  {"messages": [{"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from anvil.client.service import ServiceClient
from anvil.protocol.messages import Example, Message, TextPart
from anvil.protocol.types import AdamParams, SamplingParams

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # roadmap smoke band: 0.5B–4B

# ChatML fallback for base models that ship no chat template (same shape the
# renderer tests pin). Training with template A and sampling with template B
# silently corrupts the loss — pass one template and stick to it.
CHATML = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

_DEMO: list[tuple[str, str]] = [
    ("2+2?", "4"),
    ("3+3?", "6"),
    ("5+5?", "10"),
    ("1+1?", "2"),
    ("10-4?", "6"),
    ("Capital of France?", "Paris"),
    ("Opposite of hot?", "cold"),
    ("Color of the sky on a clear day?", "blue"),
]


def _demo_examples() -> list[Example]:
    return [
        Example(
            messages=(
                Message(role="user", content=(TextPart(text=q),)),
                Message(role="assistant", content=(TextPart(text=a),)),
            )
        )
        for q, a in _DEMO
    ]


def _load_jsonl(path: str) -> list[Example]:
    examples: list[Example] = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        msgs = tuple(
            Message(role=m["role"], content=(TextPart(text=m["content"]),))
            for m in raw["messages"]
        )
        if len(msgs) < 2:
            raise ValueError(f"{path}:{lineno}: need at least 2 messages")
        examples.append(Example(messages=msgs))
    if not examples:
        raise ValueError(f"{path}: no examples found")
    return examples


def _prompt_prefix(example: Example) -> Sequence[Message]:
    """Conversation up to the last assistant turn — the sample-side prompt."""
    msgs = list(example.messages)
    if msgs and msgs[-1].role == "assistant":
        msgs = msgs[:-1]
    return msgs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal SFT loop against any Anvil endpoint.")
    p.add_argument("--endpoint", default="local://",
                   help="local:// (in-process) or http(s)://host:port (anvil serve)")
    p.add_argument("--base-model", default=DEFAULT_MODEL)
    p.add_argument("--data", default=None, help="JSONL of chat examples (default: 8-example demo set)")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--sample-tokens", type=int, default=16)
    p.add_argument("--chat-template", default=None,
                   help="'chatml' for the built-in fallback, or a path to a .jinja template "
                        "(required when the base model ships no chat template)")
    p.add_argument("--export-dir", default="out/sl-loop-adapter")
    p.add_argument("--no-export", action="store_true")
    # in-process (local://) only — ignored by remote endpoints, where the
    # `anvil serve` process owns device/target-modules:
    p.add_argument("--device", default=None, help="cpu/cuda (default: auto)")
    p.add_argument("--target-modules", default=None,
                   help="comma-separated LoRA targets (default: peft's per-architecture choice)")
    p.add_argument("--root", default=None, help="checkpoint root (default: .anvil-local)")
    p.add_argument("--allow-tiny-models", action="store_true",
                   help="bypass the hidden-size capacity gate (debug only)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.endpoint.startswith("local"):
        from anvil.backends.local import LocalBackend

        backend = LocalBackend(
            device=args.device,
            root=args.root,
            target_modules=(args.target_modules.split(",") if args.target_modules else None),
            allow_tiny_models=args.allow_tiny_models,
        )
        svc = ServiceClient(endpoint=args.endpoint, backend=backend)
        import torch

        torch.manual_seed(args.seed)
        print(f"backend: local:// device={backend.device} dtype={backend.dtype}")
    else:
        svc = ServiceClient(endpoint=args.endpoint)
        print(f"backend: {args.endpoint}")

    from anvil.render.hf import HFChatRenderer

    template = None
    if args.chat_template == "chatml":
        template = CHATML
    elif args.chat_template:
        template = Path(args.chat_template).read_text()
    renderer = HFChatRenderer(args.base_model, chat_template=template)

    examples = _load_jsonl(args.data) if args.data else _demo_examples()
    data = [renderer.render_example_for_sft(ex) for ex in examples]
    print(f"data: {len(data)} examples, {sum(len(d.model_input.token_ids()) for d in data)} tokens")

    tc = svc.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    print(f"adapter: {tc.adapter_id}  rank={args.rank} lr={args.lr} steps={args.steps}")

    losses: list[float] = []
    for step in range(1, args.steps + 1):
        fb = tc.forward_backward(data).result()
        tc.optim_step(AdamParams(learning_rate=args.lr)).result()
        losses.append(fb.loss)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step {step:>4}/{args.steps}  loss={fb.loss:.4f}")

    ref = tc.save_state("sl-loop-final")
    print(f"checkpoint: {ref.path}")

    sc = tc.save_weights_and_get_sampling_client("sl-loop-final")
    prompt = renderer.render_prompt(_prompt_prefix(examples[0]))
    out = sc.sample(
        prompt,
        SamplingParams(max_tokens=args.sample_tokens, temperature=0.0, seed=args.seed),
    ).result()
    first_user = "".join(
        p.text if isinstance(p, TextPart) else "<media>"
        for p in examples[0].messages[0].parts()
    )
    print(f"sample (greedy, trained adapter) on {first_user!r}:")
    print(f"  {renderer.decode(out.sequences[0].tokens)!r}")

    if not args.no_export:
        res = tc.export_adapter(args.export_dir)
        print(f"export: {res.path} ({res.format.value})")

    ok = losses[-1] < losses[0]
    print(f"loss: {losses[0]:.4f} → {losses[-1]:.4f}  "
          f"({'decreased' if ok else 'DID NOT DECREASE'})")
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
