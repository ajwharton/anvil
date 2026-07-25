#!/usr/bin/env python3
"""Productized GRPO demo with live anvil-web observe.

Writes ``metrics.jsonl`` / ``probes.jsonl`` under
``$ANVIL_OBSERVE_ROOT/<run_id>/`` (default ``/mnt/data/anvil-observe`` on lab
hosts, else ``~/.anvil/observe``). Point anvil-web at the same root and open
``/observe/<run_id>`` — reward curves and probes stream via SSE.

Examples::

  # laptop / CI (fake backend — proves observe wiring)
  python scripts/grpo_observe_demo.py --endpoint fake:// --steps 5

  # forge — real LoRA GRPO on Qwen2.5-1.5B arithmetic
  python scripts/grpo_observe_demo.py \\
    --endpoint local:// \\
    --model /mnt/data/models/qwen2.5-1.5b-instruct \\
    --steps 30 --group-size 4 \\
    --run-id grpo-arith-demo

  # then (same host or Mac with shared FS / ssh tunnel):
  ANVIL_OBSERVE_ROOT=/mnt/data/anvil-observe anvil-web --host 0.0.0.0 --port 7600
  # open http://forge:7600/observe/grpo-arith-demo
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path


def _default_observe_root() -> Path:
    env = os.environ.get("ANVIL_OBSERVE_ROOT")
    if env:
        return Path(env)
    lab = Path("/mnt/data/anvil-observe")
    if lab.parent.is_dir():
        return lab
    return Path.home() / ".anvil" / "observe"


def _load_tokenizer(model: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def _chat_prompt_ids(tok, user_text: str) -> list[int]:
    messages = [{"role": "user", "content": user_text}]
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return [int(t) for t in tok.encode(text, add_special_tokens=False)]
    return [int(t) for t in tok.encode(user_text, add_special_tokens=True)]


def _arith_problems() -> list[tuple[str, str]]:
    """(user prompt, canonical answer digits). Small verifiable set."""
    pairs = [
        ("What is 2+2? Reply with only the number.", "4"),
        ("What is 3+5? Reply with only the number.", "8"),
        ("What is 7-2? Reply with only the number.", "5"),
        ("What is 4*3? Reply with only the number.", "12"),
        ("What is 9+1? Reply with only the number.", "10"),
        ("What is 6/2? Reply with only the number.", "3"),
        ("What is 5+5? Reply with only the number.", "10"),
        ("What is 8-3? Reply with only the number.", "5"),
    ]
    return pairs


def _make_reward_and_detok(tok, answer_by_prompt_key: dict[str, str]):
    """Reward: 1.0 if first integer in completion matches gold, else 0.0."""

    def detokenize(tokens) -> str:
        return tok.decode(list(tokens), skip_special_tokens=True)

    def reward_fn(_text: str, tokens) -> float:
        # prompt identity is not passed by run_grpo — score by answer pattern only
        # when a single global answer set is used we match ANY gold (loose) or
        # extract first integer and check membership. For product demo we use
        # per-completion: first integer in text must equal one of the golds
        # that appear as a full match preference.
        text = detokenize(tokens).strip()
        m = re.search(r"-?\d+", text)
        if not m:
            return 0.0
        got = m.group(0)
        # Prefer exact gold set membership
        if got in answer_by_prompt_key.values():
            return 1.0
        return 0.0

    return reward_fn, detokenize


def _make_exact_answer_reward(tok, gold: str):
    """Verifiable reward for a *single* gold answer (completion-only API).

    Stock ``run_grpo`` only passes completion tokens into ``reward_fn``, so a
    multi-prompt gold *set* is too loose (any correct number from any problem
    scores 1). Use one problem per run for a clean reward curve, or extend
    run_grpo later to pass prompt identity.
    """

    def detokenize(tokens) -> str:
        return tok.decode(list(tokens), skip_special_tokens=True)

    def reward_fn(_text: str, tokens) -> float:
        text = detokenize(tokens).strip()
        # First integer in the completion must equal gold exactly.
        m = re.search(r"-?\d+", text)
        if not m:
            return 0.0
        return 1.0 if m.group(0) == gold else 0.0

    return reward_fn, detokenize


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="fake://")
    p.add_argument(
        "--model",
        default="/mnt/data/models/qwen2.5-1.5b-instruct",
        help="local path or HF id (local:// needs a real LM)",
    )
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument(
        "--observe-root",
        default=None,
        help="override ANVIL_OBSERVE_ROOT (metrics parent dir)",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="observe run folder name (default: grpo-<timestamp>)",
    )
    p.add_argument("--probe-every", type=int, default=1)
    p.add_argument(
        "--print-url",
        action="store_true",
        help="print observe URL path and sleep so web can attach first",
    )
    p.add_argument(
        "--attach-wait",
        type=float,
        default=0.0,
        help="seconds to wait after creating run_dir before training (for SSE attach)",
    )
    args = p.parse_args(argv)

    observe_root = Path(args.observe_root) if args.observe_root else _default_observe_root()
    run_id = args.run_id or f"grpo-{time.strftime('%Y%m%d-%H%M%S')}"
    # Safe id: alnum + dash only (web _SAFE_RUN_ID)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}", run_id):
        raise SystemExit(f"bad run-id {run_id!r}")
    run_dir = observe_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Touch empty metrics so /observe/{id} 200s before first step
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        metrics_path.touch()

    print(f"observe_root={observe_root}")
    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"open: /observe/{run_id}")
    print(f"      ANVIL_OBSERVE_ROOT={observe_root} anvil-web --host 0.0.0.0 --port 7600")
    if args.attach_wait > 0:
        print(f"waiting {args.attach_wait:.0f}s for web attach…")
        time.sleep(args.attach_wait)

    from anvil.recipes.grpo import run_grpo

    if args.endpoint.startswith("fake://"):
        # Fake path: integer prompts + toy even-token reward (unit-test shape)
        prompts = [list(range(10, 26)), list(range(20, 36)), list(range(30, 46))]
        probes = [list(range(10, 18)), list(range(20, 28))]
        result = run_grpo(
            base_model=args.model,
            prompts=prompts,
            steps=args.steps,
            group_size=args.group_size,
            endpoint=args.endpoint,
            run_dir=str(run_dir),
            probes=probes,
            probe_every=args.probe_every,
            detokenize=lambda toks: f"<{len(list(toks))} toks>",
            overrides={"rank": args.rank, "max_tokens": args.max_tokens},
        )
    else:
        tok = _load_tokenizer(args.model)
        # Single verifiable problem so completion-only reward stays meaningful.
        # (Multi-prompt needs prompt-aware reward_fn in run_grpo — later.)
        user, gold = _arith_problems()[0]
        prompt_ids = _chat_prompt_ids(tok, user)
        # Repeat the same prompt so each step has a small "batch" of groups.
        prompts = [list(prompt_ids) for _ in range(4)]
        probes = [list(prompt_ids)]
        reward_fn, detokenize = _make_exact_answer_reward(tok, gold)
        print(
            f"problem={user!r} gold={gold!r} "
            f"prompt_groups={len(prompts)} group_size={args.group_size} steps={args.steps}"
        )
        result = run_grpo(
            base_model=args.model,
            prompts=prompts,
            reward_fn=reward_fn,
            steps=args.steps,
            group_size=args.group_size,
            endpoint=args.endpoint,
            run_dir=str(run_dir),
            probes=probes,
            probe_every=args.probe_every,
            detokenize=detokenize,
            overrides={
                "rank": args.rank,
                "max_tokens": args.max_tokens,
                "temperature": 0.9,
            },
        )

    print(
        f"steps={result.steps_run} adapter={result.adapter_id} "
        f"final_reward={result.mean_reward[-1] if result.mean_reward else None}"
    )
    print("reward_mean:", [round(r, 3) for r in result.mean_reward])
    print("losses:", [round(x, 4) for x in result.losses])
    print(f"metrics: {run_dir / 'metrics.jsonl'}")
    print(f"probes:  {run_dir / 'probes.jsonl'}")
    print(f"LIVE UI: /observe/{run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
