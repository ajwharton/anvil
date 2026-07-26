#!/usr/bin/env python3
"""Productized GRPO demo with live anvil-web observe.

Writes ``metrics.jsonl`` / ``probes.jsonl`` under
``$ANVIL_OBSERVE_ROOT/<run_id>/`` (default ``/mnt/data/anvil-observe`` on lab
hosts, else ``~/.anvil/observe``). Point anvil-web at the same root and open
``/observe/<run_id>`` — reward curves and probes stream via SSE.

Verifiable rewards live in ``anvil.recipes.verifiable`` (unit-tested). Default
``--problem hard`` uses multi-digit arithmetic so small bases are not already
saturated at reward=1 (unlike ``2+2``).

Examples::

  # laptop / CI (fake backend — proves observe wiring)
  python scripts/grpo_observe_demo.py --endpoint fake:// --steps 5

  # forge — real LoRA GRPO on Qwen2.5-1.5B
  python scripts/grpo_observe_demo.py \\
    --endpoint local:// \\
    --model /mnt/data/models/qwen2.5-1.5b-instruct \\
    --problem hard --steps 40 --group-size 8 \\
    --run-id grpo-hard-demo

  ANVIL_OBSERVE_ROOT=/mnt/data/anvil-observe anvil-web --host 0.0.0.0 --port 7600
  # open http://forge:7600/observe/grpo-hard-demo
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

from anvil.recipes.verifiable import (
    DEFAULT_HARD_PROBLEMS,
    detokenize_via_tokenizer,
    exact_integer_reward,
)


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


def _easy_problems() -> list[tuple[str, str]]:
    return [
        ("What is 2+2? Reply with only the number.", "4"),
        ("What is 3+5? Reply with only the number.", "8"),
    ]


def select_problem(kind: str, index: int) -> tuple[str, str]:
    """Return (user_prompt, gold) for ``easy`` / ``hard`` problem banks."""
    bank = list(DEFAULT_HARD_PROBLEMS) if kind == "hard" else _easy_problems()
    if not bank:
        raise ValueError("empty problem bank")
    return bank[index % len(bank)]


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
        "--problem",
        choices=("hard", "easy"),
        default="hard",
        help="hard = multi-digit (default); easy = 2+2-style (often already solved)",
    )
    p.add_argument(
        "--problem-index",
        type=int,
        default=0,
        help="which problem in the bank (mod length)",
    )
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
        "--attach-wait",
        type=float,
        default=0.0,
        help="seconds to wait after creating run_dir before training (for SSE attach)",
    )
    args = p.parse_args(argv)

    observe_root = Path(args.observe_root) if args.observe_root else _default_observe_root()
    run_id = args.run_id or f"grpo-{time.strftime('%Y%m%d-%H%M%S')}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}", run_id):
        raise SystemExit(f"bad run-id {run_id!r}")
    run_dir = observe_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
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
        user, gold = select_problem(args.problem, args.problem_index)
        prompt_ids = _chat_prompt_ids(tok, user)
        prompts = [list(prompt_ids) for _ in range(4)]
        probes = [list(prompt_ids)]
        detok = detokenize_via_tokenizer(tok)
        reward_fn = exact_integer_reward(detok, gold)
        print(
            f"problem={user!r} gold={gold!r} bank={args.problem} "
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
            detokenize=detok,
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
