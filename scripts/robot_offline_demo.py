#!/usr/bin/env python3
"""Phase 4.A smoke: offline robot policy with text-tokenized actions.

Default base is **SmolVLM-256M** (memory-constrained robot). Uses synthetic
trajectories on ``fake://`` so CI/laptops need no weights.

Examples::

  # laptop / CI
  python scripts/robot_offline_demo.py --steps 3

  # lab host with real endpoint + media
  python scripts/robot_offline_demo.py \\
    --endpoint local:// \\
    --model /mnt/data/models/SmolVLM-256M-Instruct \\
    --run-dir /mnt/data/anvil-runs/robot-offline-smoke \\
    --steps 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Robot offline + action tokens smoke")
    p.add_argument(
        "--model",
        default=None,
        help="base model id/path (default: SmolVLM-256M-Instruct)",
    )
    p.add_argument("--endpoint", default="fake://")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--run-dir", default=None)
    p.add_argument(
        "--scheme",
        choices=("bins", "continuous"),
        default="bins",
        help="action text scheme (default: bins)",
    )
    p.add_argument("--n-bins", type=int, default=256)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--fetch-remote", action="store_true")
    args = p.parse_args(argv)

    from anvil.protocol.action_tokens import ActionTokenizer
    from anvil.recipes.robot_offline import (
        DEFAULT_ROBOT_BASE,
        run_robot_offline,
        toy_robot_trajectories,
    )

    base = args.model or (
        "HuggingFaceTB/SmolLM2-135M-Instruct" if args.text_only else DEFAULT_ROBOT_BASE
    )
    tok = ActionTokenizer(scheme=args.scheme, n_bins=args.n_bins)
    res = run_robot_offline(
        base_model=base,
        trajectories=toy_robot_trajectories(),
        steps=args.steps,
        endpoint=args.endpoint,
        run_dir=args.run_dir,
        action_tokenizer=tok,
        fetch_remote=args.fetch_remote,
        text_only=args.text_only,
        early_stop=False,
    )
    summary = {
        "base_model": res.base_model,
        "adapter_id": res.adapter_id,
        "steps_run": res.steps_run,
        "n_train_examples": res.n_train_examples,
        "n_probe_examples": res.n_probe_examples,
        "n_train_episodes": res.n_train_episodes,
        "n_heldout_episodes": res.n_heldout_episodes,
        "action_tokenizer": res.action_tokenizer,
        "losses": res.losses,
        "run_dir": res.run_dir,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
