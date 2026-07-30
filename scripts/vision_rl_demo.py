#!/usr/bin/env python3
"""Phase 4.B smoke: on-policy vision GRPO (+ optional stage queue).

  python scripts/vision_rl_demo.py --steps 3
  python scripts/vision_rl_demo.py --queue --steps 4
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
    p = argparse.ArgumentParser(description="Vision on-policy RL demo")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--queue", action="store_true", help="run two-stage vision RL queue")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--endpoint", default="fake://")
    p.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    args = p.parse_args(argv)

    from anvil.recipes.vision_rl import run_vision_grpo, run_vision_rl_queue

    if args.queue:
        q = run_vision_rl_queue(
            base_model=args.model,
            endpoint=args.endpoint,
            run_dir=args.run_dir or "/tmp/anvil-vision-rl-queue",
        )
        print(
            json.dumps(
                {
                    "mode": "queue",
                    "stages_run": q.stages_run,
                    "adapter_id": q.adapter_id,
                    "stages": [
                        {
                            "id": o.stage.id,
                            "steps_run": o.result.steps_run,
                            "mean_reward": o.result.mean_reward,
                            "early_stop": o.result.early_stop_reason,
                            "advanced": o.advanced,
                        }
                        for o in q.stages
                    ],
                },
                indent=2,
            )
        )
        return 0

    res = run_vision_grpo(
        base_model=args.model,
        steps=args.steps,
        endpoint=args.endpoint,
        run_dir=args.run_dir or "/tmp/anvil-vision-rl",
        early_stop=False,
    )
    print(
        json.dumps(
            {
                "mode": "vision_grpo",
                "adapter_id": res.adapter_id,
                "steps_run": res.steps_run,
                "mean_reward": res.mean_reward,
                "losses": res.losses,
                "run_dir": args.run_dir or "/tmp/anvil-vision-rl",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
