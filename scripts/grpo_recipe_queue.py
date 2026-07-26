#!/usr/bin/env python3
"""Run a multi-stage RL recipe queue with early-stop handoff.

When a stage hits a dead signal (reward ceiling / collapse), Anvil abandons
that stage and immediately starts the next problem on the **same LoRA** —
so overnight power goes to the next hard task, not a flatlined chart.

Examples::

  # CI / laptop
  python scripts/grpo_recipe_queue.py --endpoint fake:// --recipe-builtin hard-bank

  # forge overnight
  python scripts/grpo_recipe_queue.py \\
    --endpoint local:// \\
    --model /mnt/data/models/qwen2.5-1.5b-instruct \\
    --recipe recipes/arith_curriculum_v1.json \\
    --observe-root /mnt/data/anvil-observe \\
    --run-prefix grpo-curric-night

  # watch: /observe/<prefix>-queue  (stage events)
  #        /observe/<prefix>-s0-15x8p7  (per-stage curves)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="fake://")
    p.add_argument(
        "--model",
        default="/mnt/data/models/qwen2.5-1.5b-instruct",
    )
    p.add_argument(
        "--recipe",
        default=None,
        help="path to recipe JSON (default: recipes/arith_curriculum_v1.json)",
    )
    p.add_argument(
        "--recipe-builtin",
        choices=("hard-bank",),
        default=None,
        help="use built-in recipe instead of JSON file",
    )
    p.add_argument("--observe-root", default=None)
    p.add_argument("--run-prefix", default=None)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--probe-every", type=int, default=1)
    args = p.parse_args(argv)

    from anvil.recipes.rl_queue import (
        load_rl_queue_recipe,
        recipe_from_hard_bank,
        run_rl_queue,
    )

    root = args.observe_root
    if root is None:
        root = os.environ.get("ANVIL_OBSERVE_ROOT")
    if root is None:
        lab = Path("/mnt/data/anvil-observe")
        root = str(lab if lab.parent.is_dir() else Path.home() / ".anvil" / "observe")

    if args.recipe_builtin == "hard-bank":
        recipe = recipe_from_hard_bank(max_steps=20 if args.endpoint.startswith("fake") else 100)
    else:
        recipe_path = args.recipe
        if recipe_path is None:
            # repo-relative default
            here = Path(__file__).resolve().parents[1]
            recipe_path = str(here / "recipes" / "arith_curriculum_v1.json")
        recipe = load_rl_queue_recipe(recipe_path)

    print(f"recipe={recipe.id} stages={len(recipe.stages)} observe_root={root}")
    for i, s in enumerate(recipe.stages):
        print(f"  [{i}] {s.id} gold={s.gold} max_steps={s.max_steps}")

    fake = args.endpoint.startswith("fake://")
    result = run_rl_queue(
        recipe,
        base_model=args.model if not fake else "toy/lm",
        endpoint=args.endpoint,
        observe_root=root,
        run_prefix=args.run_prefix or recipe.id,
        rank=args.rank,
        max_tokens=args.max_tokens,
        probe_every=args.probe_every,
        fake_prompts=fake,
    )

    print(f"adapter={result.adapter_id} stages_completed={result.stages_run}")
    for o in result.stages:
        r = o.result
        print(
            f"  {o.stage.id}: steps={r.steps_run} "
            f"final_r={r.mean_reward[-1] if r.mean_reward else None} "
            f"early_stop={r.early_stop_reason} "
            f"advanced={o.advanced} halt={o.queue_halted} "
            f"observe=/{o.observe_run_id}"
        )
    print(f"queue events: {root}/{args.run_prefix or recipe.id}-queue/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
