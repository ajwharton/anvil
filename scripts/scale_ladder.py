#!/usr/bin/env python3
"""Expert-v2 scale ladder — convert (+ optional train) at 1k / 5k / 50k.

Does **not** download Bridge/OXE. For forge, point ``--source`` at an episode
pack already on NVMe. For CI/laptop use ``--demo`` (synthetic pack + tiny
row counts that exercise the same code path).

Examples::

  # CI / laptop — all rungs, synthetic, seconds
  python scripts/scale_ladder.py --demo --work-dir /tmp/anvil-ladder

  # forge — 1k real rows from Bridge episode pack
  python scripts/scale_ladder.py --rung 1k \\
    --source /mnt/data/datasets/bridge_v2/episode_pack \\
    --media-root /mnt/data/anvil-media \\
    --work-dir /mnt/data/anvil-runs/scale-ladder \\
    --endpoint local:// --no-demo

  # convert only (no train)
  python scripts/scale_ladder.py --demo --rung 5k --convert-only

  # print plan JSON
  python scripts/scale_ladder.py --plan-only --rung all
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rung",
        action="append",
        default=None,
        help="rung id (1k|5k|50k); repeatable; default all",
    )
    p.add_argument("--demo", action="store_true", help="synthetic pack + demo_rows")
    p.add_argument(
        "--no-demo",
        action="store_true",
        help="force real max_rows (requires --source on forge)",
    )
    p.add_argument("--source", default=None, help="episode_pack directory")
    p.add_argument("--media-root", default=None)
    p.add_argument(
        "--work-dir",
        default=str(Path.home() / ".anvil" / "scale-ladder"),
        help="jsonl + train run_dir parent",
    )
    p.add_argument("--endpoint", default="fake://")
    p.add_argument("--dataset", default="bridge_v2")
    p.add_argument("--convert-only", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--list", action="store_true", help="list rungs and exit")
    args = p.parse_args(argv)

    from anvil.recipes.scale_ladder import (
        build_ladder_plan,
        exercise_ladder,
        list_rungs,
    )

    if args.list:
        for r in list_rungs():
            print(
                f"{r.id:4} max_rows={r.max_rows:6} train_steps={r.train_steps:4} "
                f"ckpt_every={r.checkpoint_every:3} demo_rows={r.demo_rows:4}  {r.notes}"
            )
        return 0

    demo = True if args.demo or not args.no_demo else False
    if args.no_demo:
        demo = False
    if not demo and not args.source:
        print("error: --no-demo requires --source (lab episode pack)", file=sys.stderr)
        return 2

    rungs = args.rung or ["all"]
    plan = build_ladder_plan(
        rungs=rungs,
        demo=demo,
        source=args.source,
        media_root=args.media_root,
        jsonl_root=args.work_dir,
        dataset=args.dataset,
    )
    if args.plan_only:
        print(json.dumps(plan.to_public(), indent=2))
        return 0

    results = exercise_ladder(
        plan,
        work_dir=args.work_dir,
        endpoint=args.endpoint,
        train=not args.convert_only,
    )
    failed = 0
    report = []
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"[{status}] rung={r.rung.id} {r.detail}")
        if not r.ok:
            failed += 1
        report.append(
            {
                "rung": r.rung.id,
                "ok": r.ok,
                "rows_converted": r.rows_converted,
                "steps_run": r.steps_run,
                "jsonl_path": r.jsonl_path,
                "run_dir": r.run_dir,
                "checkpoint_path": r.checkpoint_path,
                "detail": r.detail,
            }
        )
    out = Path(args.work_dir) / "ladder_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"demo": demo, "results": report}, indent=2) + "\n", encoding="utf-8")
    print(f"report={out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
