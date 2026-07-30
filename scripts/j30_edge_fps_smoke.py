#!/usr/bin/env python3
"""Storage-safe edge sample FPS smoke (Phase 4.C residual).

Measures Ollama/smol sample latency from the **lab** (or dry-run in CI).
Does **not** train, does **not** write under ``~/vision`` on the robot, and
does not leave Anvil run dirs on the edge device.

Robotics owns j30 disk policy. Prefer::

  # dry-run CI / laptop
  python scripts/j30_edge_fps_smoke.py --dry-run

  # lab → device Ollama (set URL yourself; never commit host IPs)
  ANVIL_JETSON_URL=http://<robot>:11434 \\
    python scripts/j30_edge_fps_smoke.py --n 5 --image ./one_frame.jpg

Optional ``--report`` writes JSON **only on the machine running this script**.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Edge GGUF/Ollama FPS smoke (lab-side)")
    p.add_argument("--n", type=int, default=5, help="timed samples (keep small)")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--model", default=None)
    p.add_argument("--url", default=None, help="Ollama base URL (default env/dry-run)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--image", default=None, help="optional local JPEG/PNG (lab path)")
    p.add_argument(
        "--report",
        default=None,
        help="write JSON report on this machine only (not on the robot)",
    )
    p.add_argument("--max-n", type=int, default=20, help="hard cap for n (storage/ops)")
    args = p.parse_args(argv)

    from anvil.backends.jetson import JetsonSampleConfig, measure_sample_fps

    n = min(max(1, args.n), max(1, args.max_n))
    if args.n > args.max_n:
        print(f"warning: clamped n={args.n} → {n} (max-n={args.max_n})", file=sys.stderr)

    dry = bool(args.dry_run or not (args.url or os.environ.get("ANVIL_JETSON_URL")))
    if dry and not args.dry_run and not args.url:
        # Safe default: never hit a random LAN host from CI/scripts without URL.
        dry = True

    cfg = JetsonSampleConfig(
        url=args.url or os.environ.get("ANVIL_JETSON_URL", "http://127.0.0.1:11434"),
        model=args.model or os.environ.get("ANVIL_JETSON_MODEL", "smolvlm-256m"),
        dry_run=dry,
    )
    stats = measure_sample_fps(
        config=cfg,
        n=n,
        warmup=args.warmup,
        image_path=args.image,
    )
    stats["dry_run"] = dry
    print(json.dumps(stats, indent=2))

    if args.report:
        out = Path(args.report)
        # Refuse paths that look like on-robot vision dumps if user is careless
        if "vision/out" in str(out).replace("\\", "/"):
            print(
                "refusing to write report under vision/out (edge storage policy)",
                file=sys.stderr,
            )
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"report → {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
