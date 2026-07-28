#!/usr/bin/env python3
"""Pull LeRobot hub frames into Anvil JSONL + CAS (real-domain Expert-v0 data).

Writes rows with ``cas://`` refs under ``--media-root`` and a JSONL of
instruction/images/response for ``expert_v0_smoke.py --skip-convert``.

Example::

  python scripts/build_lerobot_jsonl.py \\
    --repo lerobot/pusht --n 64 \\
    --media-root /mnt/data/anvil-media \\
    --output /mnt/data/datasets/anvil_jsonl/lerobot_pusht.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="lerobot/pusht")
    p.add_argument("--n", type=int, default=64, help="max frames/examples")
    p.add_argument("--media-root", required=True)
    p.add_argument("--output", "-o", required=True)
    args = p.parse_args(argv)

    # Reuse lab demo helper (same download + frame extract path)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import robot_vlm_sft_demo as demo  # type: ignore

    from anvil.media import LocalMediaStore

    store = LocalMediaStore(args.media_root)
    try:
        examples = demo._examples_from_lerobot(store, args.n, args.repo)
    except Exception as e:
        print(f"lerobot build failed: {e}", file=sys.stderr)
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for ex in examples:
            instr = ""
            resp = ""
            refs: list[str] = []
            for m in ex.messages:
                if m.role == "user":
                    for part in m.parts():
                        t = getattr(part, "text", None)
                        if t:
                            instr = t
                        ref = getattr(part, "ref", None)
                        if ref:
                            refs.append(str(ref))
                if m.role == "assistant":
                    for part in m.parts():
                        t = getattr(part, "text", None)
                        if t:
                            resp = t
                    if not resp and isinstance(m.content, str):
                        resp = m.content
            if not instr or not resp or not refs:
                continue
            row = {
                "instruction": instr,
                "images": refs,
                "response": resp,
                "dataset": args.repo.replace("/", "_"),
                "episode_id": f"{args.repo}:{n}",
                "license": "LeRobot hub — check dataset card before redistribute",
                "source": "lerobot",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows → {out}")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
