#!/usr/bin/env python3
"""Convert a **local** j30 ``vision/out`` tree → Anvil house episode pack.

**Ownership / storage (read this):**

- The j30 (and on-device ``~/vision``) is under **robotics project operational
  control**, not Anvil. Anvil only consumes **already-pulled** snapshots.
- The Orin is **storage- and RAM-constrained**. Logging frames is expensive.
  Prefer short bursts → rsync **off-device** → prune on the robot. Do **not**
  leave multi-GB capture trees, train run dirs, or HF caches on the device.
- This script never SSHes, never writes to the robot, and never stores
  credentials. Host IPs/passwords must not enter the Anvil git tree.

Operator flow (lab Mac / forge — not on the robot)::

  # 1) pull a *small* snapshot (robotics ops decides retention on device)
  rsync -avz --max-size=5m user@jetson:~/vision/out/ ./j30-out/
  # 2) convert locally
  python scripts/j30_vision_out_to_pack.py --source ./j30-out --out ./house_pack
  # 3) train offline on lab, not on j30
  python scripts/robot_pack_smoke.py --pack ./house_pack --steps 20
  # 4) optional: after pull, prune on-device out/ per robotics policy

Expected source layout (subset OK)::

  out/
    llm/rgb_*.jpg + see_*.txt
    loop/esc_*.jpg + esc_*.txt
    ab/f*_rgb.jpg
    live_pull/frame_*.jpg + captions.json + capture_meta.json
    frame_*.jpg
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _parse_see_txt(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\nVLM\n(.*?)(?:\n\nFOLLOWUP|\nFOLLOWUP|\Z)", text, re.S)
    if not m:
        return None
    cap = m.group(1).strip()
    if len(cap) < 8 or cap.lower().startswith("frame 0:"):
        return None
    return re.split(r"\n", cap)[0].strip()[:240]


def convert(source: Path, out: Path) -> list[str]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    episodes: list[str] = []

    llm = source / "llm"
    if llm.is_dir():
        for jpg in sorted(llm.glob("rgb_*.jpg")):
            stem = jpg.stem.replace("rgb_", "see_")
            txt = llm / f"{stem}.txt"
            cap = _parse_see_txt(txt) if txt.is_file() else None
            if not cap:
                continue
            ep_id = f"llm_{jpg.stem}"
            ep = out / ep_id
            (ep / "frames").mkdir(parents=True)
            shutil.copy2(jpg, ep / "frames" / "0000.jpg")
            meta = {
                "language_instruction": (
                    "Describe this room for a mobile robot in one short sentence."
                ),
                "captions": [cap],
                "source": "j30_llm_see",
                "robot_id": "j30",
            }
            (ep / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            episodes.append(ep_id)

    loop = source / "loop"
    if loop.is_dir():
        for jpg in sorted(loop.glob("esc_*.jpg")):
            txt = jpg.with_suffix(".txt")
            notes = "scene escalate"
            if txt.is_file():
                t = txt.read_text(encoding="utf-8", errors="replace")
                m = re.search(r"Safe Next Action:\s*(.+)", t)
                if m:
                    notes = m.group(1).strip()
                m2 = re.search(r"local:\s*(.+)", t)
                det = m2.group(1).strip() if m2 else "object"
            else:
                det = "object"
            ep = out / f"loop_{jpg.stem}"
            (ep / "frames").mkdir(parents=True)
            shutil.copy2(jpg, ep / "frames" / "0000.jpg")
            label = det.split(":")[0].strip() if det else "object"
            meta = {
                "language_instruction": (
                    "What hazard is present and what should the robot do?"
                ),
                "captions": [notes],
                "detections": [[{"label": label, "conf": 1.0}]],
                "source": "j30_loop_esc",
                "robot_id": "j30",
            }
            (ep / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            episodes.append(ep.name)

    ab = source / "ab"
    if ab.is_dir():
        for jpg in sorted(ab.glob("f*_rgb.jpg")):
            ep = out / f"ab_{jpg.stem}"
            (ep / "frames").mkdir(parents=True)
            shutil.copy2(jpg, ep / "frames" / "0000.jpg")
            meta = {
                "language_instruction": (
                    "Describe this room in one short sentence for a robot."
                ),
                "captions": ["indoor room view"],
                "source": "j30_ab_rgb",
                "robot_id": "j30",
            }
            (ep / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            episodes.append(ep.name)

    live = source / "live_pull"
    if live.is_dir():
        frames = sorted(live.glob("frame_*.jpg"))
        if frames:
            caps: dict[str, str] = {}
            cp = live / "captions.json"
            if cp.is_file():
                for c in json.loads(cp.read_text(encoding="utf-8")):
                    caps[str(c.get("frame"))] = str(c.get("caption") or "")
            det_by_i: dict[int, list] = {}
            mp = live / "capture_meta.json"
            if mp.is_file():
                for fr in json.loads(mp.read_text(encoding="utf-8")).get("frames", []):
                    det_by_i[int(fr["i"])] = fr.get("detections") or []
            ep = out / "live_pull"
            (ep / "frames").mkdir(parents=True)
            captions: list[str] = []
            detections: list = []
            for i, f in enumerate(frames):
                shutil.copy2(f, ep / "frames" / f"{i:04d}.jpg")
                captions.append(caps.get(f.name) or "scene")
                detections.append(det_by_i.get(i, []))
            meta = {
                "language_instruction": (
                    "Describe this room for a mobile robot in one short sentence."
                ),
                "captions": captions,
                "detections": detections,
                "source": "j30_live_pull",
                "robot_id": "j30",
                "license": "house-private-not-for-redistribution",
            }
            (ep / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            episodes.append(ep.name)

    for jpg in sorted(source.glob("frame_*.jpg")):
        ep = out / f"root_{jpg.stem}"
        (ep / "frames").mkdir(parents=True)
        shutil.copy2(jpg, ep / "frames" / "0000.jpg")
        meta = {
            "language_instruction": "Describe this room for a mobile robot.",
            "captions": ["house interior frame"],
            "source": "j30_root_frame",
            "robot_id": "j30",
        }
        (ep / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        episodes.append(ep.name)

    return episodes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="j30 vision/out → house pack")
    p.add_argument("--source", required=True, help="path to vision/out (local)")
    p.add_argument("--out", required=True, help="output house pack directory")
    args = p.parse_args(argv)
    eps = convert(Path(args.source), Path(args.out))
    print(json.dumps({"n_episodes": len(eps), "episodes": eps, "out": args.out}, indent=2))
    if not eps:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
