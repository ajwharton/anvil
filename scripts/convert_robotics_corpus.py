#!/usr/bin/env python3
"""Convert robotics frames → Anvil CAS + JSONL (Phase 3.B).

Does **not** download Bridge/OXE (multi‑GB). Point ``--source`` at a lab-side
**episode pack** (or path-based JSONL) and emit Anvil-shaped rows with
``cas://`` image refs under a media root.

Episode pack layout (Bridge-like intermediate)::

  source/
    ep_0001/
      meta.json          # language_instruction, actions[], optional license
      frames/0000.jpg
      frames/0001.jpg
    ep_0002/
      ...

Path JSONL (images as local paths)::

  {"instruction": "…", "images": ["frames/a.jpg"], "response": "…"}

Examples::

  # CI / laptop — synthetic pack
  python scripts/convert_robotics_corpus.py --demo --max-rows 12 \\
    --media-root /tmp/anvil-media --output /tmp/anvil_jsonl/demo.jsonl

  # lab — up to 1k rows from an extracted Bridge episode pack
  python scripts/convert_robotics_corpus.py \\
    --source /mnt/data/datasets/bridge_v2/episode_pack \\
    --media-root /mnt/data/anvil-media \\
    --output /mnt/data/datasets/anvil_jsonl/bridge_1k.jsonl \\
    --dataset bridge_v2 \\
    --license "BridgeData V2 — check RAIL terms before redistribute" \\
    --max-rows 1000 \\
    --frames-per-episode 4

  # resume after interrupt (default)
  python scripts/convert_robotics_corpus.py ...   # same args; skips done keys

  # train on converted JSONL
  python scripts/robot_vlm_sft_demo.py --source jsonl \\
    --jsonl /mnt/data/datasets/anvil_jsonl/bridge_1k.jsonl \\
    --run-id bridge-1k-sft --steps 50
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert robotics episode pack / path JSONL → Anvil CAS + JSONL",
    )
    p.add_argument(
        "--source",
        default=None,
        help="episode_pack directory or path_jsonl file",
    )
    p.add_argument(
        "--kind",
        choices=("episode_pack", "path_jsonl"),
        default="episode_pack",
        help="source layout (default: episode_pack)",
    )
    p.add_argument(
        "--media-root",
        default=str(Path.home() / ".anvil" / "media"),
        help="LocalMediaStore root for cas:// blobs",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        help="output Anvil JSONL path",
    )
    p.add_argument("--dataset", default="robotics", help="dataset tag on each row")
    p.add_argument(
        "--license",
        default=None,
        help="license/attribution string written on each row when meta omits it",
    )
    p.add_argument("--max-rows", type=int, default=None, help="cap emitted rows")
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="cap episodes (episode_pack only)",
    )
    p.add_argument(
        "--frames-per-episode",
        type=int,
        default=None,
        help="subsample frames per episode (evenly spaced)",
    )
    p.add_argument(
        "--row-mode",
        choices=("per_frame", "keyframe"),
        default="per_frame",
        help="one JSONL row per frame or one keyframe per episode",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore state file; truncate output and start clean",
    )
    p.add_argument(
        "--state",
        default=None,
        help="state file path (default: <output>.state.json)",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="write a tiny synthetic episode_pack under --demo-dir and convert it",
    )
    p.add_argument(
        "--demo-dir",
        default=None,
        help="where to place synthetic pack (default: <media-root>/_demo_episode_pack)",
    )
    p.add_argument(
        "--demo-episodes",
        type=int,
        default=5,
        help="synthetic episodes when --demo",
    )
    args = p.parse_args(argv)

    from anvil.data.convert import ConvertConfig, convert_corpus, write_demo_episode_pack

    media_root = Path(args.media_root)
    if args.demo:
        demo_dir = Path(args.demo_dir) if args.demo_dir else media_root / "_demo_episode_pack"
        print(f"writing demo episode_pack → {demo_dir}")
        write_demo_episode_pack(demo_dir, n_episodes=args.demo_episodes, frames_per=2)
        source = demo_dir
        kind = "episode_pack"
        if args.output is None:
            args.output = str(media_root.parent / "anvil_jsonl" / "demo_bridge_like.jsonl")
        if args.dataset == "robotics":
            args.dataset = "demo_bridge_like"
        if args.license is None:
            args.license = "synthetic-demo-not-bridge"
    else:
        if not args.source:
            p.error("--source is required unless --demo")
        source = Path(args.source)
        kind = args.kind

    if not args.output:
        p.error("--output is required (or use --demo for a default path)")

    cfg = ConvertConfig(
        source=source,
        media_root=media_root,
        output_jsonl=Path(args.output),
        source_kind=kind,
        max_rows=args.max_rows,
        max_episodes=args.max_episodes,
        frames_per_episode=args.frames_per_episode,
        row_mode=args.row_mode,
        dataset=args.dataset,
        license_note=args.license,
        resume=not args.no_resume,
        state_path=Path(args.state) if args.state else None,
    )
    print(f"kind={cfg.source_kind} source={cfg.source}")
    print(f"media_root={cfg.media_root}")
    print(f"output={cfg.output_jsonl}")
    if cfg.max_rows:
        print(f"max_rows={cfg.max_rows}")
    result = convert_corpus(cfg)
    print(
        f"done: rows_this_run={result.n_rows} episodes={result.n_episodes} "
        f"skipped={result.n_skipped} images_in_store≈{result.n_images}"
    )
    print(f"jsonl: {result.output_jsonl}")
    print(f"state: {result.state_path}")
    print(
        "next: python scripts/robot_vlm_sft_demo.py --source jsonl "
        f"--jsonl {result.output_jsonl} --media-root {result.media_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
