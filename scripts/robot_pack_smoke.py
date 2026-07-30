#!/usr/bin/env python3
"""House robot pack → action bins → robot_offline (Track 1 smoke).

CI/laptop uses a synthetic j30-like pack. Lab/robot: point ``--pack`` at a
real episode directory tree (frames + meta.json).

Examples::

  python scripts/robot_pack_smoke.py --steps 2
  python scripts/robot_pack_smoke.py --pack /path/to/house_pack --steps 50 \\
    --endpoint local:// --model HuggingFaceTB/SmolVLM-256M-Instruct
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
    p = argparse.ArgumentParser(description="House robot pack → robot_offline smoke")
    p.add_argument("--pack", default=None, help="episode pack root (default: write demo)")
    p.add_argument("--demo-dir", default=None, help="where to write synthetic pack")
    p.add_argument("--media-root", default=None)
    p.add_argument("--jsonl-out", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--endpoint", default="fake://")
    p.add_argument("--model", default=None)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--action-scheme", choices=("bins", "continuous"), default="bins")
    p.add_argument("--export-dir", default=None)
    p.add_argument("--export-format", default="peft", help="peft|gguf|onnx|merged_hf")
    p.add_argument("--fetch-remote", action="store_true")
    args = p.parse_args(argv)

    from anvil.data.robot_pack import (
        HousePackConfig,
        house_pack_to_jsonl,
        house_pack_to_trajectories,
        write_demo_house_pack,
    )
    from anvil.protocol.action_tokens import ActionTokenizer
    from anvil.recipes.robot_offline import DEFAULT_ROBOT_BASE, run_robot_offline

    if args.pack:
        pack_root = Path(args.pack)
    else:
        demo = Path(args.demo_dir) if args.demo_dir else Path("/tmp/anvil-house-pack-demo")
        pack_root = write_demo_house_pack(demo, n_episodes=4, frames_per=3)
        print(f"wrote demo house pack → {pack_root}")

    media = Path(args.media_root) if args.media_root else pack_root / "_media"
    cfg = HousePackConfig(
        source=pack_root,
        media_root=media,
        action_scheme=args.action_scheme,
        dataset="house_robot",
        robot_id="j30",
    )
    tok = ActionTokenizer(
        scheme=args.action_scheme,  # type: ignore[arg-type]
        n_bins=256,
    )
    result = house_pack_to_trajectories(cfg, tokenizer=tok)
    print(
        f"pack: episodes={result.n_episodes} steps={result.n_steps} "
        f"images={result.n_images}"
    )
    if result.n_episodes == 0:
        print("no episodes — check pack layout", file=sys.stderr)
        return 2

    if args.jsonl_out:
        house_pack_to_jsonl(cfg, Path(args.jsonl_out), tokenizer=tok)
        print(f"jsonl → {args.jsonl_out}")

    run_dir = args.run_dir or str(Path("/tmp") / "anvil-robot-pack-run")
    res = run_robot_offline(
        base_model=args.model or DEFAULT_ROBOT_BASE,
        trajectories=result.trajectories,
        steps=args.steps,
        endpoint=args.endpoint,
        run_dir=run_dir,
        action_tokenizer=tok,
        fetch_remote=args.fetch_remote,
        export_dir=args.export_dir,
        early_stop=False,
    )
    out = {
        "pack": str(pack_root),
        "n_train_examples": res.n_train_examples,
        "n_heldout_episodes": res.n_heldout_episodes,
        "steps_run": res.steps_run,
        "adapter_id": res.adapter_id,
        "run_dir": res.run_dir,
        "export_path": res.export_path,
        "losses": res.losses,
        "action_tokenizer": res.action_tokenizer,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
