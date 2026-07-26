#!/usr/bin/env python3
"""Short VLM LoRA SFT on open robotics-style vision data (lab host).

Default data: a few **LeRobot** hub frames (or local JSONL) converted to Anvil
``Example`` rows with ``cas://`` images. Runs CE LoRA on Qwen2.5-VL-3B with
vision encoder frozen.

Examples::

  # forge — synthetic solid frames if dataset pull fails
  python scripts/robot_vlm_sft_demo.py \\
    --endpoint local:// \\
    --model /mnt/data/models/Qwen2.5-VL-3B-Instruct \\
    --media-root /mnt/data/anvil-media \\
    --export /mnt/data/anvil-runs/robot-vlm-demo \\
    --steps 20

  # prefer real LeRobot images (downloads via huggingface_hub)
  python scripts/robot_vlm_sft_demo.py --source lerobot --n-examples 8 --steps 30

  # your converted JSONL (instruction/images/response fields)
  python scripts/robot_vlm_sft_demo.py --jsonl /mnt/data/datasets/anvil_jsonl/bridge_smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _make_png(path: Path, rgb: tuple[int, int, int]) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 128), rgb).save(path, format="PNG")


def _synthetic_robot_examples(store, n: int):
    """Tabletop-style color scenes with short action/instruction labels."""
    from anvil.data.ingest import put_images_from_paths
    from anvil.protocol.messages import Example, ImagePart, Message, TextPart

    # (rgb, instruction, response) — simple verifiable vision+language targets
    catalog = [
        ((30, 90, 200), "What color is the block on the table?", "blue"),
        ((200, 40, 40), "What color is the block on the table?", "red"),
        ((40, 160, 50), "What color is the block on the table?", "green"),
        ((220, 180, 40), "What color is the block on the table?", "yellow"),
        ((30, 90, 200), "Is the gripper above a blue block? Answer yes or no.", "yes"),
        ((200, 40, 40), "Is the gripper above a blue block? Answer yes or no.", "no"),
        ((40, 160, 50), "Name the object color for pick-and-place.", "green"),
        ((220, 180, 40), "Name the object color for pick-and-place.", "yellow"),
    ]
    examples = []
    root = Path(store.root) / "_robot_demo"
    for i in range(n):
        rgb, instruction, response = catalog[i % len(catalog)]
        path = root / f"frame_{i:03d}.png"
        _make_png(path, rgb)
        refs = put_images_from_paths(store, [path])
        examples.append(
            Example(
                messages=(
                    Message(
                        role="user",
                        content=(
                            TextPart(text=instruction),
                            ImagePart(ref=refs[0], detail="low"),
                        ),
                    ),
                    Message(role="assistant", content=response),
                ),
                meta={"source": "synthetic_tabletop", "i": i},
            )
        )
    return examples


def _examples_from_jsonl(store, jsonl_path: Path, n: int):
    from anvil.data.ingest import examples_from_vlm_jsonl

    exs = examples_from_vlm_jsonl(jsonl_path, store, limit=n)
    if not exs:
        raise SystemExit(f"no examples loaded from {jsonl_path}")
    return exs


def _ffmpeg_extract_frames(video: Path, out_dir: Path, n: int) -> list[Path]:
    """Extract up to ``n`` PNGs from a video via ffmpeg (1 fps sample)."""
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.png"):
        old.unlink()
    pattern = out_dir / "frame_%04d.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            "fps=1",
            str(pattern),
        ],
        check=True,
        capture_output=True,
    )
    frames = sorted(out_dir.glob("frame_*.png"))[:n]
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames from {video}")
    return frames


def _examples_from_lerobot(store, n: int, repo: str):
    """Pull a few RGB frames + task text from a LeRobot hub dataset."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:
        raise SystemExit("huggingface_hub required for --source lerobot") from e

    from PIL import Image

    from anvil.data.ingest import put_images_from_paths
    from anvil.protocol.messages import Example, ImagePart, Message, TextPart

    files = list_repo_files(repo, repo_type="dataset")
    task = "complete the robot manipulation task shown in the camera frame"
    info_path = next(
        (f for f in files if f.endswith("info.json") or f == "meta/info.json"),
        None,
    )
    if info_path:
        local_info = hf_hub_download(repo, info_path, repo_type="dataset")
        try:
            info = json.loads(Path(local_info).read_text(encoding="utf-8"))
            # Prefer an explicit task / total_episodes note when present
            if isinstance(info.get("total_episodes"), int):
                task = (
                    "You are a robot policy assistant. Given the camera frame, "
                    "state the high-level skill being demonstrated in a few words."
                )
        except Exception:
            pass

    image_files = [
        f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    video_files = [f for f in files if f.lower().endswith(".mp4")]
    cache = Path(store.root) / "_lerobot_dl" / repo.replace("/", "__")
    cache.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    if image_files:
        for i, rel in enumerate(image_files[:n]):
            local = hf_hub_download(repo, rel, repo_type="dataset")
            img = Image.open(local).convert("RGB")
            out = cache / f"still_{i:03d}.png"
            img.save(out, format="PNG")
            frame_paths.append(out)
    elif video_files:
        rel = video_files[0]
        print(f"downloading LeRobot video {repo}/{rel} …", file=sys.stderr)
        local = Path(hf_hub_download(repo, rel, repo_type="dataset"))
        frame_paths = _ffmpeg_extract_frames(local, cache / "frames", n)
    else:
        print(
            f"lerobot repo {repo!r} has no images/video; using synthetic",
            file=sys.stderr,
        )
        return _synthetic_robot_examples(store, n)

    # Varied short targets so CE is not a single string collapse.
    responses = [
        "reach toward the object",
        "grasp the object",
        "lift the object",
        "place the object",
        "retract the arm",
        "align with the target",
        "close the gripper",
        "open the gripper",
    ]
    examples = []
    for i, path in enumerate(frame_paths[:n]):
        refs = put_images_from_paths(store, [path])
        response = responses[i % len(responses)]
        instruction = (
            f"{task} Reply with one short action phrase."
        )
        examples.append(
            Example(
                messages=(
                    Message(
                        role="user",
                        content=(
                            TextPart(text=instruction),
                            ImagePart(ref=refs[0], detail="low"),
                        ),
                    ),
                    Message(role="assistant", content=response),
                ),
                meta={"source": "lerobot", "repo": repo, "frame": str(path.name)},
            )
        )
    print(f"loaded {len(examples)} frames from {repo}")
    return examples


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="local://")
    p.add_argument(
        "--model",
        default="/mnt/data/models/Qwen2.5-VL-3B-Instruct",
        help="local snapshot path or HF id",
    )
    p.add_argument("--media-root", default="/mnt/data/anvil-media")
    p.add_argument("--export", default="/mnt/data/anvil-runs/robot-vlm-demo")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--n-examples", type=int, default=8)
    p.add_argument(
        "--source",
        choices=("synthetic", "lerobot", "jsonl"),
        default="synthetic",
        help="synthetic color blocks (default), LeRobot hub stills, or --jsonl",
    )
    p.add_argument(
        "--lerobot-repo",
        default="lerobot/pusht",
        help="HF dataset repo for --source lerobot",
    )
    p.add_argument("--jsonl", default=None, help="path for --source jsonl")
    p.add_argument("--rank", type=int, default=16)
    args = p.parse_args(argv)

    from anvil.media import LocalMediaStore
    from anvil.recipes.vlm_sft import run_vlm_sft

    store = LocalMediaStore(args.media_root)
    if args.source == "jsonl":
        if not args.jsonl:
            raise SystemExit("--jsonl path required for --source jsonl")
        examples = _examples_from_jsonl(store, Path(args.jsonl), args.n_examples)
    elif args.source == "lerobot":
        try:
            examples = _examples_from_lerobot(store, args.n_examples, args.lerobot_repo)
        except Exception as e:
            print(f"lerobot pull failed ({e}); falling back to synthetic", file=sys.stderr)
            examples = _synthetic_robot_examples(store, args.n_examples)
    else:
        examples = _synthetic_robot_examples(store, args.n_examples)

    print(
        f"examples={len(examples)} steps={args.steps} model={args.model} "
        f"endpoint={args.endpoint}"
    )
    result = run_vlm_sft(
        base_model=args.model,
        examples=examples,
        steps=args.steps,
        endpoint=args.endpoint,
        export_dir=args.export,
        fetch_remote=False,
        media_store=store,
        overrides={"rank": args.rank},
    )
    print(
        f"steps={result.steps_run} adapter={result.adapter_id} "
        f"export={result.export_path}"
    )
    print("losses:", [round(x, 4) for x in result.losses])
    if result.losses and len(result.losses) >= 2:
        delta = result.losses[0] - result.losses[-1]
        print(
            f"loss first→last: {result.losses[0]:.4f} → {result.losses[-1]:.4f} "
            f"(Δ {delta:+.4f})"
        )
        if delta > 0.05:
            print("status: learning signal OK (loss dropped)")
        else:
            print("status: weak/flat loss — check pixels + freeze policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
