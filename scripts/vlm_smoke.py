#!/usr/bin/env python3
"""P3.3/P3.6 forge smoke — VLM SFT with real frames + optional live observe.

Creates a tiny PNG (or uses --image paths), CAS-stores them, builds Examples,
runs a few CE steps via fake:// (always) or local:// (needs GPU + VLM weights).
With ``--run-id``, writes ``metrics.jsonl`` under ANVIL_OBSERVE_ROOT for
``/observe/<run_id>``.

Examples::

  # CI / laptop (fake backend, toy renderer path)
  python scripts/vlm_smoke.py --endpoint fake:// --steps 2

  # live observe (fake path)
  python scripts/vlm_smoke.py --endpoint fake:// --steps 5 --run-id vlm-smoke-demo

  # forge with Qwen2.5-VL-3B
  python scripts/vlm_smoke.py \\
    --endpoint local:// \\
    --model /mnt/data/models/Qwen2.5-VL-3B-Instruct \\
    --media-root /mnt/data/anvil-media \\
    --steps 5 \\
    --run-id vlm-forge-smoke \\
    --export /mnt/data/anvil-runs/vlm-smoke-out
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def _make_png(path: Path, rgb: tuple[int, int, int] = (40, 120, 200)) -> None:
    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit("Pillow required: pip install pillow") from e
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), rgb).save(path, format="PNG")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="fake://")
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="base model id or local path",
    )
    p.add_argument("--media-root", default=str(Path.home() / ".anvil" / "media"))
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--export", default=None, help="optional PEFT export dir")
    p.add_argument(
        "--image",
        action="append",
        default=[],
        help="optional image path (repeatable); default: generate a solid PNG",
    )
    p.add_argument(
        "--use-vlm-renderer",
        action="store_true",
        help="force HFVLMRenderer (needs processor weights for --model)",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="observe run id → ANVIL_OBSERVE_ROOT/<run-id>/metrics.jsonl",
    )
    p.add_argument(
        "--observe-root",
        default=None,
        help="override ANVIL_OBSERVE_ROOT (default: env or ~/.anvil/observe)",
    )
    args = p.parse_args(argv)

    from anvil.data.ingest import put_images_from_paths
    from anvil.media import LocalMediaStore
    from anvil.protocol.messages import Example, ImagePart, Message, TextPart
    from anvil.recipes.vlm_sft import run_vlm_sft

    run_dir = None
    if args.run_id:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}", args.run_id):
            raise SystemExit(f"bad run-id {args.run_id!r}")
        root = Path(
            args.observe_root
            or os.environ.get("ANVIL_OBSERVE_ROOT")
            or (Path.home() / ".anvil" / "observe")
        )
        run_dir = root / args.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"observe → /observe/{args.run_id}  (root={root})")

    store = LocalMediaStore(args.media_root)
    paths = [Path(x) for x in args.image]
    if not paths:
        tmp = Path(args.media_root) / "_smoke" / "frame.png"
        _make_png(tmp)
        paths = [tmp]
    refs = put_images_from_paths(store, paths)
    print(f"media refs: {refs}")

    ex = Example(
        messages=(
            Message(
                role="user",
                content=(
                    TextPart(text="Is this a solid color test pattern? Answer yes or no."),
                    ImagePart(ref=refs[0], detail="low"),
                ),
            ),
            Message(role="assistant", content="yes"),
        ),
        meta={"source": "vlm_smoke", "dataset": "synthetic"},
    )

    renderer = None
    media_store = store
    if args.endpoint.startswith("fake://") and not args.use_vlm_renderer:
        # Toy path: no multi-GB processor
        media_store = None
        print("endpoint fake:// → ToyTextRenderer (image ref as text placeholder)")
    elif args.use_vlm_renderer or args.endpoint.startswith("local://"):
        print(f"using HFVLMRenderer for {args.model}")
        media_store = store

    result = run_vlm_sft(
        base_model=args.model,
        examples=[ex],
        steps=args.steps,
        endpoint=args.endpoint,
        export_dir=args.export,
        fetch_remote=False,
        media_store=media_store,
        renderer=renderer,
        run_dir=str(run_dir) if run_dir is not None else None,
    )
    print(
        f"steps={result.steps_run} losses={result.losses} "
        f"adapter={result.adapter_id} export={result.export_path}"
    )
    if result.losses and len(result.losses) >= 2:
        print(
            "loss trend:",
            "down" if result.losses[-1] < result.losses[0] else "flat/up",
        )
    if run_dir is not None:
        metrics = run_dir / "metrics.jsonl"
        print(f"metrics: {metrics}")
        print(f"LIVE UI: /observe/{args.run_id}")
        print(f"  ANVIL_OBSERVE_ROOT={run_dir.parent} anvil-web --host 0.0.0.0 --port 7600")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
