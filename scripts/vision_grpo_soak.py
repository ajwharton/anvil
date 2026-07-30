#!/usr/bin/env python3
"""Forge vision-rubric GRPO soak under live observe (Phase 4.B residual).

Runs multi-step on-policy GRPO with **vision rubrics** (keyword/hazard golds)
and real HF tokenize/detokenize. Metrics land under ``ANVIL_OBSERVE_ROOT``.

Lab host (forge)::

  /mnt/data/anvil-venv/bin/python scripts/vision_grpo_soak.py \\
    --endpoint local:// \\
    --model /mnt/data/models/qwen2.5-1.5b-instruct \\
    --steps 40 --group-size 4 \\
    --run-id vision-grpo-soak-$(date +%Y%m%d-%H%M%S)

Notes
-----
* LocalBackend sample is still **text-token** generation; image refs on
  rollouts are schema/observe signals until pixel-fused sample lands.
  Rubrics score detokenized completions (scene/hazard keywords).
* Do **not** run this on j30 — train on forge, keep edge storage clean.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _default_model() -> str:
    for p in (
        "/mnt/data/models/qwen2.5-1.5b-instruct",
        "/mnt/data/models/Qwen2.5-1.5B-Instruct",
        "/mnt/data/models/Qwen2.5-VL-3B-Instruct",
    ):
        if Path(p).is_dir():
            return p
    return "Qwen/Qwen2.5-1.5B-Instruct"


def _soak_rollouts():
    """Scene / hazard rubrics aligned with house-robot caption style."""
    from anvil.recipes.vision_rewards import RubricCriterion
    from anvil.recipes.vision_rl import VisionRollout

    dig = "f" * 64
    ref = f"cas://sha256/{dig}.png"  # schema placeholder; sample is text-token today
    return [
        VisionRollout(
            id="kitchen_caption",
            instruction=(
                "You are a mobile home robot. Describe the room in one short sentence. "
                "Mention kitchen fixtures if present (stove, cabinets, table)."
            ),
            image_refs=(ref,),
            required_keywords=("kitchen",),
            any_of_keywords=("stove", "cabinet", "table", "chair"),
        ),
        VisionRollout(
            id="chair_hazard",
            instruction=(
                "You are a mobile home robot. Name the main furniture obstacle or hazard "
                "in one short sentence. Prefer the word chair if a chair is present."
            ),
            image_refs=(ref,),
            required_keywords=("chair",),
            any_of_keywords=("table", "obstacle", "furniture"),
        ),
        VisionRollout(
            id="safe_next",
            instruction=(
                "You are a mobile home robot. In one short sentence, state a safe next action "
                "when a chair is in the path (e.g. stop, reassess, avoid)."
            ),
            image_refs=(ref,),
            rubric=(
                RubricCriterion(
                    name="caution",
                    weight=1.0,
                    required_keywords=("stop", "avoid", "reassess", "slow", "wait"),
                ),
                RubricCriterion(
                    name="mention_chair",
                    weight=0.5,
                    required_keywords=("chair",),
                ),
            ),
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Vision-rubric GRPO soak (forge)")
    p.add_argument("--endpoint", default="local://")
    p.add_argument("--model", default=None)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--observe-root",
        default=os.environ.get("ANVIL_OBSERVE_ROOT", "/mnt/data/anvil-observe"),
    )
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--early-stop-patience", type=int, default=12)
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument(
        "--modalities",
        default="text",
        help="comma list: text or text,image (image needs VLM base; sample still text-token v0)",
    )
    args = p.parse_args(argv)

    from anvil.recipes.verifiable import detokenize_via_tokenizer
    from anvil.recipes.vision_rl import run_vision_grpo

    model = args.model or _default_model()
    run_id = args.run_id or f"vision-grpo-soak-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.observe_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer for real ids + detok
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers required on forge (anvil-venv)", file=sys.stderr)
        return 2

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(text: str) -> list[int]:
        return list(tok.encode(text, add_special_tokens=True))

    detok = detokenize_via_tokenizer(tok)
    mods = tuple(m.strip() for m in args.modalities.split(",") if m.strip()) or ("text",)

    print(
        json.dumps(
            {
                "event": "soak_start",
                "model": model,
                "endpoint": args.endpoint,
                "steps": args.steps,
                "group_size": args.group_size,
                "run_dir": str(run_dir),
                "modalities": list(mods),
            },
            indent=2,
        ),
        flush=True,
    )
    t0 = time.time()
    res = run_vision_grpo(
        rollouts=_soak_rollouts(),
        base_model=model,
        group_size=args.group_size,
        steps=args.steps,
        endpoint=args.endpoint,
        run_dir=str(run_dir),
        detokenize=detok,
        tokenize=tokenize,
        overrides={
            "rank": args.rank,
            "learning_rate": args.lr,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        early_stop=not args.no_early_stop,
        early_stop_patience=args.early_stop_patience,
        stop_on_southward=True,
        checkpoint_every=args.checkpoint_every,
        modalities=mods,
        fetch_remote=False,
    )
    wall = time.time() - t0
    summary = {
        "event": "soak_done",
        "run_dir": str(run_dir),
        "adapter_id": res.adapter_id,
        "steps_run": res.steps_run,
        "mean_reward": res.mean_reward,
        "losses": res.losses,
        "early_stop_reason": res.early_stop_reason,
        "checkpoint_path": res.checkpoint_path,
        "wall_s": round(wall, 1),
        "steps_per_hour": round(res.steps_run / wall * 3600, 1) if wall > 0 else None,
    }
    (run_dir / "soak_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
