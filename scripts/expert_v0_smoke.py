#!/usr/bin/env python3
"""Expert-v0 smoke — place data → VLM SFT under observe → export.

Closes the repeatable “ship one specialist” path without requiring Bridge on
the laptop. Lab: point ``--source`` at a real episode_pack (or converted JSONL).

Examples::

  # CI / laptop (fake backend, synthetic pack)
  python scripts/expert_v0_smoke.py --endpoint fake:// --max-rows 20 --steps 3

  # forge — real pack + VLM
  python scripts/expert_v0_smoke.py \\
    --endpoint local:// \\
    --source /mnt/data/datasets/bridge_v2/episode_pack \\
    --media-root /mnt/data/anvil-media \\
    --output-jsonl /mnt/data/datasets/anvil_jsonl/bridge_1k.jsonl \\
    --max-rows 1000 \\
    --model /mnt/data/models/Qwen2.5-VL-3B-Instruct \\
    --steps 50 \\
    --run-id expert-v0-bridge-1k \\
    --export /mnt/data/anvil-runs/expert-v0-bridge-1k

Checklist: ``docs/expert-v0-smoke.md``
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="fake://")
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="base model id or local path",
    )
    p.add_argument(
        "--source",
        default=None,
        help="episode_pack dir (default: write synthetic --demo pack)",
    )
    p.add_argument(
        "--kind",
        choices=("episode_pack", "path_jsonl"),
        default="episode_pack",
    )
    p.add_argument(
        "--media-root",
        default=None,
        help="CAS root (default: ~/.anvil/media or lab path)",
    )
    p.add_argument(
        "--output-jsonl",
        default=None,
        help="converted Anvil JSONL path",
    )
    p.add_argument("--max-rows", type=int, default=32)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--frames-per-episode", type=int, default=2)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument(
        "--run-id",
        default=None,
        help="observe run id under ANVIL_OBSERVE_ROOT",
    )
    p.add_argument("--observe-root", default=None)
    p.add_argument(
        "--export",
        default=None,
        help="PEFT export dir (default under ~/.anvil/runs/<run-id>)",
    )
    p.add_argument(
        "--holdout",
        type=int,
        default=2,
        help="held-out examples used as live probes (0 disables)",
    )
    p.add_argument("--probe-every", type=int, default=1)
    p.add_argument(
        "--dataset",
        default="expert_v0",
        help="dataset tag on converted rows",
    )
    p.add_argument(
        "--skip-convert",
        action="store_true",
        help="use --output-jsonl as already-converted Anvil JSONL",
    )
    p.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="cap train+holdout examples after load (safe full-batch size)",
    )
    p.add_argument(
        "--early-stop-mode",
        choices=("production", "calibration"),
        default="production",
        help="production: plateau early-stop; calibration: overshoot to map plateaus",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="override consecutive non-improve steps (default: type prior)",
    )
    p.add_argument(
        "--no-early-stop",
        action="store_true",
        help="disable early-stop (same as calibration with full --steps)",
    )
    p.add_argument(
        "--promote-recipe",
        default=None,
        help="save personal recipe id into ANVIL_RECIPE_BOOK after run",
    )
    p.add_argument(
        "--recipe-family",
        default=None,
        help="family tag for promoted recipe (e.g. Qwen2.5-VL)",
    )
    p.add_argument(
        "--scan-southward",
        action="store_true",
        default=True,
        help="scan metrics/probes for southward flags after train (default on)",
    )
    p.add_argument(
        "--no-scan-southward",
        action="store_true",
        help="disable southward scan",
    )
    p.add_argument(
        "--fail-on-southward",
        action="store_true",
        help="exit non-zero if any cliff-severity southward flag fires",
    )
    p.add_argument(
        "--run-meta",
        action="store_true",
        help="after train, run a tiny meta-recipe (sft→export) executor for dogfood",
    )
    args = p.parse_args(argv)

    from anvil.data.convert import ConvertConfig, convert_corpus, write_demo_episode_pack
    from anvil.data.ingest import examples_from_vlm_jsonl
    from anvil.media import LocalMediaStore
    from anvil.recipes.vlm_sft import run_vlm_sft

    media_root = Path(
        args.media_root
        or os.environ.get("ANVIL_MEDIA_ROOT")
        or (Path.home() / ".anvil" / "media")
    )
    observe_root = Path(
        args.observe_root
        or os.environ.get("ANVIL_OBSERVE_ROOT")
        or (Path.home() / ".anvil" / "observe")
    )
    run_id = args.run_id or "expert-v0-smoke"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}", run_id):
        raise SystemExit(f"bad run-id {run_id!r}")
    run_dir = observe_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = Path(
        args.output_jsonl
        or (media_root.parent / "anvil_jsonl" / f"{run_id}.jsonl")
    )
    export_dir = Path(
        args.export
        or (Path.home() / ".anvil" / "runs" / run_id / "adapter")
    )

    # --- place data ----------------------------------------------------------
    if args.skip_convert:
        if not out_jsonl.is_file():
            raise SystemExit(f"--skip-convert requires existing JSONL: {out_jsonl}")
        print(f"using existing jsonl={out_jsonl}")
    else:
        if args.source:
            source = Path(args.source)
            kind = args.kind
        else:
            source = media_root / "_expert_v0_demo_pack"
            print(f"writing demo episode_pack → {source}")
            write_demo_episode_pack(source, n_episodes=max(4, args.holdout + 2), frames_per=2)
            kind = "episode_pack"

        print(f"convert kind={kind} source={source}")
        result = convert_corpus(
            ConvertConfig(
                source=source,
                media_root=media_root,
                output_jsonl=out_jsonl,
                source_kind=kind,
                max_rows=args.max_rows,
                max_episodes=args.max_episodes,
                frames_per_episode=args.frames_per_episode,
                dataset=args.dataset,
                license_note="expert-v0-smoke",
                resume=False,
            )
        )
        print(
            f"convert: rows={result.n_rows} episodes={result.n_episodes} "
            f"jsonl={result.output_jsonl}"
        )

    store = LocalMediaStore(media_root)
    examples = examples_from_vlm_jsonl(out_jsonl, store)
    if not examples:
        raise SystemExit(f"no examples loaded from {out_jsonl}")
    if args.max_train is not None and args.max_train > 0:
        examples = examples[: args.max_train]

    holdout_n = max(0, min(args.holdout, len(examples) - 1))
    if holdout_n > 0 and len(examples) > holdout_n:
        train_ex = examples[:-holdout_n]
        probe_ex = examples[-holdout_n:]
    else:
        train_ex = examples
        probe_ex = examples[:1] if examples and args.holdout else []

    print(
        f"train_examples={len(train_ex)} probes={len(probe_ex)} "
        f"steps={args.steps} endpoint={args.endpoint} "
        f"early_stop_mode={args.early_stop_mode}"
    )
    print(f"observe → /observe/{run_id}  root={observe_root}")
    print(f"export → {export_dir}")

    # --- train under observe -------------------------------------------------
    sft = run_vlm_sft(
        base_model=args.model,
        examples=train_ex,
        steps=args.steps,
        endpoint=args.endpoint,
        export_dir=str(export_dir),
        fetch_remote=False,
        media_store=store if not args.endpoint.startswith("fake://") else None,
        overrides={"rank": args.rank},
        run_dir=str(run_dir),
        probes=probe_ex or None,
        probe_every=args.probe_every,
        early_stop=False if args.no_early_stop else None,
        early_stop_mode=args.early_stop_mode,
        early_stop_patience=args.early_stop_patience,
    )

    metrics = run_dir / "metrics.jsonl"
    probes_path = run_dir / "probes.jsonl"
    print(
        f"done: steps={sft.steps_run} adapter={sft.adapter_id} "
        f"export={sft.export_path} probe_records={sft.n_probe_records}"
    )
    if sft.early_stop_reason:
        print(f"early_stop: {sft.early_stop_reason}")
    # print compact loss summary (full list is huge on long runs)
    if sft.losses:
        print(
            f"losses: first={sft.losses[0]:.4f} last={sft.losses[-1]:.4f} "
            f"min={min(sft.losses):.4f} n={len(sft.losses)}"
        )
    print(f"metrics: {metrics}")
    if probes_path.is_file():
        print(f"probes:  {probes_path}")
    print(f"LIVE UI: /observe/{run_id}")
    print(f"  ANVIL_OBSERVE_ROOT={observe_root} anvil-web --host 0.0.0.0 --port 7600")
    print(
        "MCP: anvil_observe_metrics / anvil_observe_probes "
        f'with run_id="{run_id}"'
    )

    if args.promote_recipe:
        from anvil.recipes.book import RecipeBook, promote_from_run

        book = RecipeBook()
        rec = promote_from_run(
            recipe_id=args.promote_recipe,
            title=args.promote_recipe,
            run_dir=run_dir,
            run_id=run_id,
            pattern="vlm_sft",
            family=args.recipe_family or "Qwen2.5-VL",
            job="vlm_sft",
            knobs={"rank": args.rank, "base_model": args.model, "steps_cap": args.steps},
            stop_policy={
                "mode": args.early_stop_mode,
                "patience": args.early_stop_patience,
                "early_stop": not args.no_early_stop,
            },
            notes=f"promoted from expert_v0_smoke endpoint={args.endpoint}",
            tags=["expert_v0", "auto_promote"],
            book=book,
            early_stop_reason=sft.early_stop_reason,
        )
        print(f"recipe_book: saved {rec.id} → {book.root / (rec.id + '.json')}")

    southward_cliff = False
    if args.scan_southward and not args.no_scan_southward and metrics.is_file():
        from anvil.observe.southward import scan_and_log

        rep = scan_and_log(run_dir)
        if rep.flags:
            print(f"southward: {[f.name for f in rep.flags]} ok={rep.ok}")
            for f in rep.flags:
                print(f"  - {f.severity} {f.name}: {f.detail}")
            southward_cliff = not rep.ok
        else:
            print("southward: no flags")

    if args.run_meta:
        from anvil.recipes.meta import example_vlm_ladder
        from anvil.recipes.meta_runners import DefaultRunnerConfig, run_meta_with_defaults

        # Default live runners (not injectable stubs). Short budgets for smoke.
        meta_dir = run_dir / "meta_exec"
        cfg = DefaultRunnerConfig(
            endpoint=args.endpoint,
            base_model=args.model,
            run_dir=meta_dir,
            sft_steps=min(30, max(sft.steps_run, 8)),
            vlm_steps=min(30, max(sft.steps_run, 8)),
            early_stop_patience=12,
            stop_on_southward=False,
            export_root=meta_dir / "export",
            fetch_remote=False,
        )
        # Prefer the same base the expert smoke already used
        base = getattr(sft.plan, "base_model", None) or cfg.base_model
        cfg.base_model = base
        meta_res = run_meta_with_defaults(
            example_vlm_ladder(family=args.recipe_family or "Qwen2.5-VL"),
            config=cfg,
        )
        print(
            f"meta_exec: stages={meta_res.stages_run} stop={meta_res.stopped_reason} "
            f"dir={meta_dir} (default live runners)"
        )

    if not sft.export_path:
        print("warn: no export path", file=sys.stderr)
        return 1
    if not metrics.is_file():
        print("warn: metrics.jsonl missing", file=sys.stderr)
        return 1
    if args.fail_on_southward and southward_cliff:
        print("fail: southward cliff flags present", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
