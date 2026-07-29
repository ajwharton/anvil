#!/usr/bin/env python3
"""Lab live smoke suite — periodic GPU/host checks, **not** GitHub CI.

CI stays lightweight (``[dev,web]`` + fake://). This suite is for forge/hammer
(or any machine with ``anvil-train[local]`` + weights) and can be cron'd.

Profiles::

  quick       — fake:// GRPO/SFT/DPO/meta/southward/expert (~seconds, no GPU)
  nightly     — quick + short real GRPO early-stop + 2-stage queue on 1.5B
  full        — nightly + VLM SFT smoke
  multi_hour  — quick + scale ladder demo + throughput defaults + resume contract
                (wall-clock multi-hour on forge: scale_ladder.py --no-demo)

Examples::

  # laptop / no GPU
  python scripts/lab_smokes.py --profile quick

  # Expert-v2 ops (scale ladder + multi-hour resume contract)
  python scripts/lab_smokes.py --profile multi_hour

  # forge nightly (cron-friendly)
  python scripts/lab_smokes.py --profile nightly --endpoint local:// \\
    --model /mnt/data/models/qwen2.5-1.5b-instruct \\
    --vlm-model /mnt/data/models/Qwen2.5-VL-3B-Instruct \\
    --report-dir /mnt/data/anvil-runs/lab-smokes

  # one smoke only
  python scripts/lab_smokes.py --only grpo_early_stop_local,rl_queue_local
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))



@dataclass
class SmokeResult:
    name: str
    ok: bool
    duration_s: float
    detail: str = ""
    skipped: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


SmokeFn = Callable[["SmokeContext"], SmokeResult]


@dataclass
class SmokeContext:
    endpoint: str
    model: str
    vlm_model: str
    observe_root: Path
    media_root: Path
    report_run: Path
    group_size: int
    dry_run: bool


def _now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _have_local_stack() -> bool:
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _model_exists(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "config.json").is_file()


# --- individual smokes -------------------------------------------------------


def smoke_fake_sft_early_stop(ctx: SmokeContext) -> SmokeResult:
    t0 = time.monotonic()
    from anvil.protocol.messages import Example, Message, TextPart
    from anvil.recipes.sft import run_sft

    ex = Example(
        messages=(
            Message(role="user", content=(TextPart(text="hi"),)),
            Message(role="assistant", content=(TextPart(text="yo"),)),
        )
    )
    run_dir = ctx.report_run / "fake-sft-early-stop"
    res = run_sft(
        endpoint="fake://",
        examples=[ex],
        steps=200,
        run_dir=str(run_dir),
        early_stop_mode="production",
        early_stop_patience=15,
    )
    ok = res.early_stop_reason is not None and res.steps_run < 200
    return SmokeResult(
        name="fake_sft_early_stop",
        ok=ok,
        duration_s=time.monotonic() - t0,
        detail=f"steps={res.steps_run} reason={res.early_stop_reason}",
        meta={"steps": res.steps_run, "early_stop_reason": res.early_stop_reason},
    )


def smoke_fake_dpo_observe(ctx: SmokeContext) -> SmokeResult:
    t0 = time.monotonic()
    from anvil.observe.metrics import read_jsonl
    from anvil.recipes.dpo import PreferencePair, run_dpo

    run_dir = ctx.report_run / "fake-dpo"
    hold = PreferencePair(prompt="2+2?", preferred="4", rejected="nope long")
    res = run_dpo(
        endpoint="fake://",
        steps=8,
        run_dir=str(run_dir),
        early_stop=False,
        stop_on_southward=False,
        probes=[hold],
        probe_every=2,
    )
    steps = read_jsonl(run_dir / "metrics.jsonl")
    probes = read_jsonl(run_dir / "probes.jsonl")
    ok = (
        res.steps_run == 8
        and steps
        and steps[0].get("job") == "dpo"
        and "length_bias" in steps[0]
        and res.n_probe_records >= 1
        and probes
    )
    return SmokeResult(
        name="fake_dpo_observe",
        ok=ok,
        duration_s=time.monotonic() - t0,
        detail=(
            f"steps={res.steps_run} length_bias={res.mean_length_bias} "
            f"probes={res.n_probe_records}"
        ),
        meta={
            "steps": res.steps_run,
            "length_bias": res.mean_length_bias,
            "n_probes": res.n_probe_records,
        },
    )


def smoke_fake_meta_exec(ctx: SmokeContext) -> SmokeResult:
    """Executor graph plumbing (injectable runner) — kept as a pure unit-ish smoke."""
    t0 = time.monotonic()
    from anvil.recipes.meta import MetaEdge, MetaRecipe, MetaStage
    from anvil.recipes.meta_exec import StageRunResult, run_meta_recipe

    meta = MetaRecipe(
        id="lab-smoke-meta",
        title="lab",
        stages=[
            MetaStage(id="a", recipe_id="r0"),
            MetaStage(id="b", recipe_id="r1"),
        ],
        edges=[MetaEdge(on="early_stop:*", from_stage="a", to_stage="b")],
    )
    calls: list[str] = []

    def runner(stage, *, step_index: int) -> StageRunResult:
        calls.append(stage.id)
        if stage.id == "a":
            return StageRunResult(signal="early_stop:loss_plateau_patience_40")
        return StageRunResult(signal="done")

    run_dir = ctx.report_run / "fake-meta"
    res = run_meta_recipe(meta, runner, run_dir=run_dir)
    ok = calls == ["a", "b"] and res.stages_run == 2
    return SmokeResult(
        name="fake_meta_exec",
        ok=ok,
        duration_s=time.monotonic() - t0,
        detail=f"stages={res.stages_run} stop={res.stopped_reason}",
        meta={"stages_run": res.stages_run},
    )


def smoke_fake_meta_live_runners(ctx: SmokeContext) -> SmokeResult:
    """Default live runners: SFT → GRPO on fake:// (Expert-v1 wire-up)."""
    t0 = time.monotonic()
    from anvil.recipes.meta import MetaEdge, MetaRecipe, MetaStage
    from anvil.recipes.meta_runners import DefaultRunnerConfig, run_meta_with_defaults

    meta = MetaRecipe(
        id="lab-smoke-meta-live",
        title="live SFT then GRPO",
        stages=[
            MetaStage(id="sft", recipe_id="sft_chat", pattern="sft_chat"),
            MetaStage(id="grpo", recipe_id="rl", pattern="rl_verifiable"),
        ],
        edges=[MetaEdge(on="early_stop:*", from_stage="sft", to_stage="grpo")],
    )
    run_dir = ctx.report_run / "fake-meta-live"
    cfg = DefaultRunnerConfig(
        endpoint=ctx.endpoint if ctx.endpoint.startswith("fake") else "fake://",
        run_dir=run_dir,
        sft_steps=40,
        grpo_steps=20,
        early_stop_patience=12,
        grpo_patience=5,
        stop_on_southward=False,
    )
    res = run_meta_with_defaults(meta, config=cfg)
    ok = (
        res.stages_run == 2
        and res.outcomes[0].advanced
        and (run_dir / "sft" / "metrics.jsonl").is_file()
        and (run_dir / "grpo" / "metrics.jsonl").is_file()
    )
    return SmokeResult(
        name="fake_meta_live_runners",
        ok=ok,
        duration_s=time.monotonic() - t0,
        detail=(
            f"stages={res.stages_run} stop={res.stopped_reason} "
            f"s0={res.outcomes[0].result.signal if res.outcomes else None}"
        ),
        meta={"stages_run": res.stages_run, "stopped_reason": res.stopped_reason},
    )


def smoke_fake_scale_ladder(ctx: SmokeContext) -> SmokeResult:
    """Expert-v2: demo scale ladder 1k→5k→50k (tiny demo_rows, same code path)."""
    t0 = time.monotonic()
    from anvil.recipes.scale_ladder import build_ladder_plan, exercise_ladder

    work = ctx.report_run / "scale-ladder"
    plan = build_ladder_plan(rungs=["all"], demo=True, dataset="demo_bridge_like")
    results = exercise_ladder(
        plan,
        work_dir=work,
        endpoint="fake://",
        train=True,
    )
    ok = len(results) == 3 and all(r.ok for r in results)
    detail = " ".join(f"{r.rung.id}={r.rows_converted}r/{r.steps_run}s" for r in results)
    return SmokeResult(
        name="fake_scale_ladder",
        ok=ok,
        duration_s=time.monotonic() - t0,
        detail=detail,
        meta={
            "rungs": [
                {
                    "id": r.rung.id,
                    "rows": r.rows_converted,
                    "steps": r.steps_run,
                    "ok": r.ok,
                }
                for r in results
            ]
        },
    )


def smoke_fake_throughput_defaults(ctx: SmokeContext) -> SmokeResult:
    """Expert-v2: shape×pattern throughput defaults are coherent."""
    t0 = time.monotonic()
    from anvil.recipes.throughput import list_throughput_profiles, throughput_defaults

    profiles = list_throughput_profiles()
    vlm = throughput_defaults(shape="dense_vlm", pattern="vlm_sft")
    grpo = throughput_defaults(shape="dense_lm", pattern="rl_verifiable")
    ok = (
        len(profiles) >= 4
        and vlm.batch_size == 1
        and vlm.checkpoint_every >= 1
        and grpo.learning_rate <= 1e-4
        and "batch_size" in vlm.as_overrides()
    )
    return SmokeResult(
        name="fake_throughput_defaults",
        ok=ok,
        duration_s=time.monotonic() - t0,
        detail=f"n_profiles={len(profiles)} vlm_bs={vlm.batch_size} grpo_lr={grpo.learning_rate}",
        meta={"n_profiles": len(profiles)},
    )


def smoke_fake_multi_hour_resume(ctx: SmokeContext) -> SmokeResult:
    """Expert-v2 multi-hour ops stand-in: checkpoint mid-run, resume, finish budget.

    Not wall-clock multi-hour — exercises the same resume contract forge long jobs use.
    """
    t0 = time.monotonic()
    from anvil.protocol.messages import Example, Message, TextPart
    from anvil.recipes.checkpoint import resume_path
    from anvil.recipes.sft import run_sft

    ex = Example(
        messages=(
            Message(role="user", content=(TextPart(text="scale?"),)),
            Message(role="assistant", content=(TextPart(text="yes"),)),
        )
    )
    run_dir = ctx.report_run / "multi-hour-resume"
    # Phase 1: first half of budget
    r1 = run_sft(
        endpoint="fake://",
        examples=[ex],
        steps=20,
        run_dir=str(run_dir),
        early_stop=False,
        stop_on_southward=False,
        checkpoint_every=5,
    )
    ok1 = r1.steps_run == 20 and resume_path(run_dir).is_file()
    # Phase 2: extend budget via resume (same contract as multi-hour restarts)
    r2 = run_sft(
        endpoint="fake://",
        examples=[ex],
        steps=40,
        run_dir=str(run_dir),
        early_stop=False,
        stop_on_southward=False,
        checkpoint_every=5,
        resume=True,
    )
    ok = ok1 and r2.resumed_from_step >= 20 and r2.steps_run == 20
    return SmokeResult(
        name="fake_multi_hour_resume",
        ok=ok,
        duration_s=time.monotonic() - t0,
        detail=(
            f"phase1_steps={r1.steps_run} resumed_from={r2.resumed_from_step} "
            f"phase2_steps={r2.steps_run} ckpt={r2.checkpoint_path}"
        ),
        meta={
            "resumed_from_step": r2.resumed_from_step,
            "phase2_steps": r2.steps_run,
        },
    )


def smoke_fake_southward(ctx: SmokeContext) -> SmokeResult:
    t0 = time.monotonic()
    from anvil.observe.metrics import RunMetricsWriter
    from anvil.observe.southward import scan_and_log

    run_dir = ctx.report_run / "fake-southward"
    w = RunMetricsWriter(run_dir)
    for i in range(6):
        w.log_step(
            step=i,
            reward_mean=0.5,
            reward_std=0.0,
            group_reward_std_mean=0.0,
            loss=0.1,
            n_datums=4,
        )
    rep = scan_and_log(run_dir)
    ok = "advantage_collapse" in rep.names and not rep.ok
    return SmokeResult(
        name="fake_southward",
        ok=ok,
        duration_s=time.monotonic() - t0,
        detail=f"flags={rep.names}",
        meta={"flags": rep.names},
    )


def smoke_fake_expert_v0(ctx: SmokeContext) -> SmokeResult:
    """Full expert_v0_smoke path on fake:// (convert → train → southward → meta)."""
    t0 = time.monotonic()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    # Keep all artifacts under report_run so laptop smokes never touch /mnt/...
    media = ctx.report_run / "media"
    observe = ctx.report_run / "observe"
    book = ctx.report_run / "recipe_book"
    media.mkdir(parents=True, exist_ok=True)
    observe.mkdir(parents=True, exist_ok=True)
    book.mkdir(parents=True, exist_ok=True)
    env["ANVIL_RECIPE_BOOK"] = str(book)
    run_id = f"lab-expert-v0-{_now_id()}"
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "expert_v0_smoke.py"),
        "--endpoint",
        "fake://",
        "--media-root",
        str(media),
        "--observe-root",
        str(observe),
        "--output-jsonl",
        str(ctx.report_run / f"{run_id}.jsonl"),
        "--export",
        str(ctx.report_run / f"{run_id}-export"),
        "--run-id",
        run_id,
        "--max-rows",
        "8",
        "--steps",
        "30",
        "--holdout",
        "1",
        "--early-stop-mode",
        "production",
        "--early-stop-patience",
        "12",
        "--run-meta",
        "--promote-recipe",
        f"lab-{run_id}",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    ok = (
        proc.returncode == 0
        and "done:" in out
        and "southward:" in out
        and "meta_exec:" in out
    )
    return SmokeResult(
        name="fake_expert_v0",
        ok=ok,
        duration_s=time.monotonic() - t0,
        detail=out.replace("\n", " ")[-500:],
        meta={"returncode": proc.returncode, "run_id": run_id},
    )


def smoke_fake_early_stop(ctx: SmokeContext) -> SmokeResult:
    t0 = time.monotonic()
    from anvil.recipes.grpo import run_grpo

    run_dir = ctx.report_run / "fake-early-stop"
    res = run_grpo(
        endpoint="fake://",
        steps=40,
        group_size=4,
        run_dir=str(run_dir),
        reward_fn=lambda _t, _toks: 1.0,
        early_stop=True,
        early_stop_patience=5,
    )
    ok = (
        res.early_stop_reason == "ceiling_x5"
        and res.steps_run == 5
        and res.steps_run < 40
    )
    return SmokeResult(
        name="fake_early_stop",
        ok=ok,
        duration_s=time.monotonic() - t0,
        detail=f"steps={res.steps_run} reason={res.early_stop_reason}",
        meta={"early_stop_reason": res.early_stop_reason, "steps": res.steps_run},
    )


def smoke_fake_rl_queue(ctx: SmokeContext) -> SmokeResult:
    t0 = time.monotonic()
    from anvil.recipes.rl_queue import RLQueueRecipe, RLStage, run_rl_queue

    recipe = RLQueueRecipe(
        id="lab-smoke-queue",
        name="lab smoke",
        stages=(
            RLStage(id="a", prompt="p1", gold="always_one", max_steps=30),
            RLStage(id="b", prompt="p2", gold="always_one", max_steps=30),
        ),
        early_stop_patience=3,
        advance_on=("ceiling",),
        stop_queue_on=("floor",),
    )
    result = run_rl_queue(
        recipe,
        base_model="toy/lm",
        endpoint="fake://",
        observe_root=ctx.observe_root / "lab-smoke-queue",
        run_prefix="lab-q",
        fake_prompts=True,
    )
    ok = (
        result.stages_run == 2
        and result.stages[0].advanced
        and result.stages[0].result.early_stop_reason is not None
        and result.stages[0].result.adapter_id == result.stages[1].result.adapter_id
    )
    return SmokeResult(
        name="fake_rl_queue",
        ok=ok,
        duration_s=time.monotonic() - t0,
        detail=(
            f"stages={result.stages_run} "
            f"s0_stop={result.stages[0].result.early_stop_reason} "
            f"advanced={result.stages[0].advanced}"
        ),
        meta={"stages_run": result.stages_run},
    )


def smoke_grpo_early_stop_local(ctx: SmokeContext) -> SmokeResult:
    t0 = time.monotonic()
    if not ctx.endpoint.startswith("local"):
        return SmokeResult(
            name="grpo_early_stop_local",
            ok=True,
            duration_s=0.0,
            skipped=True,
            detail="skipped: endpoint is not local://",
        )
    if not _have_local_stack():
        return SmokeResult(
            name="grpo_early_stop_local",
            ok=False,
            duration_s=0.0,
            skipped=True,
            detail="skipped: peft/torch not installed",
        )
    if not _model_exists(ctx.model):
        return SmokeResult(
            name="grpo_early_stop_local",
            ok=False,
            duration_s=0.0,
            skipped=True,
            detail=f"skipped: missing model {ctx.model}",
        )

    from transformers import AutoTokenizer

    from anvil.recipes.grpo import run_grpo
    from anvil.recipes.verifiable import (
        DEFAULT_HARD_PROBLEMS,
        detokenize_via_tokenizer,
        exact_integer_reward,
    )

    user, gold = DEFAULT_HARD_PROBLEMS[0]
    tok = AutoTokenizer.from_pretrained(ctx.model, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    text = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = [int(t) for t in tok.encode(text, add_special_tokens=False)]
    detok = detokenize_via_tokenizer(tok)
    reward_fn = exact_integer_reward(detok, gold)
    run_dir = ctx.observe_root / f"lab-smoke-grpo-es-{_now_id()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Budget high enough that only early-stop should finish (if policy saturates).
    # If it never saturates, still OK if we got signal (steps>0, finite rewards).
    res = run_grpo(
        base_model=ctx.model,
        prompts=[list(ids) for _ in range(4)],
        reward_fn=reward_fn,
        group_size=ctx.group_size,
        steps=80,
        endpoint=ctx.endpoint,
        run_dir=str(run_dir),
        probes=[list(ids)],
        probe_every=2,
        detokenize=detok,
        early_stop=True,
        early_stop_patience=8,
        overrides={"rank": 8, "max_tokens": 16, "temperature": 1.1},
    )
    stopped = res.early_stop_reason is not None
    ok = res.steps_run >= 1 and (
        stopped or (res.mean_reward and max(res.mean_reward) > 0)
    )
    detail = (
        f"steps={res.steps_run}/80 early_stop={res.early_stop_reason} "
        f"final_r={res.mean_reward[-1] if res.mean_reward else None} "
        f"observe={run_dir.name}"
    )
    # Prefer early-stop for this smoke when reward hit ceiling
    if res.mean_reward and res.mean_reward[-1] >= 0.99 and not stopped:
        ok = False
        detail += " FAIL: saturated without early_stop"
    return SmokeResult(
        name="grpo_early_stop_local",
        ok=ok,
        duration_s=time.monotonic() - t0,
        detail=detail,
        meta={
            "steps": res.steps_run,
            "early_stop_reason": res.early_stop_reason,
            "observe_run": run_dir.name,
        },
    )


def smoke_rl_queue_local(ctx: SmokeContext) -> SmokeResult:
    t0 = time.monotonic()
    if not ctx.endpoint.startswith("local"):
        return SmokeResult(
            name="rl_queue_local",
            ok=True,
            duration_s=0.0,
            skipped=True,
            detail="skipped: endpoint is not local://",
        )
    if not _have_local_stack() or not _model_exists(ctx.model):
        return SmokeResult(
            name="rl_queue_local",
            ok=False,
            duration_s=0.0,
            skipped=True,
            detail="skipped: local stack or model missing",
        )

    from anvil.recipes.rl_queue import RLQueueRecipe, RLStage, run_rl_queue
    from anvil.recipes.verifiable import DEFAULT_HARD_PROBLEMS

    # Two stages from hard bank; short max_steps so early-stop can hand off.
    stages = tuple(
        RLStage(
            id=f"s{i}-{gold}",
            prompt=prompt,
            gold=gold,
            max_steps=60,
            early_stop_patience=6,
        )
        for i, (prompt, gold) in enumerate(DEFAULT_HARD_PROBLEMS[:2])
    )
    recipe = RLQueueRecipe(
        id="lab-smoke-curric",
        name="lab smoke curriculum",
        stages=stages,
        group_size=ctx.group_size,
        early_stop_patience=6,
        advance_on=("ceiling", "collapsed"),
        stop_queue_on=("floor",),
        advance_on_budget=True,
    )
    prefix = f"lab-curric-{_now_id()}"
    result = run_rl_queue(
        recipe,
        base_model=ctx.model,
        endpoint=ctx.endpoint,
        observe_root=ctx.observe_root,
        run_prefix=prefix,
        rank=8,
        max_tokens=16,
        temperature=1.1,
        probe_every=2,
        carry_adapter=True,
        fake_prompts=False,
    )
    # Success: at least one stage completed; if early-stop ceiling, second stage started
    ok = result.stages_run >= 1
    if (
        result.stages
        and result.stages[0].result.early_stop_reason
        and result.stages[0].result.early_stop_reason.startswith("ceiling")
    ):
        ok = result.stages_run >= 2 and result.stages[0].advanced
    detail = (
        f"stages_run={result.stages_run} adapter={result.adapter_id} "
        + " | ".join(
            f"{o.stage.id}:steps={o.result.steps_run},stop={o.result.early_stop_reason},adv={o.advanced}"
            for o in result.stages
        )
    )
    return SmokeResult(
        name="rl_queue_local",
        ok=ok,
        duration_s=time.monotonic() - t0,
        detail=detail,
        meta={
            "stages_run": result.stages_run,
            "prefix": prefix,
            "adapter_id": result.adapter_id,
        },
    )


def smoke_vlm_sft_local(ctx: SmokeContext) -> SmokeResult:
    t0 = time.monotonic()
    if not ctx.endpoint.startswith("local"):
        return SmokeResult(
            name="vlm_sft_local",
            ok=True,
            duration_s=0.0,
            skipped=True,
            detail="skipped: endpoint is not local://",
        )
    if not _have_local_stack() or not _model_exists(ctx.vlm_model):
        return SmokeResult(
            name="vlm_sft_local",
            ok=False,
            duration_s=0.0,
            skipped=True,
            detail=f"skipped: VLM missing or no local stack ({ctx.vlm_model})",
        )

    export = ctx.report_run / "vlm-out"
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "vlm_smoke.py"),
        "--endpoint",
        ctx.endpoint,
        "--model",
        ctx.vlm_model,
        "--media-root",
        str(ctx.media_root),
        "--steps",
        "3",
        "--export",
        str(export),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    ok = proc.returncode == 0 and "steps=" in proc.stdout
    # Prefer loss trend down when printed
    if "loss trend: down" in proc.stdout or "losses=" in proc.stdout:
        ok = proc.returncode == 0
    return SmokeResult(
        name="vlm_sft_local",
        ok=ok,
        duration_s=time.monotonic() - t0,
        detail=(proc.stdout[-400:] + proc.stderr[-200:]).replace("\n", " ")[:500],
        meta={"returncode": proc.returncode},
    )


def smoke_observe_root_writable(ctx: SmokeContext) -> SmokeResult:
    t0 = time.monotonic()
    try:
        ctx.observe_root.mkdir(parents=True, exist_ok=True)
        probe = ctx.observe_root / f".lab-smoke-write-{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return SmokeResult(
            name="observe_root_writable",
            ok=True,
            duration_s=time.monotonic() - t0,
            detail=str(ctx.observe_root),
        )
    except Exception as e:
        return SmokeResult(
            name="observe_root_writable",
            ok=False,
            duration_s=time.monotonic() - t0,
            detail=str(e),
        )


def smoke_host_env(ctx: SmokeContext) -> SmokeResult:
    t0 = time.monotonic()
    detail = (
        f"host={platform.node()} py={sys.version.split()[0]} "
        f"cuda={_cuda_ok()} local_stack={_have_local_stack()} "
        f"model={_model_exists(ctx.model)} vlm={_model_exists(ctx.vlm_model)}"
    )
    return SmokeResult(
        name="host_env",
        ok=True,
        duration_s=time.monotonic() - t0,
        detail=detail,
        meta={
            "cuda": _cuda_ok(),
            "local_stack": _have_local_stack(),
            "model_ok": _model_exists(ctx.model),
            "vlm_ok": _model_exists(ctx.vlm_model),
        },
    )


# --- registry / profiles -----------------------------------------------------

SMOKES: dict[str, SmokeFn] = {
    "host_env": smoke_host_env,
    "observe_root_writable": smoke_observe_root_writable,
    "fake_early_stop": smoke_fake_early_stop,
    "fake_sft_early_stop": smoke_fake_sft_early_stop,
    "fake_dpo_observe": smoke_fake_dpo_observe,
    "fake_meta_exec": smoke_fake_meta_exec,
    "fake_meta_live_runners": smoke_fake_meta_live_runners,
    "fake_scale_ladder": smoke_fake_scale_ladder,
    "fake_throughput_defaults": smoke_fake_throughput_defaults,
    "fake_multi_hour_resume": smoke_fake_multi_hour_resume,
    "fake_southward": smoke_fake_southward,
    "fake_rl_queue": smoke_fake_rl_queue,
    "fake_expert_v0": smoke_fake_expert_v0,
    "grpo_early_stop_local": smoke_grpo_early_stop_local,
    "rl_queue_local": smoke_rl_queue_local,
    "vlm_sft_local": smoke_vlm_sft_local,
}

_QUICK_CORE: tuple[str, ...] = (
    "host_env",
    "observe_root_writable",
    "fake_early_stop",
    "fake_sft_early_stop",
    "fake_dpo_observe",
    "fake_meta_exec",
    "fake_meta_live_runners",
    "fake_southward",
    "fake_rl_queue",
    "fake_expert_v0",
)

PROFILES: dict[str, tuple[str, ...]] = {
    # Run often: laptop / pre-push / CI-adjacent (fake only, seconds)
    "quick": _QUICK_CORE,
    # Expert-v2: scale ladder + throughput defaults + multi-hour resume contract
    "multi_hour": _QUICK_CORE
    + (
        "fake_scale_ladder",
        "fake_throughput_defaults",
        "fake_multi_hour_resume",
    ),
    "nightly": _QUICK_CORE
    + (
        "grpo_early_stop_local",
        "rl_queue_local",
    ),
    "full": _QUICK_CORE
    + (
        "fake_scale_ladder",
        "fake_throughput_defaults",
        "fake_multi_hour_resume",
        "grpo_early_stop_local",
        "rl_queue_local",
        "vlm_sft_local",
    ),
}


def run_suite(
    names: Sequence[str],
    ctx: SmokeContext,
) -> list[SmokeResult]:
    results: list[SmokeResult] = []
    for name in names:
        fn = SMOKES[name]
        print(f"\n=== smoke: {name} ===", flush=True)
        if ctx.dry_run:
            results.append(
                SmokeResult(
                    name=name,
                    ok=True,
                    duration_s=0.0,
                    skipped=True,
                    detail="dry-run",
                )
            )
            continue
        try:
            r = fn(ctx)
        except Exception as e:
            r = SmokeResult(
                name=name,
                ok=False,
                duration_s=0.0,
                detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}",
            )
        status = "SKIP" if r.skipped else ("OK" if r.ok else "FAIL")
        print(f"→ {status} {r.duration_s:.1f}s  {r.detail}", flush=True)
        results.append(r)
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        default="quick",
        help="smoke set (default: quick)",
    )
    p.add_argument(
        "--only",
        default=None,
        help="comma-separated smoke names (overrides --profile)",
    )
    p.add_argument("--endpoint", default="fake://")
    p.add_argument(
        "--model",
        default=os.environ.get(
            "ANVIL_LAB_MODEL", "/mnt/data/models/qwen2.5-1.5b-instruct"
        ),
    )
    p.add_argument(
        "--vlm-model",
        default=os.environ.get(
            "ANVIL_LAB_VLM", "/mnt/data/models/Qwen2.5-VL-3B-Instruct"
        ),
    )
    p.add_argument(
        "--observe-root",
        default=os.environ.get("ANVIL_OBSERVE_ROOT", ""),
    )
    p.add_argument(
        "--media-root",
        default=os.environ.get("ANVIL_MEDIA_ROOT", "/mnt/data/anvil-media"),
    )
    p.add_argument(
        "--report-dir",
        default=os.environ.get("ANVIL_LAB_SMOKE_DIR", ""),
        help="write report.json here (default: observe_root/lab-smokes or ./lab-smokes)",
    )
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--list", action="store_true", help="list smokes and exit")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        print("Profiles:")
        for k, v in PROFILES.items():
            print(f"  {k}: {', '.join(v)}")
        print("Smokes:")
        for k in SMOKES:
            print(f"  {k}")
        return 0

    names = (
        tuple(x.strip() for x in args.only.split(",") if x.strip())
        if args.only
        else PROFILES[args.profile]
    )
    for n in names:
        if n not in SMOKES:
            print(f"unknown smoke: {n}", file=sys.stderr)
            return 2

    observe = Path(
        args.observe_root
        or (
            "/mnt/data/anvil-observe"
            if Path("/mnt/data").is_dir()
            else str(Path.home() / ".anvil" / "observe")
        )
    )
    report_base = Path(
        args.report_dir
        or (
            str(Path("/mnt/data/anvil-runs/lab-smokes"))
            if Path("/mnt/data").is_dir()
            else str(REPO / "lab-smokes")
        )
    )
    report_run = report_base / f"run-{_now_id()}"
    report_run.mkdir(parents=True, exist_ok=True)

    ctx = SmokeContext(
        endpoint=args.endpoint,
        model=args.model,
        vlm_model=args.vlm_model,
        observe_root=observe,
        media_root=Path(args.media_root),
        report_run=report_run,
        group_size=args.group_size,
        dry_run=args.dry_run,
    )

    print(f"profile={args.profile if not args.only else 'custom'} smokes={list(names)}")
    print(f"endpoint={ctx.endpoint} report={report_run}")
    t0 = time.monotonic()
    results = run_suite(names, ctx)
    wall = time.monotonic() - t0

    failed = [r for r in results if not r.ok and not r.skipped]
    skipped = [r for r in results if r.skipped]
    passed = [r for r in results if r.ok and not r.skipped]

    report = {
        "ts": time.time(),
        "host": platform.node(),
        "profile": args.profile if not args.only else "custom",
        "endpoint": ctx.endpoint,
        "wall_s": wall,
        "passed": len(passed),
        "failed": len(failed),
        "skipped": len(skipped),
        "ok": len(failed) == 0,
        "results": [asdict(r) for r in results],
    }
    report_path = report_run / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # latest symlink-ish copy
    latest = report_base / "latest.json"
    try:
        latest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError:
        pass

    print("\n=== summary ===")
    print(f"passed={len(passed)} failed={len(failed)} skipped={len(skipped)} wall={wall:.1f}s")
    print(f"report={report_path}")
    for r in results:
        flag = "SKIP" if r.skipped else ("OK" if r.ok else "FAIL")
        print(f"  [{flag}] {r.name}: {r.detail[:120]}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
