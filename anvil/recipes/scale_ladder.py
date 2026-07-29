"""Corpus scale ladder 1k → 5k → 50k+ (Expert-v2).

Defines rungs operators climb on forge: convert max-rows, recommended train
steps, checkpoint cadence, and observe expectations. Does **not** download
Bridge/OXE; use ``scripts/scale_ladder.py`` against a lab episode pack or
``--demo`` synthetic pack for CI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from anvil.recipes.throughput import ThroughputDefaults, throughput_defaults


@dataclass(frozen=True, slots=True)
class ScaleRung:
    """One rung of the corpus scale ladder."""

    id: str  # "1k" | "5k" | "50k"
    max_rows: int
    # Recommended train budget when exercising this rung (not full multi-epoch)
    train_steps: int
    checkpoint_every: int
    frames_per_episode: int
    # CI / laptop stand-in row count for the same code path
    demo_rows: int
    notes: str
    min_probe_records: int = 1

    def to_public(self) -> dict[str, Any]:
        return asdict(self)


# Product ladder (roadmap Expert-v2). demo_rows keep unit/lab-quick cheap.
SCALE_RUNGS: tuple[ScaleRung, ...] = (
    ScaleRung(
        id="1k",
        max_rows=1_000,
        train_steps=50,
        checkpoint_every=25,
        frames_per_episode=4,
        demo_rows=32,
        notes="First real Bridge/Robo2VLM slice; loss curve + probes must look sane.",
    ),
    ScaleRung(
        id="5k",
        max_rows=5_000,
        train_steps=150,
        checkpoint_every=25,
        frames_per_episode=4,
        demo_rows=64,
        notes="Curriculum stretch; expect multi-tens of minutes on 3B VLM Spark.",
    ),
    ScaleRung(
        id="50k",
        max_rows=50_000,
        train_steps=500,
        checkpoint_every=50,
        frames_per_episode=2,
        demo_rows=128,
        notes=(
            "Multi-hour territory: require checkpoint_every + resume; "
            "prefer frames_per_episode=2 to bound CAS size."
        ),
    ),
)


def get_rung(rung_id: str) -> ScaleRung:
    rid = str(rung_id).lower().strip().replace(" ", "")
    aliases = {"1000": "1k", "5000": "5k", "50000": "50k", "50k+": "50k"}
    rid = aliases.get(rid, rid)
    for r in SCALE_RUNGS:
        if r.id == rid:
            return r
    known = ", ".join(x.id for x in SCALE_RUNGS)
    raise ValueError(f"unknown scale rung {rung_id!r}; choose one of: {known}")


def list_rungs() -> list[ScaleRung]:
    return list(SCALE_RUNGS)


@dataclass
class ScaleLadderPlan:
    """Concrete plan for one or more rungs (convert + train knobs)."""

    rungs: list[ScaleRung]
    throughput: ThroughputDefaults
    demo: bool = False
    dataset: str = "bridge_v2"
    # Paths filled by CLI when known
    source: str | None = None
    media_root: str | None = None
    jsonl_root: str | None = None
    observe_root: str | None = None

    def to_public(self) -> dict[str, Any]:
        return {
            "rungs": [r.to_public() for r in self.rungs],
            "throughput": self.throughput.to_public(),
            "demo": self.demo,
            "dataset": self.dataset,
            "source": self.source,
            "media_root": self.media_root,
            "jsonl_root": self.jsonl_root,
            "observe_root": self.observe_root,
        }


def build_ladder_plan(
    *,
    rungs: Sequence[str] | None = None,
    demo: bool = False,
    shape: str = "dense_vlm",
    pattern: str = "vlm_sft",
    dataset: str = "bridge_v2",
    source: str | None = None,
    media_root: str | None = None,
    jsonl_root: str | None = None,
    observe_root: str | None = None,
) -> ScaleLadderPlan:
    """Build a scale-ladder plan for convert+train exercise."""
    if rungs is None or (len(rungs) == 1 and rungs[0] in {"all", "*"}):
        selected = list(SCALE_RUNGS)
    else:
        selected = [get_rung(r) for r in rungs]
    thr = throughput_defaults(shape=shape, pattern=pattern)
    return ScaleLadderPlan(
        rungs=selected,
        throughput=thr,
        demo=demo,
        dataset=dataset,
        source=source,
        media_root=media_root,
        jsonl_root=jsonl_root,
        observe_root=observe_root,
    )


@dataclass
class RungExerciseResult:
    rung: ScaleRung
    rows_converted: int
    jsonl_path: str | None
    steps_run: int
    early_stop_reason: str | None
    checkpoint_path: str | None
    run_dir: str | None
    ok: bool
    detail: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def expected_convert_rows(rung: ScaleRung, *, demo: bool) -> int:
    return rung.demo_rows if demo else rung.max_rows


def exercise_rung(
    rung: ScaleRung,
    *,
    demo: bool = True,
    endpoint: str = "fake://",
    work_dir: str | Path,
    source: str | Path | None = None,
    media_root: str | Path | None = None,
    train: bool = True,
    train_steps: int | None = None,
    checkpoint: bool = True,
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    dataset: str = "bridge_v2",
) -> RungExerciseResult:
    """Convert one rung (demo pack or lab source) and optionally train.

    **Demo/CI:** synthetic episode pack sized to ``rung.demo_rows``.
    **Forge:** pass ``source`` episode pack + real ``max_rows`` (``demo=False``).

    Train path uses text SFT on fake:// (or local://) from instruction/response
    text so CI has no VLM pixel dependency; forge VLM train stays
    ``run_vlm_sft`` / expert_v0_smoke.
    """
    from anvil.data.convert import ConvertConfig, convert_corpus, write_demo_episode_pack
    from anvil.protocol.messages import Example, Message, TextPart
    from anvil.recipes.sft import run_sft

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    media = Path(media_root) if media_root else work / "media"
    media.mkdir(parents=True, exist_ok=True)
    jsonl_path = work / f"{rung.id}.jsonl"
    rows_target = expected_convert_rows(rung, demo=demo)

    if demo or source is None:
        pack = work / f"pack-{rung.id}"
        # Enough episodes so max_rows is reachable (keyframe-ish density)
        n_eps = max(rows_target, 8)
        write_demo_episode_pack(pack, n_episodes=n_eps, frames_per=2)
        src = pack
        kind = "episode_pack"
    else:
        src = Path(source)
        kind = "episode_pack"

    cfg = ConvertConfig(
        source=src,
        media_root=media,
        output_jsonl=jsonl_path,
        source_kind=kind,
        max_rows=rows_target,
        frames_per_episode=rung.frames_per_episode if not demo else 2,
        dataset=dataset if not demo else "demo_bridge_like",
        license_note="synthetic-demo" if demo else None,
    )
    conv = convert_corpus(cfg)
    n_rows = int(conv.n_rows)
    if n_rows <= 0 and jsonl_path.is_file():
        n_rows = sum(1 for _ in jsonl_path.open(encoding="utf-8") if _.strip())

    if not train:
        ok = n_rows >= min(rows_target, 1)
        return RungExerciseResult(
            rung=rung,
            rows_converted=n_rows,
            jsonl_path=str(jsonl_path),
            steps_run=0,
            early_stop_reason=None,
            checkpoint_path=None,
            run_dir=None,
            ok=ok,
            detail=f"convert_only rows={n_rows}/{rows_target}",
            meta={"convert": getattr(conv, "__dict__", {})},
        )

    # Build text Examples from JSONL rows (instruction → response)
    examples: list[Example] = []
    if jsonl_path.is_file():
        import json

        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            inst = str(row.get("instruction") or row.get("prompt") or "do task")
            resp = str(row.get("response") or row.get("action") or "ok")
            examples.append(
                Example(
                    messages=(
                        Message(role="user", content=(TextPart(text=inst),)),
                        Message(role="assistant", content=(TextPart(text=resp),)),
                    )
                )
            )
    if not examples:
        return RungExerciseResult(
            rung=rung,
            rows_converted=n_rows,
            jsonl_path=str(jsonl_path),
            steps_run=0,
            early_stop_reason=None,
            checkpoint_path=None,
            run_dir=None,
            ok=False,
            detail="no examples after convert",
        )

    steps = train_steps if train_steps is not None else (
        min(8, rung.train_steps) if demo else rung.train_steps
    )
    ckpt_every = (
        max(2, min(rung.checkpoint_every, steps // 2 or 1))
        if checkpoint
        else None
    )
    run_dir = work / f"train-{rung.id}"
    res = run_sft(
        base_model=base_model,
        examples=examples[: max(1, min(len(examples), 64))],
        steps=steps,
        endpoint=endpoint,
        run_dir=str(run_dir),
        early_stop=False,
        stop_on_southward=False,
        checkpoint_every=ckpt_every,
        job="sft",
    )
    ok = res.steps_run == steps and n_rows >= min(rows_target, 1)
    if checkpoint and ckpt_every:
        from anvil.recipes.checkpoint import resume_path

        ok = ok and resume_path(run_dir).is_file()

    return RungExerciseResult(
        rung=rung,
        rows_converted=n_rows,
        jsonl_path=str(jsonl_path),
        steps_run=res.steps_run,
        early_stop_reason=res.early_stop_reason,
        checkpoint_path=res.checkpoint_path,
        run_dir=str(run_dir),
        ok=ok,
        detail=(
            f"rows={n_rows}/{rows_target} steps={res.steps_run}/{steps} "
            f"ckpt={res.checkpoint_path}"
        ),
        meta={
            "adapter_id": res.adapter_id,
            "checkpoint_every": ckpt_every,
        },
    )


def exercise_ladder(
    plan: ScaleLadderPlan,
    *,
    work_dir: str | Path,
    endpoint: str = "fake://",
    train: bool = True,
) -> list[RungExerciseResult]:
    """Run convert(+train) for each rung in ``plan``."""
    work = Path(work_dir)
    out: list[RungExerciseResult] = []
    for rung in plan.rungs:
        # Demo rungs: scale train_steps down; forge uses full train_steps
        steps = None if not plan.demo else min(6, max(4, rung.train_steps // 20))
        out.append(
            exercise_rung(
                rung,
                demo=plan.demo,
                endpoint=endpoint,
                work_dir=work / rung.id,
                source=plan.source,
                media_root=plan.media_root or str(work / "media"),
                train=train,
                train_steps=steps,
                checkpoint=True,
                dataset=plan.dataset,
            )
        )
    return out


__all__ = [
    "SCALE_RUNGS",
    "RungExerciseResult",
    "ScaleLadderPlan",
    "ScaleRung",
    "build_ladder_plan",
    "exercise_ladder",
    "exercise_rung",
    "expected_convert_rows",
    "get_rung",
    "list_rungs",
]
