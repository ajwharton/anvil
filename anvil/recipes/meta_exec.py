"""Meta-recipe executor — drive stage graphs with live signals (Expert-v1).

Does not train by itself: each stage is a callable that returns a
:class:`StageRunResult` (signal + optional metrics). The executor advances via
:func:`~anvil.recipes.meta.next_stage`, logs stage events to
``metrics.jsonl`` (observe SSOT), and stops on halt or exhausted stages.

Agents/operators inject runners (SFT, GRPO queue stage, no-op export, …).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from anvil.observe.metrics import RunMetricsWriter
from anvil.recipes.meta import MetaRecipe, MetaStage, next_stage


@dataclass
class StageRunResult:
    """Outcome of one meta-recipe stage."""

    signal: str | None = None
    """Live signal for edge matching (e.g. early_stop:loss_plateau_patience_40)."""

    halted: bool = False
    """If True, stop the whole meta-recipe (no further stages)."""

    metrics: dict[str, Any] = field(default_factory=dict)
    """Optional free-form summary for the transcript / event log."""


class StageRunner(Protocol):
    def __call__(self, stage: MetaStage, *, step_index: int) -> StageRunResult: ...


@dataclass
class MetaStageOutcome:
    stage: MetaStage
    result: StageRunResult
    advanced: bool


@dataclass
class MetaExecResult:
    meta: MetaRecipe
    outcomes: list[MetaStageOutcome] = field(default_factory=list)
    stopped_reason: str | None = None
    run_dir: str | None = None

    @property
    def stages_run(self) -> int:
        return len(self.outcomes)


def run_meta_recipe(
    meta: MetaRecipe,
    runner: StageRunner,
    *,
    run_dir: str | Path | None = None,
    max_stages: int | None = None,
    start_stage_id: str | None = None,
) -> MetaExecResult:
    """Execute ``meta`` stages until halt, no next stage, or ``max_stages``.

    Parameters
    ----------
    runner
        Called once per stage. Return a signal for edge matching (prefix match
        supported via :func:`next_stage`).
    run_dir
        If set, append ``stage_start`` / ``stage_end`` / ``meta_halt`` events
        to ``metrics.jsonl``.
    """
    if not meta.stages:
        return MetaExecResult(meta=meta, stopped_reason="empty_meta", run_dir=str(run_dir) if run_dir else None)

    writer = RunMetricsWriter(run_dir) if run_dir else None
    by_id = {s.id: s for s in meta.stages}
    current: MetaStage | None
    if start_stage_id:
        current = by_id.get(start_stage_id)
        if current is None:
            raise ValueError(f"unknown start_stage_id {start_stage_id!r}")
    else:
        current = meta.stages[0]

    outcomes: list[MetaStageOutcome] = []
    stopped: str | None = None
    step_index = 0
    cap = max_stages if max_stages is not None else len(meta.stages) + len(meta.edges) + 8
    seen: set[str] = set()

    while current is not None and step_index < cap:
        if current.id in seen:
            stopped = f"cycle_detected:{current.id}"
            break
        seen.add(current.id)

        if writer is not None:
            writer.log_event(
                step=step_index,
                event="stage_start",
                reason=None,
                meta_id=meta.id,
                stage_id=current.id,
                recipe_id=current.recipe_id,
                source=current.source,
                pattern=current.pattern,
            )

        result = runner(current, step_index=step_index)

        if writer is not None:
            writer.log_event(
                step=step_index,
                event="stage_end",
                reason=result.signal,
                meta_id=meta.id,
                stage_id=current.id,
                recipe_id=current.recipe_id,
                halted=result.halted,
                **{k: v for k, v in result.metrics.items() if k not in {"reason", "event", "step"}},
            )

        advanced = False
        if result.halted:
            outcomes.append(MetaStageOutcome(stage=current, result=result, advanced=False))
            stopped = result.signal or "stage_halted"
            if writer is not None:
                writer.log_event(
                    step=step_index,
                    event="meta_halt",
                    reason=stopped,
                    meta_id=meta.id,
                    stage_id=current.id,
                )
            break

        nxt = next_stage(meta, current_stage_id=current.id, signal=result.signal)
        if nxt is not None and nxt.id != current.id:
            advanced = True
        outcomes.append(MetaStageOutcome(stage=current, result=result, advanced=advanced))

        if nxt is None or nxt.id == current.id:
            stopped = result.signal or "complete"
            break
        current = nxt
        step_index += 1
    else:
        if stopped is None:
            stopped = "max_stages" if step_index >= cap else "complete"

    return MetaExecResult(
        meta=meta,
        outcomes=outcomes,
        stopped_reason=stopped,
        run_dir=str(run_dir) if run_dir else None,
    )


__all__ = [
    "MetaExecResult",
    "MetaStageOutcome",
    "StageRunResult",
    "StageRunner",
    "run_meta_recipe",
]
