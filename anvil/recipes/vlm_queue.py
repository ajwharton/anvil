"""Multi-stage VLM/SFT queue — advance when a stage early-stops (roadmap 3.C).

Mirrors :mod:`anvil.recipes.rl_queue` for vision / text SFT curricula: run
ordered stages on the **same LoRA adapter**, advance on loss plateau (or any
early-stop), halt on explicit stop signals. Uses ``run_vlm_sft`` / ``run_sft``
client reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from anvil.client.service import ServiceClient
from anvil.observe.metrics import RunMetricsWriter
from anvil.protocol.messages import Example
from anvil.protocol.types import LoraTargets
from anvil.recipes.rl_queue import should_advance
from anvil.recipes.sft import SFTResult, run_sft
from anvil.recipes.vlm_sft import build_plan, run_vlm_sft


@dataclass(frozen=True, slots=True)
class VLMStage:
    """One data slice / curriculum step for VLM or text SFT."""

    id: str
    examples: tuple[Example, ...]
    max_steps: int = 50
    early_stop_patience: int | None = None
    probes: tuple[Example, ...] = ()
    job: str = "vlm_sft"  # vlm_sft | sft
    notes: str = ""


@dataclass(frozen=True, slots=True)
class VLMQueueRecipe:
    """Ordered VLM/SFT stages + advance/stop policy."""

    id: str
    name: str
    stages: tuple[VLMStage, ...]
    early_stop_patience: int = 15
    # loss_plateau / southward / any early_stop → next stage
    advance_on: tuple[str, ...] = ("loss_plateau", "southward", "dpo_")
    stop_queue_on: tuple[str, ...] = ()
    advance_on_budget: bool = True
    notes: str = ""


@dataclass
class VLMStageOutcome:
    stage: VLMStage
    result: SFTResult
    advanced: bool
    queue_halted: bool
    observe_run_id: str


@dataclass
class VLMQueueResult:
    recipe: VLMQueueRecipe
    stages: list[VLMStageOutcome] = field(default_factory=list)
    adapter_id: str | None = None

    @property
    def stages_run(self) -> int:
        return len(self.stages)


def run_vlm_queue(
    recipe: VLMQueueRecipe,
    *,
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    endpoint: str = "fake://",
    run_dir: str | None = None,
    fetch_remote: bool = False,
    overrides: dict[str, Any] | None = None,
    media_store: Any | None = None,
    renderer: Any | None = None,
) -> VLMQueueResult:
    """Execute VLM/SFT stages with shared adapter + early-stop advance.

    Observe layout when ``run_dir`` is set::

        <run_dir>/queue.jsonl          # stage events
        <run_dir>/<stage.id>/          # per-stage metrics/probes
    """
    if not recipe.stages:
        raise ValueError("VLMQueueRecipe requires at least one stage")

    plan = build_plan(base_model, fetch_remote=fetch_remote, **(overrides or {}))
    k = plan.as_knobs()
    svc = ServiceClient(endpoint=endpoint)
    tc = svc.create_lora_training_client(
        base_model=plan.base_model,
        rank=k["rank"],
        alpha=k.get("alpha"),
        modalities=k["modalities"],
        lora_targets=LoraTargets(
            language=k["language_lora"],
            vision_encoder=k["vision_encoder_lora"],
            mm_projector=k["mm_projector_lora"],
        ),
    )
    writer = RunMetricsWriter(run_dir) if run_dir else None
    out = VLMQueueResult(recipe=recipe, adapter_id=str(tc.adapter_id))

    try:
        for stage in recipe.stages:
            stage_dir = f"{run_dir.rstrip('/')}/{stage.id}" if run_dir else None
            patience = (
                stage.early_stop_patience
                if stage.early_stop_patience is not None
                else recipe.early_stop_patience
            )
            if writer is not None:
                writer.log_event(
                    step=-1,
                    event="vlm_stage_start",
                    stage_id=stage.id,
                    max_steps=stage.max_steps,
                    job=stage.job,
                )

            shared = dict(
                base_model=plan.base_model,
                examples=list(stage.examples),
                steps=stage.max_steps,
                endpoint=endpoint,
                run_dir=stage_dir,
                probes=list(stage.probes) if stage.probes else None,
                early_stop=True,
                early_stop_mode="production",
                early_stop_patience=patience,
                stop_on_southward=bool(stage_dir),
                service_client=svc,
                training_client=tc,
                close_clients=False,
            )
            if stage.job == "sft":
                res = run_sft(
                    job="sft",
                    plan=plan,
                    renderer=renderer,
                    **shared,
                )
            else:
                res = run_vlm_sft(
                    fetch_remote=False,
                    media_store=media_store,
                    renderer=renderer,
                    overrides=overrides,
                    **shared,
                )

            hit_budget = res.early_stop_reason is None and res.steps_run >= stage.max_steps
            advanced, halted = should_advance(
                res.early_stop_reason,
                hit_budget=hit_budget,
                advance_on=recipe.advance_on,
                stop_queue_on=recipe.stop_queue_on,
                advance_on_budget=recipe.advance_on_budget,
            )
            # last stage never advances
            if stage is recipe.stages[-1]:
                advanced = False

            if writer is not None:
                writer.log_event(
                    step=res.steps_run,
                    event="vlm_stage_end",
                    stage_id=stage.id,
                    early_stop_reason=res.early_stop_reason,
                    steps_run=res.steps_run,
                    advanced=advanced,
                    queue_halted=halted,
                    job=stage.job,
                )

            out.stages.append(
                VLMStageOutcome(
                    stage=stage,
                    result=res,
                    advanced=advanced,
                    queue_halted=halted,
                    observe_run_id=stage.id,
                )
            )
            if halted or not advanced:
                break
    finally:
        svc.close()

    return out


__all__ = [
    "VLMQueueRecipe",
    "VLMQueueResult",
    "VLMStage",
    "VLMStageOutcome",
    "run_vlm_queue",
]
