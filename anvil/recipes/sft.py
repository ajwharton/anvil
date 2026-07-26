"""Basic SFT recipe — CE on assistant tokens (forward_backward + optim_step).

Runs against any ServiceClient backend. Shape/knobs should come from
``plan_recipe`` / model card inspection. Pass a multimodal renderer
(``HFVLMRenderer``) for vision Examples.

Observability (P3.6 / P.Sufficiency): pass ``run_dir`` to append per-step
records (loss, wall, n_image_refs, n_tokens) to ``<run_dir>/metrics.jsonl``
via the same :class:`~anvil.observe.metrics.RunMetricsWriter` SSOT as GRPO.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from anvil.client.service import ServiceClient
from anvil.client.training import TrainingClient
from anvil.observe.metrics import RunMetricsWriter
from anvil.protocol.messages import Example, Message, TextPart
from anvil.protocol.types import AdamParams, Datum, LoraTargets
from anvil.recipes.profiles import JobPattern, RecipePlan, plan_recipe
from anvil.render.text import ToyTextRenderer


class _SFTRenderer(Protocol):
    def render_example_for_sft(self, example: Example) -> Datum: ...


@dataclass
class SFTResult:
    plan: RecipePlan
    steps_run: int
    losses: list[float]
    adapter_id: str
    export_path: str | None = None
    run_dir: str | None = None


def build_plan(base_model: str, **overrides: Any) -> RecipePlan:
    return plan_recipe(
        base_model=base_model,
        pattern=JobPattern.SFT_CHAT,
        overrides=overrides or None,
    )


def examples_to_data(
    examples: Sequence[Example],
    *,
    renderer: _SFTRenderer | None = None,
) -> list[Datum]:
    r: _SFTRenderer = renderer if renderer is not None else ToyTextRenderer()
    return [r.render_example_for_sft(ex) for ex in examples]


def count_image_refs(data: Sequence[Datum]) -> int:
    """Sum ``image_refs`` recorded on each datum (renderer / LocalBackend path)."""
    n = 0
    for d in data:
        refs = d.loss_fn_inputs.get("image_refs") or []
        n += len(refs)
    return n


def run_sft(
    *,
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    examples: Sequence[Example] | None = None,
    steps: int | None = None,
    endpoint: str = "fake://",
    export_dir: str | None = None,
    plan: RecipePlan | None = None,
    overrides: dict[str, Any] | None = None,
    renderer: _SFTRenderer | None = None,
    run_dir: str | None = None,
    job: str = "sft",
) -> SFTResult:
    """Minimal SFT: create LoRA client → CE steps → optional export.

    Pass ``run_dir`` to write ``metrics.jsonl`` for anvil-web ``/observe``.
    ``job`` labels records (``sft`` or ``vlm_sft``) so the UI charts loss.
    """
    plan = plan or build_plan(base_model, **(overrides or {}))
    k = plan.as_knobs()
    svc = ServiceClient(endpoint=endpoint)
    tc: TrainingClient = svc.create_lora_training_client(
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

    data = examples_to_data(examples, renderer=renderer) if examples else _toy_batch()
    n_image_refs = count_image_refs(data)
    if n_image_refs == 0 and examples:
        n_image_refs = sum(len(ex.image_refs()) for ex in examples)
    writer = RunMetricsWriter(run_dir) if run_dir else None
    n = steps if steps is not None else min(5, plan.max_steps)
    losses: list[float] = []
    for step_ix in range(n):
        t0 = time.monotonic()
        fb = tc.forward_backward(data, loss_fn=plan.loss_fn).result()
        tc.optim_step(AdamParams(learning_rate=plan.learning_rate)).result()
        losses.append(fb.loss)
        if writer is not None:
            fb_metrics = {str(k_): float(v) for k_, v in dict(fb.metrics or {}).items()
                          if isinstance(v, (int, float))}
            writer.log_sft_step(
                step=step_ix,
                loss=fb.loss,
                n_datums=len(data),
                n_image_refs=n_image_refs,
                fb_metrics=fb_metrics,
                wall_time_s=time.monotonic() - t0,
                job=job,
            )

    export_path = None
    if export_dir:
        export_path = tc.export_adapter(
            export_dir, format=plan.export_hint if plan.export_hint == "peft" else "peft"
        ).path

    svc.close()
    return SFTResult(
        plan=plan,
        steps_run=n,
        losses=losses,
        adapter_id=str(tc.adapter_id),
        export_path=export_path,
        run_dir=run_dir,
    )


def _toy_batch() -> list[Datum]:
    ex = Example(
        messages=(
            Message(role="user", content=(TextPart(text="2+2?"),)),
            Message(role="assistant", content=(TextPart(text="4"),)),
        )
    )
    return examples_to_data([ex])
