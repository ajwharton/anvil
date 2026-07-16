"""Basic SFT recipe — CE on assistant tokens (Tinker-shaped loop).

Runs against any ServiceClient backend (fake today; PEFT worker later).
Shape/knobs should come from ``plan_recipe`` / model card inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from anvil.client.service import ServiceClient
from anvil.client.training import TrainingClient
from anvil.protocol.messages import Example, Message, TextPart
from anvil.protocol.types import AdamParams, Datum, LoraTargets
from anvil.recipes.profiles import JobPattern, RecipePlan, plan_recipe
from anvil.render.text import ToyTextRenderer


@dataclass
class SFTResult:
    plan: RecipePlan
    steps_run: int
    losses: list[float]
    adapter_id: str
    export_path: str | None = None


def build_plan(base_model: str, **overrides: Any) -> RecipePlan:
    return plan_recipe(
        base_model=base_model,
        pattern=JobPattern.SFT_CHAT,
        overrides=overrides or None,
    )


def examples_to_data(
    examples: Sequence[Example],
    *,
    renderer: ToyTextRenderer | None = None,
) -> list[Datum]:
    r = renderer or ToyTextRenderer()
    return [r.render_example_for_sft(ex) for ex in examples]


def run_sft(
    *,
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    examples: Sequence[Example] | None = None,
    steps: int | None = None,
    endpoint: str = "fake://",
    export_dir: str | None = None,
    plan: RecipePlan | None = None,
    overrides: dict[str, Any] | None = None,
) -> SFTResult:
    """Minimal SFT: create LoRA client → CE steps → optional export."""
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

    data = examples_to_data(examples) if examples else _toy_batch()
    n = steps if steps is not None else min(5, plan.max_steps)
    losses: list[float] = []
    for _ in range(n):
        fb = tc.forward_backward(data, loss_fn=plan.loss_fn).result()
        tc.optim_step(AdamParams(learning_rate=plan.learning_rate)).result()
        losses.append(fb.loss)

    export_path = None
    if export_dir:
        export_path = tc.export_adapter(export_dir, format=plan.export_hint if plan.export_hint == "peft" else "peft").path

    return SFTResult(
        plan=plan,
        steps_run=n,
        losses=losses,
        adapter_id=str(tc.adapter_id),
        export_path=export_path,
    )


def _toy_batch() -> list[Datum]:
    ex = Example(
        messages=(
            Message(role="user", content=(TextPart(text="2+2?"),)),
            Message(role="assistant", content=(TextPart(text="4"),)),
        )
    )
    return examples_to_data([ex])
