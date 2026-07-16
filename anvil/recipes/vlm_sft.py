"""VLM instruction SFT recipe — shape from model card, freeze vision by default.

Public anchors: HF TRL VLM cookbook (LoRA on LM projections), Qwen2.5-VL card
(image-text-to-text, agentic). Our fine-tune data is separate; the *recipe shape*
is card-derivable.
"""

from __future__ import annotations

from typing import Any, Sequence

from anvil.protocol.messages import Example, ImagePart, Message, TextPart
from anvil.recipes.model_card import ModelCardFacts, inspect_base_model
from anvil.recipes.profiles import JobPattern, RecipePlan, plan_recipe
from anvil.recipes.sft import SFTResult, run_sft


def build_plan(
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    *,
    card: ModelCardFacts | None = None,
    fetch_remote: bool = True,
    **overrides: Any,
) -> RecipePlan:
    card = card or inspect_base_model(base_model, fetch_remote=fetch_remote)
    return plan_recipe(
        base_model=card.repo_id if card.repo_id else base_model,
        pattern=JobPattern.VLM_SFT,
        shape=card.shape,
        overrides=overrides or None,
        card=card,
    )


def toy_vlm_examples() -> list[Example]:
    """Multimodal examples with media refs (no blobs in-repo)."""
    return [
        Example(
            messages=(
                Message(
                    role="user",
                    content=(
                        TextPart(text="Is the grasp reachable?"),
                        ImagePart(ref="cas://sha256/demo_frame", detail="high"),
                    ),
                ),
                Message(role="assistant", content=(TextPart(text="yes"),)),
            ),
            meta={"env": "robot-sim"},
        )
    ]


def run_vlm_sft(
    *,
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    examples: Sequence[Example] | None = None,
    steps: int = 3,
    endpoint: str = "fake://",
    export_dir: str | None = None,
    fetch_remote: bool = True,
    overrides: dict[str, Any] | None = None,
) -> SFTResult:
    card = inspect_base_model(base_model, fetch_remote=fetch_remote)
    plan = build_plan(base_model, card=card, fetch_remote=False, **(overrides or {}))
    return run_sft(
        base_model=plan.base_model,
        examples=list(examples) if examples is not None else toy_vlm_examples(),
        steps=steps,
        endpoint=endpoint,
        export_dir=export_dir,
        plan=plan,
    )
