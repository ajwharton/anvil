"""VLM instruction SFT recipe — shape from model card, freeze vision by default.

Public anchors: HF TRL VLM cookbook (LoRA on LM projections), Qwen2.5-VL card
(image-text-to-text, agentic). Our fine-tune data is separate; the *recipe shape*
is card-derivable.

Phase 3.2: pass ``media_store`` + ``renderer`` (or let this module build
``HFVLMRenderer`` for ``local://``) so Examples with ``ImagePart`` refs train
through the same four verbs.
"""

from __future__ import annotations

from typing import Any, Sequence

from anvil.protocol.messages import Example, ImagePart, Message, TextPart
from anvil.recipes.model_card import ModelCardFacts, inspect_base_model
from anvil.recipes.profiles import JobPattern, RecipePlan, plan_recipe
from anvil.recipes.sft import SFTResult, examples_to_data, run_sft
from anvil.render.text import ToyTextRenderer


def build_plan(
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    *,
    card: ModelCardFacts | None = None,
    fetch_remote: bool = True,
    **overrides: Any,
) -> RecipePlan:
    card = card or inspect_base_model(base_model, fetch_remote=fetch_remote)
    # Keep absolute lab paths so forge loads /mnt/data/models/... not a bare name.
    load_id = card.local_path or base_model or card.repo_id
    return plan_recipe(
        base_model=load_id,
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
    media_store: Any | None = None,
    renderer: Any | None = None,
    run_dir: str | None = None,
    probes: Sequence[Example] | None = None,
    probe_every: int = 1,
    early_stop: bool | None = None,
    early_stop_mode: str = "production",
    early_stop_patience: int | None = None,
    early_stop_rel_eps: float | None = None,
) -> SFTResult:
    """VLM SFT loop.

    - ``fake://`` / missing store: ``ToyTextRenderer`` (image refs as text placeholders).
    - ``local://`` + ``media_store``: ``HFVLMRenderer`` (processor-backed) unless
      ``renderer`` is passed explicitly.
    - ``run_dir``: append loss / wall / n_image_refs to ``metrics.jsonl`` for
      live ``/observe`` (P3.6 / roadmap 3.C).
    - ``probes``: held-out Examples sampled every ``probe_every`` steps → ``probes.jsonl``.
    """
    card = inspect_base_model(base_model, fetch_remote=fetch_remote)
    plan = build_plan(base_model, card=card, fetch_remote=False, **(overrides or {}))
    exs = list(examples) if examples is not None else toy_vlm_examples()

    if renderer is None:
        if media_store is not None and (
            endpoint.startswith("local://") or endpoint.startswith("http")
        ):
            from anvil.render.vlm import HFVLMRenderer

            renderer = HFVLMRenderer(plan.base_model, media_store)
        else:
            renderer = ToyTextRenderer()

    # Validate renderer can build data before opening a train session
    _ = examples_to_data(exs[:1], renderer=renderer)

    kwargs: dict[str, Any] = dict(
        base_model=plan.base_model,
        examples=exs,
        steps=steps,
        endpoint=endpoint,
        export_dir=export_dir,
        plan=plan,
        renderer=renderer,
        run_dir=run_dir,
        job="vlm_sft",
        probes=probes,
        probe_every=probe_every,
        early_stop=early_stop,
        early_stop_mode=early_stop_mode,
        early_stop_patience=early_stop_patience,
    )
    if early_stop_rel_eps is not None:
        kwargs["early_stop_rel_eps"] = early_stop_rel_eps
    return run_sft(**kwargs)
