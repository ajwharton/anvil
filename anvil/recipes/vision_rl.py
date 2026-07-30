"""On-policy vision RL (Phase 4.B) — multimodal GRPO + rubrics + stage queue.

Builds on :func:`~anvil.recipes.grpo.run_grpo` with:

- **Multimodal sample prompts** — image refs + text tokens as ``ModelInput``
- **Vision rewards** — keyword / rubric / action-bin scorers on detokenized text
- **Stage queue** — curriculum of vision tasks on a shared LoRA (same idea as
  text ``rl_queue`` / ``vlm_queue``)

Lab trains; edge only samples small students. Never dump long vision RL runs
onto a storage-constrained robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from anvil.client.service import ServiceClient
from anvil.observe.metrics import RunMetricsWriter
from anvil.protocol.types import EncodedTextChunk, ImageRefChunk, LoraTargets, ModelInput
from anvil.recipes.grpo import GRPOResult, RewardFn, run_grpo
from anvil.recipes.profiles import JobPattern, plan_recipe
from anvil.recipes.rl_queue import should_advance
from anvil.recipes.vision_rewards import (
    RubricCriterion,
    action_bin_overlap_reward,
    keyword_caption_reward,
    rubric_reward,
    toy_detokenize,
)

DetokenizeFn = Callable[[Sequence[int]], str]


@dataclass(frozen=True, slots=True)
class VisionRollout:
    """One image-grounded RL prompt (scene caption, grasp rubric, action bins)."""

    id: str
    instruction: str
    image_refs: tuple[str, ...] = ()
    """``cas://`` or path refs for the observation (may be empty for text-only)."""

    required_keywords: tuple[str, ...] = ()
    any_of_keywords: tuple[str, ...] = ()
    gold_phrase: str | None = None
    gold_action_bins: tuple[int, ...] | str | None = None
    rubric: tuple[RubricCriterion, ...] = ()
    prompt_token_ids: tuple[int, ...] = ()
    """Optional explicit token ids for the instruction (CI / fake://)."""

    def to_model_input(self, *, default_prompt_tokens: Sequence[int] | None = None) -> ModelInput:
        toks = list(self.prompt_token_ids) if self.prompt_token_ids else list(
            default_prompt_tokens or range(10, 26)
        )
        chunks: list[Any] = [EncodedTextChunk(tokens=tuple(int(t) for t in toks))]
        for ref in self.image_refs:
            chunks.append(ImageRefChunk(ref=str(ref), detail="auto"))
        return ModelInput.from_chunks(chunks)

    def make_reward(self, *, detokenize: DetokenizeFn | None = None) -> RewardFn:
        if self.rubric:
            return rubric_reward(self.rubric, detokenize=detokenize)
        if self.gold_action_bins is not None:
            return action_bin_overlap_reward(self.gold_action_bins, detokenize=detokenize)
        if self.gold_phrase:
            from anvil.recipes.vision_rewards import exact_phrase_reward

            return exact_phrase_reward(self.gold_phrase, detokenize=detokenize)
        if self.required_keywords or self.any_of_keywords:
            return keyword_caption_reward(
                required=self.required_keywords,
                any_of=self.any_of_keywords,
                detokenize=detokenize,
            )
        # Weak default: any non-empty completion
        def _any(text: str, tokens: Sequence[int]) -> float:
            body = text or (detokenize(tokens) if detokenize else "")
            return 1.0 if body.strip() else 0.0

        return _any


def toy_vision_rollouts() -> list[VisionRollout]:
    """Synthetic multimodal prompts for fake:// CI (no real pixels required)."""
    dig = "e" * 64
    ref = f"cas://sha256/{dig}.png"
    return [
        VisionRollout(
            id="kitchen_caption",
            instruction="Describe the room in one short sentence.",
            image_refs=(ref,),
            required_keywords=("kitchen",),
            any_of_keywords=("stove", "cabinet", "table"),
            prompt_token_ids=tuple(range(20, 36)),
        ),
        VisionRollout(
            id="chair_hazard",
            instruction="Name the main furniture hazard.",
            image_refs=(ref,),
            required_keywords=("chair",),
            prompt_token_ids=tuple(range(30, 46)),
        ),
    ]


def run_vision_grpo(
    *,
    rollouts: Sequence[VisionRollout] | None = None,
    base_model: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
    group_size: int = 4,
    steps: int = 3,
    endpoint: str = "fake://",
    run_dir: str | None = None,
    detokenize: DetokenizeFn | None = None,
    overrides: dict[str, Any] | None = None,
    fetch_remote: bool = False,
    early_stop: bool = True,
    early_stop_patience: int = 8,
    stop_on_southward: bool | None = None,
    service_client: Any | None = None,
    training_client: Any | None = None,
    close_clients: bool = True,
    sample_endpoint: str | None = None,
    sample_endpoints: Sequence[str] | None = None,
    checkpoint_every: int | None = None,
    resume: bool = False,
) -> GRPOResult:
    """On-policy GRPO with multimodal prompts + vision rewards.

    ``fake://`` uses synthetic token detokenization when ``detokenize`` is None.
    Lab hosts should pass a real tokenizer decode (or HF ``detokenize_via_tokenizer``).
    """
    rolls = list(rollouts) if rollouts is not None else toy_vision_rollouts()
    if not rolls:
        raise ValueError("run_vision_grpo requires at least one VisionRollout")

    detok = detokenize if detokenize is not None else toy_detokenize
    prompts = [r.to_model_input() for r in rolls]
    rewards = [r.make_reward(detokenize=detok) for r in rolls]
    ov = {**(overrides or {}), "modalities": ("text", "image")}
    plan = plan_recipe(
        base_model=base_model,
        pattern=JobPattern.RL_VERIFIABLE,
        overrides=ov,
        use_card=False,
        fetch_remote=fetch_remote,
    )

    return run_grpo(
        base_model=base_model,
        prompts=prompts,
        reward_fn=rewards,
        group_size=group_size,
        steps=steps,
        endpoint=endpoint,
        plan=plan,
        overrides=ov,
        run_dir=run_dir,
        probes=prompts[:1],
        probe_every=1,
        detokenize=detok,
        job="vision_grpo",
        early_stop=early_stop,
        early_stop_patience=early_stop_patience,
        stop_on_southward=stop_on_southward,
        service_client=service_client,
        training_client=training_client,
        close_clients=close_clients,
        sample_endpoint=sample_endpoint,
        sample_endpoints=sample_endpoints,
        checkpoint_every=checkpoint_every,
        resume=resume,
    )


# --- vision / robot stage queue ---------------------------------------------


@dataclass(frozen=True, slots=True)
class VisionRLStage:
    id: str
    rollouts: tuple[VisionRollout, ...]
    max_steps: int = 20
    group_size: int | None = None
    early_stop_patience: int | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class VisionRLQueueRecipe:
    id: str
    name: str
    stages: tuple[VisionRLStage, ...]
    group_size: int = 4
    early_stop_patience: int = 6
    advance_on: tuple[str, ...] = ("ceiling", "collapsed", "southward")
    stop_queue_on: tuple[str, ...] = ("floor",)
    advance_on_budget: bool = True
    notes: str = ""


@dataclass
class VisionRLStageOutcome:
    stage: VisionRLStage
    result: GRPOResult
    advanced: bool
    queue_halted: bool
    observe_run_id: str


@dataclass
class VisionRLQueueResult:
    recipe: VisionRLQueueRecipe
    stages: list[VisionRLStageOutcome] = field(default_factory=list)
    adapter_id: str | None = None

    @property
    def stages_run(self) -> int:
        return len(self.stages)


def toy_vision_rl_queue() -> VisionRLQueueRecipe:
    r0, r1 = toy_vision_rollouts()
    return VisionRLQueueRecipe(
        id="vision_caption_ladder_v0",
        name="Vision caption → hazard ladder",
        stages=(
            VisionRLStage(id="caption", rollouts=(r0,), max_steps=5, notes="scene keywords"),
            VisionRLStage(id="hazard", rollouts=(r1,), max_steps=5, notes="chair hazard"),
        ),
        notes="Phase 4.B toy curriculum for fake:// / lab smoke",
    )


def run_vision_rl_queue(
    recipe: VisionRLQueueRecipe | None = None,
    *,
    base_model: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
    endpoint: str = "fake://",
    run_dir: str | None = None,
    detokenize: DetokenizeFn | None = None,
    overrides: dict[str, Any] | None = None,
    fetch_remote: bool = False,
) -> VisionRLQueueResult:
    """Execute vision RL stages on a **shared** LoRA adapter."""
    recipe = recipe or toy_vision_rl_queue()
    if not recipe.stages:
        raise ValueError("VisionRLQueueRecipe requires stages")

    plan = plan_recipe(
        base_model=base_model,
        pattern=JobPattern.RL_VERIFIABLE,
        overrides={**(overrides or {}), "modalities": ("text", "image")},
        use_card=False,
        fetch_remote=fetch_remote,
    )
    k = plan.as_knobs()
    svc = ServiceClient(endpoint=endpoint)
    tc = svc.create_lora_training_client(
        base_model=plan.base_model,
        rank=k["rank"],
        modalities=k["modalities"],
        lora_targets=LoraTargets(
            language=k["language_lora"],
            vision_encoder=k["vision_encoder_lora"],
            mm_projector=k["mm_projector_lora"],
        ),
    )
    out = VisionRLQueueResult(recipe=recipe, adapter_id=str(tc.adapter_id))
    queue_writer = RunMetricsWriter(run_dir) if run_dir else None

    try:
        for stage in recipe.stages:
            stage_dir = f"{run_dir}/{stage.id}" if run_dir else None
            if queue_writer is not None:
                queue_writer.log_event(
                    step=0,
                    event="vision_rl_stage_start",
                    reason=stage.id,
                    job="vision_grpo",
                    stage=stage.id,
                )
            res = run_vision_grpo(
                rollouts=list(stage.rollouts),
                base_model=base_model,
                group_size=stage.group_size or recipe.group_size,
                steps=stage.max_steps,
                endpoint=endpoint,
                run_dir=stage_dir,
                detokenize=detokenize,
                overrides=overrides,
                fetch_remote=fetch_remote,
                early_stop=True,
                early_stop_patience=stage.early_stop_patience or recipe.early_stop_patience,
                service_client=svc,
                training_client=tc,
                close_clients=False,
            )
            hit_budget = res.early_stop_reason is None and res.steps_run >= stage.max_steps
            advanced, halted = should_advance(
                res.early_stop_reason,
                hit_budget=hit_budget,
                advance_on=recipe.advance_on,
                stop_queue_on=recipe.stop_queue_on,
                advance_on_budget=recipe.advance_on_budget,
            )
            out.stages.append(
                VisionRLStageOutcome(
                    stage=stage,
                    result=res,
                    advanced=advanced and not halted,
                    queue_halted=halted,
                    observe_run_id=stage.id,
                )
            )
            if queue_writer is not None:
                queue_writer.log_event(
                    step=res.steps_run,
                    event="vision_rl_stage_end",
                    reason=res.early_stop_reason or "budget",
                    job="vision_grpo",
                    stage=stage.id,
                    advanced=advanced and not halted,
                    halted=halted,
                )
            if halted or not advanced:
                break
    finally:
        svc.close()
    return out
