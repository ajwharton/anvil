"""Basic on-policy verifiable RL recipe (GRPO / IS-shaped).

Public pattern (DeepSeekMath GRPO + Tinker RL loop):
  1. sample G completions from current policy
  2. score with a verifiable reward (exact match, unit test, …)
  3. group-relative advantages
  4. forward_backward(importance_sampling | ppo) + optim_step

**RL datum convention** (Phase 2; enforced by LocalBackend._forward_backward_rl):
each datum carries the FULL sequence context — ``model_input`` is
``prompt + completion[:-1]``, and ``target_tokens`` / ``logprobs`` /
``advantages`` are all length-C arrays aligned to the completion. Old-policy
logprobs come straight from ``SampledSequence.logprobs``; the backend slices
the last C positions' logits so old and current logprobs align by
construction. Use ``datum_from_rollout`` — do not hand-roll this shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from anvil.client.service import ServiceClient
from anvil.protocol.types import AdamParams, Datum, LoraTargets, ModelInput, SamplingParams
from anvil.recipes.profiles import JobPattern, RecipePlan, plan_recipe

RewardFn = Callable[[str, Sequence[int]], float]


def datum_from_rollout(
    prompt_tokens: Sequence[int],
    completion_tokens: Sequence[int],
    old_logprobs: Sequence[float],
    advantage: float,
    *,
    weights: Sequence[float] | None = None,
) -> Datum:
    """Build one GRPO/IS datum from a sampled completion.

    prompt_tokens + completion_tokens are the full episode; old_logprobs are
    the sampling policy's per-token logprobs (``SampledSequence.logprobs``);
    advantage is broadcast over all completion tokens (GRPO style).
    """
    prompt = [int(t) for t in prompt_tokens]
    comp = [int(t) for t in completion_tokens]
    lp = [float(x) for x in old_logprobs]
    if not comp:
        raise ValueError("empty completion — nothing to train on")
    if len(lp) != len(comp):
        raise ValueError(
            f"old_logprobs ({len(lp)}) must align to completion tokens ({len(comp)})"
        )
    return Datum(
        model_input=ModelInput.from_ints(prompt + comp[:-1]),
        loss_fn_inputs={
            "target_tokens": comp,
            "logprobs": lp,
            "advantages": [float(advantage)] * len(comp),
            "weights": ([float(x) for x in weights] if weights is not None else [1.0] * len(comp)),
        },
    )


@dataclass
class GRPOResult:
    plan: RecipePlan
    steps_run: int
    mean_reward: list[float]
    losses: list[float]
    adapter_id: str


def build_plan(base_model: str, **overrides: Any) -> RecipePlan:
    return plan_recipe(
        base_model=base_model,
        pattern=JobPattern.RL_VERIFIABLE,
        overrides=overrides or None,
    )


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """GRPO-style: advantage = reward - group mean (no std scale in v0)."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    return [float(r) - mean for r in rewards]


def run_grpo(
    *,
    base_model: str = "Qwen/Qwen3.5-4B",
    prompts: Sequence[Sequence[int]] | None = None,
    reward_fn: RewardFn | None = None,
    group_size: int = 4,
    steps: int = 3,
    endpoint: str = "fake://",
    plan: RecipePlan | None = None,
    overrides: dict[str, Any] | None = None,
) -> GRPOResult:
    plan = plan or build_plan(base_model, **(overrides or {}))
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
    reward_fn = reward_fn or _exact_match_toy
    prompts = list(prompts) if prompts else [list(range(10, 26))]

    losses: list[float] = []
    mean_rewards: list[float] = []

    for _ in range(steps):
        sc = tc.save_weights_and_get_sampling_client(name=f"grpo-{_}")
        batch: list[Datum] = []
        step_rewards: list[float] = []

        for prompt_tokens in prompts:
            prompt = ModelInput.from_ints(prompt_tokens)
            sample = sc.sample(
                prompt,
                SamplingParams(
                    max_tokens=plan.max_tokens,
                    temperature=plan.temperature,
                    seed=None,
                ),
                num_samples=group_size,
            ).result()
            rewards = []
            for seq in sample.sequences:
                r = reward_fn("", seq.tokens)
                rewards.append(r)
                step_rewards.append(r)
            adv = group_advantages(rewards)
            for seq, a in zip(sample.sequences, adv, strict=True):
                if not seq.tokens:
                    continue
                if seq.logprobs is None:
                    raise ValueError(
                        "sampler returned no per-token logprobs — old-policy "
                        "logprobs are required for the IS/PPO family"
                    )
                batch.append(
                    datum_from_rollout(prompt_tokens, seq.tokens, seq.logprobs, a)
                )

        if not batch:
            break
        fb = tc.forward_backward(batch, loss_fn=plan.loss_fn).result()
        tc.optim_step(AdamParams(learning_rate=plan.learning_rate)).result()
        losses.append(fb.loss)
        mean_rewards.append(sum(step_rewards) / max(len(step_rewards), 1))

    return GRPOResult(
        plan=plan,
        steps_run=len(losses),
        mean_reward=mean_rewards,
        losses=losses,
        adapter_id=str(tc.adapter_id),
    )


def _exact_match_toy(_text: str, tokens: Sequence[int]) -> float:
    """Placeholder verifiable reward — prefer even token ids (demo only)."""
    if not tokens:
        return 0.0
    return 1.0 if (sum(tokens) % 2 == 0) else 0.0
