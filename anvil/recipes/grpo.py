"""Basic on-policy verifiable RL recipe (GRPO / IS-shaped).

Public pattern (DeepSeekMath GRPO + Tinker RL loop):
  1. sample G completions from current policy
  2. score with a verifiable reward (exact match, unit test, …)
  3. group-relative advantages
  4. forward_backward(importance_sampling | ppo) + optim_step

Real rewards and token logprobs land with Phase 1–2 workers; fake backend
exercises the control flow today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from anvil.client.service import ServiceClient
from anvil.protocol.types import AdamParams, Datum, LoraTargets, ModelInput, SamplingParams
from anvil.recipes.profiles import JobPattern, RecipePlan, plan_recipe

RewardFn = Callable[[str, Sequence[int]], float]


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
            for seq, a, r in zip(sample.sequences, adv, rewards, strict=True):
                # Toy: treat completion tokens as targets under IS
                toks = list(seq.tokens)
                if len(toks) < 2:
                    continue
                logprobs = list(seq.logprobs) if seq.logprobs else [-1.0] * len(toks)
                batch.append(
                    Datum(
                        model_input=ModelInput.from_ints(toks[:-1]),
                        loss_fn_inputs={
                            "target_tokens": toks[1:],
                            "weights": [1.0] * (len(toks) - 1),
                            "logprobs": logprobs[1:] if len(logprobs) > 1 else logprobs,
                            "advantages": [a] * (len(toks) - 1),
                        },
                    )
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
