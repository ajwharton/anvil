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

**Observability** (Phase 2.5): pass ``run_dir`` and every step appends a
metrics record (reward mean/std, within-group reward std — the
advantage-collapse tripwire — loss, fb metrics pass-through) to
``<run_dir>/metrics.jsonl``; pass ``probes`` and the live policy is
re-sampled greedily on those fixed prompts every ``probe_every`` steps into
``<run_dir>/probes.jsonl``. anvil-web tails both (see /api/observe/*).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from anvil.client.service import ServiceClient
from anvil.observe.metrics import RunMetricsWriter
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
    run_dir: str | None = None,
    probes: Sequence[Sequence[int]] | None = None,
    probe_every: int = 1,
    detokenize: Callable[[Sequence[int]], str] | None = None,
) -> GRPOResult:
    """Run the GRPO/IS loop.

    Phase 2.5 observability: pass ``run_dir`` to append per-step records to
    ``<run_dir>/metrics.jsonl`` (reward mean/std, within-group reward std —
    the advantage-collapse tripwire, loss, IS mean_ratio passthrough) and,
    when ``probes`` are given, greedy probe completions of the LIVE policy to
    ``probes.jsonl`` every ``probe_every`` steps. ``detokenize`` maps probe
    tokens to text for the UI (reward-hacking has no scalar signature — eyes
    do).
    """
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
    writer = RunMetricsWriter(run_dir) if run_dir else None

    losses: list[float] = []
    mean_rewards: list[float] = []

    for step_ix in range(steps):
        t0 = time.monotonic()
        sc = tc.save_weights_and_get_sampling_client(name=f"grpo-{step_ix}")
        batch: list[Datum] = []
        step_rewards: list[float] = []
        group_stds: list[float] = []

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
            group_stds.append(statistics.pstdev(rewards) if len(rewards) > 1 else 0.0)
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

        # Probe the LIVE policy (same weights that generated this batch) —
        # greedy + fixed seed so completions are comparable across steps.
        if writer is not None and probes and step_ix % probe_every == 0:
            for probe_ix, probe_tokens in enumerate(probes):
                out = sc.sample(
                    ModelInput.from_ints(probe_tokens),
                    SamplingParams(max_tokens=plan.max_tokens, temperature=0.0, seed=0),
                    num_samples=1,
                ).result()
                seq = out.sequences[0] if out.sequences else None
                toks = tuple(seq.tokens) if seq is not None else ()
                writer.log_probe(
                    step=step_ix,
                    probe_idx=probe_ix,
                    tokens=toks,
                    text=detokenize(toks) if detokenize else None,
                    reward=reward_fn("", toks),
                )

        fb = tc.forward_backward(batch, loss_fn=plan.loss_fn).result()
        tc.optim_step(AdamParams(learning_rate=plan.learning_rate)).result()
        losses.append(fb.loss)
        mean_rewards.append(sum(step_rewards) / max(len(step_rewards), 1))
        if writer is not None:
            writer.log_step(
                step=step_ix,
                reward_mean=mean_rewards[-1],
                reward_std=(
                    statistics.pstdev(step_rewards) if len(step_rewards) > 1 else 0.0
                ),
                group_reward_std_mean=(
                    sum(group_stds) / len(group_stds) if group_stds else 0.0
                ),
                loss=fb.loss,
                n_datums=len(batch),
                fb_metrics=dict(fb.metrics),
                wall_time_s=time.monotonic() - t0,
            )

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
