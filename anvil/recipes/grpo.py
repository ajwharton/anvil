"""Basic on-policy verifiable RL recipe (GRPO / IS-shaped).

Public pattern (DeepSeekMath GRPO-style on-policy loop):
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

**Adapter sync** (Phase 2.5 Tier 1): pass ``sample_endpoint`` (or inject
``sample_backend``) and every ``sync_every`` steps the loop writes a train
snapshot then ``load_snapshot`` on the sample worker so rollouts/probes hit
the hot-swapped LoRA without reloading the base.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from anvil.backends.base import Backend, SnapshotLoader
from anvil.client.sampling import SamplingClient
from anvil.client.service import ServiceClient
from anvil.observe.metrics import RunMetricsWriter
from anvil.protocol.types import (
    AdamParams,
    AdapterId,
    CheckpointRef,
    Datum,
    LoraTargets,
    ModelInput,
    SamplingParams,
)
from anvil.recipes.checkpoint import (
    apply_resume_to_client,
    load_resume_state,
    save_train_checkpoint,
)
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
    sync_count: int = 0
    early_stop_reason: str | None = None
    resumed_from_step: int = 0
    checkpoint_path: str | None = None


# Default: after this many consecutive dead-signal steps, abandon the run.
DEFAULT_EARLY_STOP_PATIENCE = 8
DEFAULT_GROUP_STD_EPS = 1e-8
DEFAULT_REWARD_HI = 0.99
DEFAULT_REWARD_LO = 0.01


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """GRPO-style: advantage = reward - group mean (no std scale in v0)."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    return [float(r) - mean for r in rewards]


def classify_dead_step(
    *,
    reward_mean: float,
    group_reward_std_mean: float,
    group_std_eps: float = DEFAULT_GROUP_STD_EPS,
    reward_hi: float = DEFAULT_REWARD_HI,
    reward_lo: float = DEFAULT_REWARD_LO,
) -> str | None:
    """Return a short reason if this step has no useful GRPO signal, else None.

    - ``ceiling``: reward saturated AND within-group std ~0 (all correct)
    - ``floor``: reward ~0 AND within-group std ~0 (all wrong the same way)
    - ``collapsed``: within-group std ~0 with mid reward (homogenized scores)
    """
    std = float(group_reward_std_mean)
    r = float(reward_mean)
    if std >= group_std_eps:
        return None
    if r >= reward_hi:
        return "ceiling"
    if r <= reward_lo:
        return "floor"
    return "collapsed"


def early_stop_reason(
    step_signals: Sequence[str | None],
    *,
    patience: int = DEFAULT_EARLY_STOP_PATIENCE,
) -> str | None:
    """If the last ``patience`` steps share the same dead-signal label, stop.

    Different labels (e.g. floor then ceiling) reset the streak — only a
    *stable* dead pattern abandons the run.
    """
    if patience < 1:
        return None
    if len(step_signals) < patience:
        return None
    window = list(step_signals[-patience:])
    if any(s is None for s in window):
        return None
    label = window[0]
    if label is None or any(s != label for s in window):
        return None
    return f"{label}_x{patience}"


def build_plan(base_model: str, **overrides: Any) -> RecipePlan:
    return plan_recipe(
        base_model=base_model,
        pattern=JobPattern.RL_VERIFIABLE,
        overrides=overrides or None,
    )


def push_adapter_snapshot(
    train_client: Any,
    sample_backend: Backend,
    *,
    name: str,
    sample_adapter_id: AdapterId | None = None,
) -> CheckpointRef:
    """Tier-1 weight sync: train snapshot → sample worker ``load_snapshot``.

    ``sample_backend`` must implement :class:`SnapshotLoader` (vLLM sample
    worker, FakeBackend in tests, RemoteBackend over ``anvil serve``).
    Paths must be readable on the sample host (shared FS / same box).
    """
    if not isinstance(sample_backend, SnapshotLoader):
        raise TypeError(
            f"sample backend {type(sample_backend).__name__!r} does not support "
            f"load_snapshot (SnapshotLoader) — use a vLLM sample worker or FakeBackend"
        )
    ref = train_client.snapshot_for_sample(name)
    aid = sample_adapter_id if sample_adapter_id is not None else train_client.adapter_id
    sample_backend.load_snapshot(aid, ref.path)
    return ref


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
    sample_endpoint: str | None = None,
    sample_backend: Backend | None = None,
    sync_every: int = 1,
    sample_adapter_id: str | None = None,
    early_stop: bool = True,
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE,
    early_stop_group_std_eps: float = DEFAULT_GROUP_STD_EPS,
    early_stop_reward_hi: float = DEFAULT_REWARD_HI,
    early_stop_reward_lo: float = DEFAULT_REWARD_LO,
    stop_on_southward: bool | None = None,
    southward_min_steps: int = 5,
    service_client: ServiceClient | None = None,
    training_client: Any | None = None,
    close_clients: bool = True,
    checkpoint_every: int | None = None,
    resume: bool = False,
) -> GRPOResult:
    """Run the GRPO/IS loop.

    Phase 2.5 observability: pass ``run_dir`` to append per-step records to
    ``<run_dir>/metrics.jsonl`` and, when ``probes`` are given, greedy probe
    completions every ``probe_every`` steps to ``probes.jsonl``.

    Phase 2.5 Tier-1 adapter sync: pass ``sample_endpoint`` (e.g.
    ``http://forge:8741`` for a vLLM sample worker) or inject
    ``sample_backend``. Every ``sync_every`` steps the live LoRA is written
    via ``snapshot_for_sample`` and pushed with ``load_snapshot``. Leave both
    unset for Tier-0 in-process sampling from the train backend.

    Early stop (default on): if advantage signal is dead for
    ``early_stop_patience`` consecutive steps — reward ceiling, floor, or
    homogenized mid rewards — abandon the run instead of burning power.

    Client reuse (recipe queue): pass ``service_client`` + ``training_client``
    to continue the same LoRA across stages; set ``close_clients=False`` so the
    queue owns teardown.

    Checkpoint / resume (Expert-v2): ``run_dir`` + ``checkpoint_every=N`` writes
    adapter ``save_state`` + ``resume.json`` every N completed steps; ``resume=True``
    reloads adapter and continues from ``steps_completed`` toward total ``steps``.
    """
    if sync_every < 1:
        raise ValueError(f"sync_every must be >= 1, got {sync_every}")
    if probe_every < 1:
        raise ValueError(f"probe_every must be >= 1, got {probe_every}")
    if early_stop_patience < 1:
        raise ValueError(f"early_stop_patience must be >= 1, got {early_stop_patience}")
    if checkpoint_every is not None and checkpoint_every < 1:
        raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
    if checkpoint_every is not None and not run_dir:
        raise ValueError("checkpoint_every requires run_dir")
    if resume and not run_dir:
        raise ValueError("resume=True requires run_dir (for resume.json)")

    plan = plan or build_plan(base_model, **(overrides or {}))
    k = plan.as_knobs()
    owns_svc = service_client is None
    svc = service_client if service_client is not None else ServiceClient(endpoint=endpoint)
    if training_client is not None:
        tc = training_client
    else:
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
    if stop_on_southward is None:
        stop_on_southward = bool(early_stop and run_dir)

    sample_svc: ServiceClient | None = None
    resolved_sample: Backend | None = sample_backend
    if resolved_sample is None and sample_endpoint:
        sample_svc = ServiceClient(endpoint=sample_endpoint, queue=False)
        resolved_sample = sample_svc.backend
    sample_aid = (
        AdapterId(sample_adapter_id) if sample_adapter_id is not None else tc.adapter_id
    )
    sample_ep_label = sample_endpoint or (
        f"injected:{type(resolved_sample).__name__}" if resolved_sample is not None else None
    )

    prior_losses: list[float] = []
    prior_mean_rewards: list[float] = []
    prior_dead: list[str | None] = []
    losses: list[float] = []
    mean_rewards: list[float] = []
    sync_count = 0
    last_ref: CheckpointRef | None = None
    dead_signals: list[str | None] = []
    stopped_reason: str | None = None
    start_step = 0
    last_ckpt_path: str | None = None
    job = "grpo"

    if resume and run_dir:
        state = load_resume_state(run_dir)
        if state is not None:
            apply_resume_to_client(tc, state)
            start_step = int(state.steps_completed)
            prior_losses = list(state.losses)
            prior_mean_rewards = list(state.mean_reward)
            prior_dead = list(state.dead_signals)
            if writer is not None:
                writer.log_event(
                    step=start_step,
                    event="resume",
                    reason="loaded_resume_json",
                    job=job,
                    steps_completed=start_step,
                    checkpoint_path=state.checkpoint_path,
                )
            if start_step >= steps:
                if sample_svc is not None:
                    sample_svc.close()
                if close_clients and owns_svc:
                    svc.close()
                return GRPOResult(
                    plan=plan,
                    steps_run=0,
                    mean_reward=[],
                    losses=[],
                    adapter_id=str(tc.adapter_id),
                    sync_count=0,
                    early_stop_reason=None,
                    resumed_from_step=start_step,
                    checkpoint_path=state.checkpoint_path,
                )
            # Tier-1 sample worker needs a fresh snapshot after resume (last_ref empty).
            if resolved_sample is not None and start_step < steps:
                last_ref = push_adapter_snapshot(
                    tc,
                    resolved_sample,
                    name=f"grpo-resume-{start_step}",
                    sample_adapter_id=sample_aid,
                )
                sync_count += 1

    def _maybe_checkpoint(step_completed: int, *, force: bool = False) -> None:
        nonlocal last_ckpt_path
        if run_dir is None:
            return
        if not force and checkpoint_every is None:
            return
        if (
            not force
            and checkpoint_every is not None
            and step_completed % checkpoint_every != 0
        ):
            return
        ref = save_train_checkpoint(
            tc,
            run_dir=run_dir,
            job=job,
            steps_completed=step_completed,
            base_model=plan.base_model,
            losses=prior_losses + losses,
            mean_reward=prior_mean_rewards + mean_rewards,
            dead_signals=prior_dead + dead_signals,
        )
        last_ckpt_path = ref.path
        if writer is not None:
            writer.log_event(
                step=step_completed - 1 if step_completed > 0 else 0,
                event="checkpoint",
                reason="periodic" if not force else "final",
                job=job,
                steps_completed=step_completed,
                checkpoint_path=ref.path,
                checkpoint_name=ref.name,
            )

    try:
        for step_ix in range(start_step, steps):
            t0 = time.monotonic()
            adapter_synced = False
            snap_path: str | None = None

            if resolved_sample is None:
                # Tier 0 — sample from the train backend's live adapter
                sc = tc.save_weights_and_get_sampling_client(name=f"grpo-{step_ix}")
                last_ref = sc.checkpoint
            else:
                # Tier 1 — push on cadence, sample from the sample worker
                if step_ix % sync_every == 0:
                    last_ref = push_adapter_snapshot(
                        tc,
                        resolved_sample,
                        name=f"grpo-{step_ix}",
                        sample_adapter_id=sample_aid,
                    )
                    adapter_synced = True
                    sync_count += 1
                    snap_path = last_ref.path
                elif last_ref is None:
                    raise RuntimeError(
                        "sample worker has no adapter yet — first step must sync "
                        "(sync_every>=1 always covers step 0)"
                    )
                sc = SamplingClient(
                    backend=resolved_sample,
                    base_model=plan.base_model,
                    adapter_id=sample_aid,
                    checkpoint=last_ref,
                )

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

            # Probe the LIVE policy (weights used for this step's rollouts)
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

            reward_mean = sum(step_rewards) / max(len(step_rewards), 1)
            group_std_mean = (
                sum(group_stds) / len(group_stds) if group_stds else 0.0
            )
            signal = classify_dead_step(
                reward_mean=reward_mean,
                group_reward_std_mean=group_std_mean,
                group_std_eps=early_stop_group_std_eps,
                reward_hi=early_stop_reward_hi,
                reward_lo=early_stop_reward_lo,
            )
            dead_signals.append(signal)

            # Skip optim when signal is already dead this step — saves a bit of
            # power; we still log then check patience.
            if signal is None:
                fb = tc.forward_backward(batch, loss_fn=plan.loss_fn).result()
                tc.optim_step(AdamParams(learning_rate=plan.learning_rate)).result()
                step_loss = fb.loss
                fb_metrics = dict(fb.metrics)
            else:
                step_loss = 0.0
                fb_metrics = {
                    "skipped_optim": 1.0,
                    "dead_signal": 1.0,
                }

            losses.append(step_loss)
            mean_rewards.append(reward_mean)
            if writer is not None:
                writer.log_step(
                    step=step_ix,
                    reward_mean=reward_mean,
                    reward_std=(
                        statistics.pstdev(step_rewards) if len(step_rewards) > 1 else 0.0
                    ),
                    group_reward_std_mean=group_std_mean,
                    loss=step_loss,
                    n_datums=len(batch),
                    fb_metrics=fb_metrics,
                    wall_time_s=time.monotonic() - t0,
                    adapter_synced=adapter_synced if resolved_sample is not None else None,
                    snapshot_path=snap_path,
                    sample_endpoint=sample_ep_label if resolved_sample is not None else None,
                )

            completed = step_ix + 1
            _maybe_checkpoint(completed)

            if early_stop:
                reason = early_stop_reason(
                    prior_dead + dead_signals, patience=early_stop_patience
                )
                if reason is not None:
                    stopped_reason = reason
                    if writer is not None:
                        writer.log_event(
                            step=step_ix,
                            event="early_stop",
                            reason=reason,
                            reward_mean=reward_mean,
                            group_reward_std_mean=group_std_mean,
                            patience=early_stop_patience,
                        )
                    break

            if stop_on_southward and run_dir:
                from anvil.observe.southward import maybe_stop_on_southward

                sw = maybe_stop_on_southward(
                    run_dir,
                    step=step_ix,
                    enabled=True,
                    min_steps=southward_min_steps,
                )
                if sw is not None:
                    stopped_reason = sw
                    if writer is not None:
                        writer.log_event(
                            step=step_ix,
                            event="early_stop",
                            reason=sw,
                            reward_mean=reward_mean,
                            group_reward_std_mean=group_std_mean,
                            trigger="southward",
                        )
                    break

        if checkpoint_every is not None and run_dir and losses:
            completed_total = start_step + len(losses)
            if last_ckpt_path is None or completed_total % checkpoint_every != 0:
                _maybe_checkpoint(completed_total, force=True)
    finally:
        if sample_svc is not None:
            sample_svc.close()
        if close_clients and owns_svc:
            svc.close()

    return GRPOResult(
        plan=plan,
        steps_run=len(losses),
        mean_reward=mean_rewards,
        losses=losses,
        adapter_id=str(tc.adapter_id),
        sync_count=sync_count,
        early_stop_reason=stopped_reason,
        resumed_from_step=start_step,
        checkpoint_path=last_ckpt_path,
    )


def _exact_match_toy(_text: str, tokens: Sequence[int]) -> float:
    """Placeholder verifiable reward — prefer even token ids (demo only)."""
    if not tokens:
        return 0.0
    return 1.0 if (sum(tokens) % 2 == 0) else 0.0
