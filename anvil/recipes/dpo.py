"""Preference DPO recipe — paired data + observe SSOT (Expert-v1).

Uses named loss ``dpo`` on the four-verb path:

- **LocalBackend** — reference-free Bradley-Terry DPO
  (``-log σ(β · (log π_θ(y_w) − log π_θ(y_l)))``); optional ``ref_logprob``
  on datums enables classic DPO with an external π_ref.
- **FakeBackend** — deterministic stub so CI can exercise metrics, early-stop,
  probes, and southward without torch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from anvil.client.service import ServiceClient
from anvil.client.training import TrainingClient
from anvil.observe.metrics import RunMetricsWriter
from anvil.protocol.messages import Example, Message, TextPart
from anvil.protocol.types import AdamParams, Datum, LoraTargets, SamplingParams
from anvil.recipes.checkpoint import (
    apply_resume_to_client,
    load_resume_state,
    save_train_checkpoint,
)
from anvil.recipes.profiles import JobPattern, RecipePlan, plan_recipe
from anvil.recipes.sft import (
    DEFAULT_SFT_EARLY_STOP_ABS_EPS,
    DEFAULT_SFT_EARLY_STOP_PATIENCE,
    DEFAULT_SFT_EARLY_STOP_REL_EPS,
    _decode_tokens,
    _match_score,
    resolve_export_format,
    sft_early_stop_reason,
)
from anvil.render.text import ToyTextRenderer


@dataclass(frozen=True, slots=True)
class PreferencePair:
    """One preferred / rejected completion for the same prompt."""

    prompt: str
    preferred: str
    rejected: str
    meta: dict[str, Any] | None = None


@dataclass
class DPOResult:
    plan: RecipePlan
    steps_run: int
    losses: list[float]
    adapter_id: str
    export_path: str | None = None
    run_dir: str | None = None
    early_stop_reason: str | None = None
    mean_length_bias: float | None = None
    n_probe_records: int = 0
    resumed_from_step: int = 0
    checkpoint_path: str | None = None


def build_plan(base_model: str, **overrides: Any) -> RecipePlan:
    return plan_recipe(
        base_model=base_model,
        pattern=JobPattern.PREFERENCE_DPO,
        overrides=overrides or None,
    )


def _pair_to_datums(pair: PreferencePair, *, renderer: ToyTextRenderer) -> tuple[Datum, Datum, int, int]:
    """Render preferred and rejected as CE-style datums (shared prompt tokens)."""
    pref_ex = Example(
        messages=(
            Message(role="user", content=(TextPart(text=pair.prompt),)),
            Message(role="assistant", content=(TextPart(text=pair.preferred),)),
        )
    )
    rej_ex = Example(
        messages=(
            Message(role="user", content=(TextPart(text=pair.prompt),)),
            Message(role="assistant", content=(TextPart(text=pair.rejected),)),
        )
    )
    pref_d = renderer.render_example_for_sft(pref_ex)
    rej_d = renderer.render_example_for_sft(rej_ex)
    # Mark for backends that understand preference later
    pref_d.loss_fn_inputs["preference"] = "preferred"
    rej_d.loss_fn_inputs["preference"] = "rejected"
    n_pref = len(pref_d.loss_fn_inputs.get("target_tokens") or [])
    n_rej = len(rej_d.loss_fn_inputs.get("target_tokens") or [])
    return pref_d, rej_d, n_pref, n_rej


def _toy_pairs() -> list[PreferencePair]:
    return [
        PreferencePair(
            prompt="2+2?",
            preferred="4",
            rejected="5 because five is bigger",
        ),
        PreferencePair(
            prompt="Capital of France?",
            preferred="Paris",
            rejected="I think it might be Lyon or maybe Marseille actually",
        ),
    ]


def run_dpo(
    *,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    pairs: Sequence[PreferencePair] | None = None,
    steps: int | None = None,
    endpoint: str = "fake://",
    export_dir: str | None = None,
    plan: RecipePlan | None = None,
    overrides: dict[str, Any] | None = None,
    run_dir: str | None = None,
    early_stop: bool | None = None,
    early_stop_mode: str = "production",
    early_stop_patience: int | None = None,
    early_stop_rel_eps: float = DEFAULT_SFT_EARLY_STOP_REL_EPS,
    early_stop_abs_eps: float = DEFAULT_SFT_EARLY_STOP_ABS_EPS,
    probes: Sequence[PreferencePair] | None = None,
    probe_every: int = 1,
    stop_on_southward: bool | None = None,
    southward_min_steps: int = 8,
    checkpoint_every: int | None = None,
    resume: bool = False,
    service_client: ServiceClient | None = None,
    training_client: TrainingClient | None = None,
    close_clients: bool = True,
) -> DPOResult:
    """DPO-style loop: preferred+rejected batch → named ``dpo`` loss → observe.

    Logs ``job=dpo`` steps (loss, n_pairs, length_bias, margin proxy). Early-stop
    reuses SFT plateau logic on the loss curve. Held-out ``probes`` (pairs) sample
    the live adapter on the prompt and score match to ``preferred``.

    ``stop_on_southward`` (production default with ``run_dir``) aborts on cliff
    flags such as ``length_bias_spike`` or probe regression.

    Checkpoint / resume (Expert-v2 parity with SFT/GRPO): ``run_dir`` +
    ``checkpoint_every=N`` writes adapter ``save_state`` + ``resume.json``;
    ``resume=True`` continues from ``steps_completed`` toward total ``steps``.

    Client reuse (meta-recipe / stage queue): pass ``service_client`` +
    ``training_client`` to continue the same LoRA (e.g. SFT → DPO on one
    adapter); set ``close_clients=False`` so the caller owns teardown.
    """
    if probe_every < 1:
        raise ValueError(f"probe_every must be >= 1, got {probe_every}")
    if checkpoint_every is not None and checkpoint_every < 1:
        raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
    if checkpoint_every is not None and not run_dir:
        raise ValueError("checkpoint_every requires run_dir")
    if resume and not run_dir:
        raise ValueError("resume=True requires run_dir (for resume.json)")
    mode = str(early_stop_mode or "production").lower().strip()
    if mode not in {"production", "calibration"}:
        raise ValueError(f"early_stop_mode must be production|calibration, got {early_stop_mode!r}")
    if early_stop is None:
        early_stop = mode == "production"
    if early_stop_patience is None:
        early_stop_patience = (
            DEFAULT_SFT_EARLY_STOP_PATIENCE if mode == "production" else 10**9
        )
    if stop_on_southward is None:
        stop_on_southward = bool(early_stop and mode == "production" and run_dir)

    plan = plan or build_plan(base_model, **(overrides or {}))
    k = plan.as_knobs()
    renderer = ToyTextRenderer()
    pair_list = list(pairs) if pairs is not None else _toy_pairs()
    if not pair_list:
        raise ValueError("run_dpo requires at least one PreferencePair")
    probe_list = list(probes) if probes is not None else []

    batch: list[Datum] = []
    pref_tok = 0
    rej_tok = 0
    for p in pair_list:
        pref_d, rej_d, np_, nr_ = _pair_to_datums(p, renderer=renderer)
        batch.extend([pref_d, rej_d])
        pref_tok += np_
        rej_tok += nr_
    n_pairs = len(pair_list)
    length_bias = float(pref_tok - rej_tok) / max(n_pairs, 1)
    # Margin proxy: prefer shorter preferred (anti length-bias signal)
    margin = -length_bias

    svc = service_client if service_client is not None else ServiceClient(endpoint=endpoint)
    owns_svc = service_client is None
    if training_client is not None:
        tc = training_client
    else:
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
    n = steps if steps is not None else min(5, plan.max_steps)
    prior_losses: list[float] = []
    losses: list[float] = []
    stopped: str | None = None
    steps_run = 0
    n_probe_records = 0
    max_tok = int(getattr(plan, "max_tokens", None) or 32)
    start_step = 0
    last_ckpt_path: str | None = None
    job = "dpo"

    if resume and run_dir:
        state = load_resume_state(run_dir)
        if state is not None:
            apply_resume_to_client(tc, state)
            start_step = int(state.steps_completed)
            prior_losses = list(state.losses)
            if writer is not None:
                writer.log_event(
                    step=start_step,
                    event="resume",
                    reason="loaded_resume_json",
                    job=job,
                    steps_completed=start_step,
                    checkpoint_path=state.checkpoint_path,
                )
            if start_step >= n:
                if close_clients and owns_svc:
                    svc.close()
                return DPOResult(
                    plan=plan,
                    steps_run=0,
                    losses=[],
                    adapter_id=str(tc.adapter_id),
                    export_path=None,
                    run_dir=run_dir,
                    early_stop_reason=None,
                    mean_length_bias=length_bias,
                    n_probe_records=0,
                    resumed_from_step=start_step,
                    checkpoint_path=state.checkpoint_path,
                )

    def _all_losses() -> list[float]:
        return prior_losses + losses

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
            losses=_all_losses(),
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

    for step_ix in range(start_step, n):
        t0 = time.monotonic()
        fb = tc.forward_backward(batch, loss_fn="dpo").result()
        tc.optim_step(AdamParams(learning_rate=plan.learning_rate)).result()
        losses.append(fb.loss)
        steps_run = step_ix + 1 - start_step
        if writer is not None:
            fb_metrics = {
                str(kk): float(vv)
                for kk, vv in dict(fb.metrics or {}).items()
                if isinstance(vv, (int, float))
            }
            writer.log_dpo_step(
                step=step_ix,
                loss=fb.loss,
                n_pairs=n_pairs,
                preferred_tokens=float(pref_tok),
                rejected_tokens=float(rej_tok),
                margin=margin,
                length_bias=length_bias,
                fb_metrics=fb_metrics,
                wall_time_s=time.monotonic() - t0,
                job=job,
            )
            if probe_list and step_ix % probe_every == 0:
                sc = tc.save_weights_and_get_sampling_client(name=f"dpo-probe-{step_ix}")
                for probe_ix, pp in enumerate(probe_list):
                    prompt = renderer.render_prompt(
                        (Message(role="user", content=(TextPart(text=pp.prompt),)),)
                    )
                    out = sc.sample(
                        prompt,
                        SamplingParams(max_tokens=max_tok, temperature=0.0, seed=0),
                        num_samples=1,
                    ).result()
                    seq = out.sequences[0] if out.sequences else None
                    toks = tuple(seq.tokens) if seq is not None else ()
                    text = _decode_tokens(renderer, toks)
                    writer.log_probe(
                        step=step_ix,
                        probe_idx=probe_ix,
                        tokens=toks,
                        text=text,
                        reward=_match_score(text, pp.preferred),
                        target=pp.preferred,
                        job=job,
                    )
                    n_probe_records += 1

        completed = step_ix + 1
        _maybe_checkpoint(completed)

        if early_stop:
            reason = sft_early_stop_reason(
                _all_losses(),
                patience=early_stop_patience,
                rel_eps=early_stop_rel_eps,
                abs_eps=early_stop_abs_eps,
            )
            if reason is not None:
                stopped = f"dpo_{reason}"
                if writer is not None:
                    writer.log_event(
                        step=step_ix,
                        event="early_stop",
                        reason=stopped,
                        mode=mode,
                        patience=early_stop_patience,
                        job=job,
                        length_bias=length_bias,
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
                stopped = sw
                if writer is not None:
                    writer.log_event(
                        step=step_ix,
                        event="early_stop",
                        reason=sw,
                        mode=mode,
                        job=job,
                        trigger="southward",
                        length_bias=length_bias,
                    )
                break

    if checkpoint_every is not None and run_dir and steps_run > 0:
        completed_total = start_step + steps_run
        if last_ckpt_path is None or completed_total % checkpoint_every != 0:
            _maybe_checkpoint(completed_total, force=True)

    export_path = None
    if export_dir:
        export_path = tc.export_adapter(
            export_dir, format=resolve_export_format(plan)
        ).path
    if close_clients and owns_svc:
        svc.close()
    return DPOResult(
        plan=plan,
        steps_run=steps_run,
        losses=losses,
        adapter_id=str(tc.adapter_id),
        export_path=export_path,
        run_dir=run_dir,
        early_stop_reason=stopped,
        mean_length_bias=length_bias,
        n_probe_records=n_probe_records,
        resumed_from_step=start_step,
        checkpoint_path=last_ckpt_path,
    )


__all__ = [
    "DPOResult",
    "PreferencePair",
    "build_plan",
    "run_dpo",
]
