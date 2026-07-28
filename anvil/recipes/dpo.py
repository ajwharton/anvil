"""Preference DPO recipe — paired data + observe SSOT (Expert-v1).

Uses named loss ``dpo`` on the four-verb path. Real LocalBackend DPO math is
still research-grade; FakeBackend provides a deterministic stub so metrics,
early-stop, and probes can ship on the same observe surface as SFT/GRPO.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from anvil.client.service import ServiceClient
from anvil.client.training import TrainingClient
from anvil.observe.metrics import RunMetricsWriter
from anvil.protocol.messages import Example, Message, TextPart
from anvil.protocol.types import AdamParams, Datum, LoraTargets
from anvil.recipes.profiles import JobPattern, RecipePlan, plan_recipe
from anvil.recipes.sft import (
    DEFAULT_SFT_EARLY_STOP_ABS_EPS,
    DEFAULT_SFT_EARLY_STOP_PATIENCE,
    DEFAULT_SFT_EARLY_STOP_REL_EPS,
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
) -> DPOResult:
    """DPO-style loop: preferred+rejected batch → named ``dpo`` loss → observe.

    Logs ``job=dpo`` steps (loss, n_pairs, length_bias, margin proxy). Early-stop
    reuses SFT plateau logic on the loss curve.
    """
    mode = str(early_stop_mode or "production").lower().strip()
    if mode not in {"production", "calibration"}:
        raise ValueError(f"early_stop_mode must be production|calibration, got {early_stop_mode!r}")
    if early_stop is None:
        early_stop = mode == "production"
    if early_stop_patience is None:
        early_stop_patience = (
            DEFAULT_SFT_EARLY_STOP_PATIENCE if mode == "production" else 10**9
        )

    plan = plan or build_plan(base_model, **(overrides or {}))
    k = plan.as_knobs()
    renderer = ToyTextRenderer()
    pair_list = list(pairs) if pairs is not None else _toy_pairs()
    if not pair_list:
        raise ValueError("run_dpo requires at least one PreferencePair")

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
    writer = RunMetricsWriter(run_dir) if run_dir else None
    n = steps if steps is not None else min(5, plan.max_steps)
    losses: list[float] = []
    stopped: str | None = None
    steps_run = 0

    for step_ix in range(n):
        t0 = time.monotonic()
        fb = tc.forward_backward(batch, loss_fn="dpo").result()
        tc.optim_step(AdamParams(learning_rate=plan.learning_rate)).result()
        losses.append(fb.loss)
        steps_run = step_ix + 1
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
                job="dpo",
            )
        if early_stop:
            reason = sft_early_stop_reason(
                losses,
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
                        job="dpo",
                        length_bias=length_bias,
                    )
                break

    export_path = None
    if export_dir:
        export_path = tc.export_adapter(
            export_dir, format=plan.export_hint if plan.export_hint == "peft" else "peft"
        ).path
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
    )


__all__ = [
    "DPOResult",
    "PreferencePair",
    "build_plan",
    "run_dpo",
]
