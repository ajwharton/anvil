"""Basic SFT recipe — CE on assistant tokens (forward_backward + optim_step).

Runs against any ServiceClient backend. Shape/knobs should come from
``plan_recipe`` / model card inspection. Pass a multimodal renderer
(``HFVLMRenderer``) for vision Examples.

Observability (P3.6 / Expert-v0): pass ``run_dir`` to append per-step
records (loss, wall, n_image_refs, n_tokens) to ``<run_dir>/metrics.jsonl``
via the same :class:`~anvil.observe.metrics.RunMetricsWriter` SSOT as GRPO.
Pass ``probes`` (held-out :class:`~anvil.protocol.messages.Example` rows) to
sample the live adapter every ``probe_every`` steps into ``probes.jsonl``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from anvil.client.service import ServiceClient
from anvil.client.training import TrainingClient
from anvil.observe.metrics import RunMetricsWriter
from anvil.protocol.messages import Example, Message, TextPart
from anvil.protocol.types import AdamParams, Datum, LoraTargets, ModelInput, SamplingParams
from anvil.recipes.checkpoint import (
    apply_resume_to_client,
    load_resume_state,
    save_train_checkpoint,
)
from anvil.recipes.profiles import JobPattern, RecipePlan, plan_recipe
from anvil.render.text import ToyTextRenderer


class _SFTRenderer(Protocol):
    def render_example_for_sft(self, example: Example) -> Datum: ...


# Production default: stop after this many consecutive non-improving steps.
# Type-scoped; VLM/SFT share this prior until the personal recipe book learns better.
DEFAULT_SFT_EARLY_STOP_PATIENCE = 40
DEFAULT_SFT_EARLY_STOP_REL_EPS = 0.01  # relative loss drop required to count as improve
# Absolute floor: once loss is tiny, relative-only checks never plateau (float noise).
DEFAULT_SFT_EARLY_STOP_ABS_EPS = 1e-4
# Calibration mode: effectively no early-stop (still capped by ``steps``).
CALIBRATION_SFT_EARLY_STOP_PATIENCE = 10**9


@dataclass
class SFTResult:
    plan: RecipePlan
    steps_run: int
    losses: list[float]
    adapter_id: str
    export_path: str | None = None
    run_dir: str | None = None
    n_probe_records: int = 0
    early_stop_reason: str | None = None
    resumed_from_step: int = 0
    checkpoint_path: str | None = None


def sft_loss_improved(
    prev: float,
    cur: float,
    *,
    rel_eps: float = DEFAULT_SFT_EARLY_STOP_REL_EPS,
    abs_eps: float = DEFAULT_SFT_EARLY_STOP_ABS_EPS,
) -> bool:
    """True if ``cur`` is meaningfully lower than ``prev``.

    Requires a drop of at least ``max(rel_eps * |prev|, abs_eps)`` so that
    near-zero losses do not "improve" forever on float noise (forge dogfood).
    """
    p = float(prev)
    c = float(cur)
    if not (p == p and c == c):  # NaN guard
        return False
    drop = p - c
    if drop <= 0:
        return False
    need = max(float(rel_eps) * max(abs(p), 1e-12), float(abs_eps))
    return drop >= need


def sft_early_stop_reason(
    losses: Sequence[float],
    *,
    patience: int = DEFAULT_SFT_EARLY_STOP_PATIENCE,
    rel_eps: float = DEFAULT_SFT_EARLY_STOP_REL_EPS,
    abs_eps: float = DEFAULT_SFT_EARLY_STOP_ABS_EPS,
) -> str | None:
    """If the last ``patience`` steps each failed to improve vs the prior step, stop.

    Production dogfood for SFT/VLM. Calibration uses a huge patience (or
    early_stop=False) so overshoot runs can map false plateaus.
    """
    if patience < 1 or len(losses) < patience + 1:
        return None
    window = list(losses[-(patience + 1) :])
    for i in range(1, len(window)):
        if sft_loss_improved(
            window[i - 1], window[i], rel_eps=rel_eps, abs_eps=abs_eps
        ):
            return None
    return f"loss_plateau_patience_{patience}"


def build_plan(base_model: str, **overrides: Any) -> RecipePlan:
    return plan_recipe(
        base_model=base_model,
        pattern=JobPattern.SFT_CHAT,
        overrides=overrides or None,
    )


def examples_to_data(
    examples: Sequence[Example],
    *,
    renderer: _SFTRenderer | None = None,
) -> list[Datum]:
    r: _SFTRenderer = renderer if renderer is not None else ToyTextRenderer()
    return [r.render_example_for_sft(ex) for ex in examples]


def count_image_refs(data: Sequence[Datum]) -> int:
    """Sum ``image_refs`` recorded on each datum (renderer / LocalBackend path)."""
    n = 0
    for d in data:
        refs = d.loss_fn_inputs.get("image_refs") or []
        n += len(refs)
    return n


def _assistant_gold(example: Example) -> str | None:
    for m in reversed(example.messages):
        if m.role != "assistant":
            continue
        parts = []
        for p in m.parts():
            if isinstance(p, TextPart):
                parts.append(p.text)
        if parts:
            return "".join(parts).strip()
    return None


def _render_probe_prompt(renderer: Any, example: Example) -> ModelInput:
    """Prompt tokens for sampling (exclude trailing assistant gold)."""
    msgs = list(example.messages)
    if msgs and msgs[-1].role == "assistant":
        msgs = msgs[:-1]
    if hasattr(renderer, "render_prompt"):
        return renderer.render_prompt(msgs)
    if hasattr(renderer, "render_messages"):
        return renderer.render_messages(msgs)
    # Last resort: train render and hope sampler tolerates full sequence
    datum = renderer.render_example_for_sft(example)
    return datum.model_input


def _decode_tokens(renderer: Any, tokens: Sequence[int]) -> str | None:
    if hasattr(renderer, "decode"):
        try:
            return str(renderer.decode(tokens))
        except Exception:
            return None
    return None


def _match_score(text: str | None, gold: str | None) -> float | None:
    if text is None or gold is None:
        return None
    t = text.strip().lower()
    g = gold.strip().lower()
    if not g:
        return None
    if t == g or g in t:
        return 1.0
    return 0.0


def run_sft(
    *,
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    examples: Sequence[Example] | None = None,
    steps: int | None = None,
    endpoint: str = "fake://",
    export_dir: str | None = None,
    plan: RecipePlan | None = None,
    overrides: dict[str, Any] | None = None,
    renderer: _SFTRenderer | None = None,
    run_dir: str | None = None,
    job: str = "sft",
    probes: Sequence[Example] | None = None,
    probe_every: int = 1,
    early_stop: bool | None = None,
    early_stop_mode: str = "production",
    early_stop_patience: int | None = None,
    early_stop_rel_eps: float = DEFAULT_SFT_EARLY_STOP_REL_EPS,
    early_stop_abs_eps: float = DEFAULT_SFT_EARLY_STOP_ABS_EPS,
    stop_on_southward: bool | None = None,
    southward_min_steps: int = 8,
    service_client: ServiceClient | None = None,
    training_client: TrainingClient | None = None,
    close_clients: bool = True,
    checkpoint_every: int | None = None,
    resume: bool = False,
) -> SFTResult:
    """Minimal SFT: create LoRA client → CE steps → optional export.

    Pass ``run_dir`` to write ``metrics.jsonl`` for anvil-web ``/observe``.
    ``job`` labels records (``sft`` or ``vlm_sft``) so the UI charts loss.
    Pass held-out ``probes`` to sample the live adapter into ``probes.jsonl``
    every ``probe_every`` steps (Expert-v0 eyes signal).

    Early-stop (dogfood): ``early_stop_mode="production"`` (default) stops after
    ``early_stop_patience`` consecutive non-improving steps. Use
    ``early_stop_mode="calibration"`` (or ``early_stop=False``) to overshoot
    for plateau mapping — still fully instrumented.

    Southward auto-stop (production default when ``run_dir`` is set): mid-train
    scan of metrics/probes; cliff flags → ``early_stop`` with reason
    ``southward:<flag>``.

    Client reuse (VLM/SFT stage queue): pass ``service_client`` +
    ``training_client`` to continue the same LoRA; set ``close_clients=False``.

    Checkpoint / resume (Expert-v2): pass ``run_dir`` + ``checkpoint_every=N`` to
    ``save_state`` adapter weights and write ``run_dir/resume.json`` every N
    completed steps (and at end / early-stop). Pass ``resume=True`` on a later
    call with the same ``run_dir`` (and same ``steps`` total budget) to load the
    adapter and continue from ``steps_completed`` without replaying prior steps.
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
        raise ValueError(
            f"early_stop_mode must be production|calibration, got {early_stop_mode!r}"
        )
    if early_stop is None:
        early_stop = mode == "production"
    if early_stop_patience is None:
        early_stop_patience = (
            DEFAULT_SFT_EARLY_STOP_PATIENCE
            if mode == "production"
            else CALIBRATION_SFT_EARLY_STOP_PATIENCE
        )
    if early_stop_patience < 1:
        raise ValueError(f"early_stop_patience must be >= 1, got {early_stop_patience}")
    if stop_on_southward is None:
        stop_on_southward = bool(early_stop and mode == "production" and run_dir)

    plan = plan or build_plan(base_model, **(overrides or {}))
    k = plan.as_knobs()
    r: _SFTRenderer = renderer if renderer is not None else ToyTextRenderer()
    owns_svc = service_client is None
    svc = service_client if service_client is not None else ServiceClient(endpoint=endpoint)
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

    data = examples_to_data(examples, renderer=r) if examples else _toy_batch()
    n_image_refs = count_image_refs(data)
    if n_image_refs == 0 and examples:
        n_image_refs = sum(len(ex.image_refs()) for ex in examples)
    writer = RunMetricsWriter(run_dir) if run_dir else None
    probe_list = list(probes) if probes else []
    n = steps if steps is not None else min(5, plan.max_steps)
    prior_losses: list[float] = []
    losses: list[float] = []  # this invocation only
    n_probe_records = 0
    max_tok = int(getattr(plan, "max_tokens", None) or 32)
    stopped_reason: str | None = None
    steps_run = 0
    start_step = 0
    last_ckpt_path: str | None = None

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
                # Already at/past total budget — nothing left to train.
                if close_clients and owns_svc:
                    svc.close()
                return SFTResult(
                    plan=plan,
                    steps_run=0,
                    losses=[],
                    adapter_id=str(tc.adapter_id),
                    export_path=None,
                    run_dir=run_dir,
                    n_probe_records=0,
                    early_stop_reason=None,
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
        fb = tc.forward_backward(data, loss_fn=plan.loss_fn).result()
        tc.optim_step(AdamParams(learning_rate=plan.learning_rate)).result()
        losses.append(fb.loss)
        steps_run = step_ix + 1 - start_step
        if writer is not None:
            fb_metrics = {str(k_): float(v) for k_, v in dict(fb.metrics or {}).items()
                          if isinstance(v, (int, float))}
            writer.log_sft_step(
                step=step_ix,
                loss=fb.loss,
                n_datums=len(data),
                n_image_refs=n_image_refs,
                fb_metrics=fb_metrics,
                wall_time_s=time.monotonic() - t0,
                job=job,
            )
            # Sample-only probes (no forward_backward — avoids polluting LoRA grads)
            if probe_list and step_ix % probe_every == 0:
                sc = tc.save_weights_and_get_sampling_client(name=f"sft-probe-{step_ix}")
                for probe_ix, pex in enumerate(probe_list):
                    prompt = _render_probe_prompt(r, pex)
                    out = sc.sample(
                        prompt,
                        SamplingParams(max_tokens=max_tok, temperature=0.0, seed=0),
                        num_samples=1,
                    ).result()
                    seq = out.sequences[0] if out.sequences else None
                    toks = tuple(seq.tokens) if seq is not None else ()
                    text = _decode_tokens(r, toks)
                    gold = _assistant_gold(pex)
                    writer.log_probe(
                        step=step_ix,
                        probe_idx=probe_ix,
                        tokens=toks,
                        text=text,
                        reward=_match_score(text, gold),
                        target=gold,
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
                stopped_reason = reason
                if writer is not None:
                    writer.log_event(
                        step=step_ix,
                        event="early_stop",
                        reason=reason,
                        mode=mode,
                        patience=early_stop_patience,
                        rel_eps=early_stop_rel_eps,
                        abs_eps=early_stop_abs_eps,
                        job=job,
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
                        mode=mode,
                        job=job,
                        trigger="southward",
                    )
                break

    # Final resume snapshot when periodic checkpoints are enabled (covers early-stop
    # mid-interval and end-of-budget if last step was not a periodic boundary).
    if checkpoint_every is not None and run_dir and steps_run > 0:
        completed_total = start_step + steps_run
        if last_ckpt_path is None or completed_total % checkpoint_every != 0:
            _maybe_checkpoint(completed_total, force=True)

    export_path = None
    if export_dir:
        export_path = tc.export_adapter(
            export_dir, format=plan.export_hint if plan.export_hint == "peft" else "peft"
        ).path

    if close_clients and owns_svc:
        svc.close()
    return SFTResult(
        plan=plan,
        steps_run=steps_run,
        losses=losses,
        adapter_id=str(tc.adapter_id),
        export_path=export_path,
        run_dir=run_dir,
        n_probe_records=n_probe_records,
        early_stop_reason=stopped_reason,
        resumed_from_step=start_step,
        checkpoint_path=last_ckpt_path,
    )


def _toy_batch() -> list[Datum]:
    ex = Example(
        messages=(
            Message(role="user", content=(TextPart(text="2+2?"),)),
            Message(role="assistant", content=(TextPart(text="4"),)),
        )
    )
    return examples_to_data([ex])
