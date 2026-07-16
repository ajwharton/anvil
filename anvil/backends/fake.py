"""In-memory fake backend for Phase 0 golden tests (no GPU).

Simulates LoRA session bookkeeping, a toy CE loss, Adam step counter,
deterministic sampling, and PEFT-dir export. Not a real trainer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from anvil.protocol.types import (
    AdamParams,
    AdapterId,
    CheckpointRef,
    Datum,
    ExportFormat,
    ExportResult,
    ForwardBackwardOutput,
    LossFn,
    ModelInput,
    OptimStepOutput,
    SampledSequence,
    SampleResult,
    SamplingParams,
    TrainConfig,
)


def _normalize_loss(loss_fn: LossFn | str) -> str:
    if isinstance(loss_fn, LossFn):
        return loss_fn.value
    return str(loss_fn)


@dataclass
class _Session:
    config: TrainConfig
    step: int = 0
    # Toy "adapter weights": one scalar per recent train token vocab hash bucket
    weights: dict[int, float] = field(default_factory=dict)
    # Accumulated grad from last forward_backward (toy)
    grads: dict[int, float] = field(default_factory=dict)
    pending_grad: bool = False
    checkpoints: dict[str, dict[str, Any]] = field(default_factory=dict)
    sampler_snapshots: dict[str, dict[int, float]] = field(default_factory=dict)


class FakeBackend:
    """Deterministic, dependency-free backend for API golden tests."""

    name = "fake"

    def __init__(self, *, root: str | Path | None = None) -> None:
        self._sessions: dict[str, _Session] = {}
        self._root = Path(root) if root is not None else Path(os.environ.get("ANVIL_FAKE_ROOT", ".anvil-fake"))
        self._root.mkdir(parents=True, exist_ok=True)

    def create_lora_session(self, config: TrainConfig) -> AdapterId:
        aid = AdapterId(value=f"adapter-{uuid.uuid4().hex[:12]}")
        self._sessions[aid.value] = _Session(config=config)
        return aid

    def _get(self, adapter_id: AdapterId) -> _Session:
        try:
            return self._sessions[adapter_id.value]
        except KeyError as e:
            raise KeyError(f"unknown adapter: {adapter_id}") from e

    def forward_backward(
        self,
        adapter_id: AdapterId,
        data: Sequence[Datum],
        loss_fn: LossFn | str,
    ) -> ForwardBackwardOutput:
        sess = self._get(adapter_id)
        name = _normalize_loss(loss_fn)
        if name not in {
            LossFn.CROSS_ENTROPY.value,
            LossFn.IMPORTANCE_SAMPLING.value,
            LossFn.PPO.value,
            LossFn.CISPO.value,
            LossFn.DRO.value,
            LossFn.DPO.value,
        }:
            raise ValueError(f"unknown loss_fn: {name!r}")

        if not data:
            raise ValueError("forward_backward requires non-empty data")

        total_loss = 0.0
        n_tokens = 0
        grads: dict[int, float] = {}

        for datum in data:
            tokens = datum.model_input.token_ids()
            targets = _as_int_list(datum.loss_fn_inputs.get("target_tokens", tokens[1:] if tokens else []))
            weights = _as_float_list(
                datum.loss_fn_inputs.get("weights", [1.0] * len(targets))
            )
            if len(weights) < len(targets):
                weights = weights + [1.0] * (len(targets) - len(weights))

            logprobs_in = datum.loss_fn_inputs.get("logprobs")
            advantages = datum.loss_fn_inputs.get("advantages")

            for i, tgt in enumerate(targets):
                w = weights[i] if i < len(weights) else 1.0
                if w == 0.0:
                    continue
                # Toy logit: weight for token id bucket
                key = int(tgt) % 10_007
                logit = sess.weights.get(key, 0.0)
                # Fake logprob ~ -log(1+e^{-logit}) style scalar
                log_p = -math.log1p(math.exp(-logit))
                if name == LossFn.CROSS_ENTROPY.value:
                    loss_i = -log_p
                    grads[key] = grads.get(key, 0.0) + w * (1.0 - 1.0 / (1.0 + math.exp(-logit)))
                elif name in {
                    LossFn.IMPORTANCE_SAMPLING.value,
                    LossFn.PPO.value,
                    LossFn.CISPO.value,
                    LossFn.DRO.value,
                }:
                    old_lp = float(logprobs_in[i]) if logprobs_in is not None else log_p
                    adv = float(advantages[i]) if advantages is not None else 1.0
                    ratio = math.exp(log_p - old_lp)
                    if name == LossFn.PPO.value:
                        # unclipped toy; clip lands in real loss module later
                        loss_i = -min(ratio, 1.2) * adv
                    else:
                        loss_i = -ratio * adv
                    grads[key] = grads.get(key, 0.0) + w * loss_i * 0.01
                else:  # dpo stub
                    loss_i = abs(logit) * 0.1
                    grads[key] = grads.get(key, 0.0) + w * 0.01

                total_loss += w * loss_i
                n_tokens += 1

        mean_loss = total_loss / max(n_tokens, 1)
        sess.grads = grads
        sess.pending_grad = True
        return ForwardBackwardOutput(
            loss=mean_loss,
            metrics={"n_tokens": float(n_tokens), "n_examples": float(len(data)), "loss_fn": 0.0},
        )

    def optim_step(self, adapter_id: AdapterId, adam: AdamParams) -> OptimStepOutput:
        sess = self._get(adapter_id)
        if not sess.pending_grad:
            # Allow no-op steps for API flexibility; real backend may warn
            pass
        lr = adam.learning_rate
        for k, g in sess.grads.items():
            sess.weights[k] = sess.weights.get(k, 0.0) - lr * g
        sess.grads = {}
        sess.pending_grad = False
        sess.step += 1
        return OptimStepOutput(step=sess.step, metrics={"lr": lr})

    def save_state(self, adapter_id: AdapterId, name: str) -> CheckpointRef:
        sess = self._get(adapter_id)
        payload = {
            "step": sess.step,
            "weights": dict(sess.weights),
            "base_model": sess.config.base_model,
            "rank": sess.config.lora.rank,
        }
        sess.checkpoints[name] = payload
        path = self._root / adapter_id.value / "train_state" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return CheckpointRef(name=name, path=str(path), kind="train_state")

    def load_state(self, adapter_id: AdapterId, ref: CheckpointRef | str) -> None:
        sess = self._get(adapter_id)
        name = ref.name if isinstance(ref, CheckpointRef) else ref
        if name in sess.checkpoints:
            payload = sess.checkpoints[name]
        else:
            path = Path(ref.path) if isinstance(ref, CheckpointRef) else self._root / adapter_id.value / "train_state" / name
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        sess.step = int(payload["step"])
        sess.weights = {int(k): float(v) for k, v in payload["weights"].items()}
        sess.grads = {}
        sess.pending_grad = False

    def snapshot_for_sample(self, adapter_id: AdapterId, name: str) -> CheckpointRef:
        sess = self._get(adapter_id)
        snap = dict(sess.weights)
        sess.sampler_snapshots[name] = snap
        path = self._root / adapter_id.value / "sampler_weights" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"weights": snap, "step": sess.step}), encoding="utf-8")
        return CheckpointRef(name=name, path=str(path), kind="sampler")

    def sample(
        self,
        *,
        base_model: str,
        adapter_id: AdapterId | None,
        prompt: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int = 1,
        include_prompt_logprobs: bool = False,
    ) -> SampleResult:
        weights: dict[int, float] = {}
        if adapter_id is not None:
            sess = self._get(adapter_id)
            # Prefer latest sampler snapshot if any; else live weights
            if sess.sampler_snapshots:
                last = next(reversed(sess.sampler_snapshots))
                weights = sess.sampler_snapshots[last]
            else:
                weights = sess.weights

        prompt_tokens = prompt.token_ids()
        seed = sampling_params.seed
        if seed is None:
            seed = int(hashlib.sha256(bytes(prompt_tokens[:32])).hexdigest()[:8], 16)

        sequences: list[SampledSequence] = []
        for s in range(num_samples):
            rng = _LCG(seed + s * 9973)
            toks: list[int] = []
            lps: list[float] = []
            for i in range(sampling_params.max_tokens):
                # Deterministic toy next-token from prompt hash + weights
                bucket = (sum(prompt_tokens) + i + s + int(sum(weights.values()) * 1000)) % 1000
                tok = 1000 + (rng.next() + bucket) % 500
                key = tok % 10_007
                logit = weights.get(key, 0.0)
                lp = -math.log1p(math.exp(-logit))
                toks.append(tok)
                lps.append(lp)
                if sampling_params.temperature <= 0:
                    break
            sequences.append(
                SampledSequence(tokens=tuple(toks), logprobs=tuple(lps), stop_reason="length")
            )

        prompt_lps = None
        if include_prompt_logprobs:
            prompt_lps = tuple(
                None if i == 0 else -1.0 - (t % 7) * 0.1 for i, t in enumerate(prompt_tokens)
            )

        _ = base_model  # reserved for multi-base fake pools
        return SampleResult(sequences=tuple(sequences), prompt_logprobs=prompt_lps)

    def compute_logprobs(
        self,
        *,
        base_model: str,
        adapter_id: AdapterId | None,
        prompt: ModelInput,
    ) -> list[float | None]:
        result = self.sample(
            base_model=base_model,
            adapter_id=adapter_id,
            prompt=prompt,
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
            num_samples=1,
            include_prompt_logprobs=True,
        )
        assert result.prompt_logprobs is not None
        return list(result.prompt_logprobs)

    def export_adapter(
        self,
        adapter_id: AdapterId,
        format: ExportFormat,
        path: str,
    ) -> ExportResult:
        sess = self._get(adapter_id)
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        if format == ExportFormat.PEFT:
            meta = {
                "base_model_name_or_path": sess.config.base_model,
                "r": sess.config.lora.rank,
                "lora_alpha": sess.config.lora.effective_alpha(),
                "anvil_adapter_id": adapter_id.value,
                "step": sess.step,
            }
            (out / "adapter_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            (out / "adapter_model.fake.json").write_text(
                json.dumps({str(k): v for k, v in sess.weights.items()}),
                encoding="utf-8",
            )
        else:
            (out / "export_meta.json").write_text(
                json.dumps({"format": format.value, "step": sess.step}),
                encoding="utf-8",
            )
        return ExportResult(format=format, path=str(out), adapter_id=adapter_id.value)


class _LCG:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def next(self) -> int:
        self.state = (1_664_525 * self.state + 1_013_904_223) & 0xFFFFFFFF
        return self.state


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, ModelInput):
        return value.token_ids()
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    raise TypeError(f"expected list of ints, got {type(value)}")


def _as_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    raise TypeError(f"expected list of floats, got {type(value)}")
