"""Local in-process backend — real torch + PEFT training behind the four verbs.

Phase 1. Single host, single GPU (CPU works for smoke tests). The verbs are
implemented by hand — no HF Trainer: ``forward_backward`` runs the forward
pass, computes a named server-side loss, and calls ``.backward()``, leaving
grads on the LoRA params; ``optim_step`` applies AdamW. A Trainer-style
abstraction would swallow exactly the verb separation that is Anvil's API
contract.

Losses v0: ``cross_entropy`` (SFT). The IS/PPO/DPO family lands with the
Phase 2 RL workers (they need old-policy logprob plumbing, not just a loss).

Optional deps: ``pip install anvil-train[local]`` (torch, transformers, peft).
Nothing in this module imports those at module load time.
"""

from __future__ import annotations

import json
import os
import uuid
import warnings
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

_SUPPORTED_LOSSES = {LossFn.CROSS_ENTROPY.value}

# Opinionated capacity gates (learned the hard way): a LoRA adapter can only
# steer a model through its hidden width. sshleifer/tiny-gpt2 (hidden_size=2,
# vocab 50k) pins loss at chance no matter the LR — attention-LoRA physically
# cannot move a 50k-way softmax through a 2-d bottleneck.
_MIN_HIDDEN_FOR_LORA = 16  # below → blocked (raise ModelTooSmallError)
_WARN_HIDDEN_FOR_LORA = 32  # below → expect glacial learning (warn)


class ModelTooSmallError(ValueError):
    """Base model is architecturally too small for LoRA to learn."""


def _deps() -> tuple[Any, Any, Any]:
    try:
        import peft
        import torch
        import transformers
    except ImportError as e:  # pragma: no cover - depends on env
        raise ImportError(
            "LocalBackend requires torch + transformers + peft; "
            "install with: pip install anvil-train[local]"
        ) from e
    return torch, transformers, peft


def _normalize_loss(loss_fn: LossFn | str) -> str:
    if isinstance(loss_fn, LossFn):
        return loss_fn.value
    return str(loss_fn)


def _stop_string_criteria(
    transformers: Any, tokenizer: Any, prompt_len: int, stop_strings: tuple[str, ...]
) -> Any:
    """Early-exit criteria: True once EVERY row's decoded suffix holds a stop string.

    HF StoppingCriteria halts the whole batch jointly (no per-row exit), so
    per-row correctness is handled by ``_truncate_at_stop``; this just ends
    generation early once all rows are done instead of burning max_tokens.
    """

    class _StopStrings(transformers.StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            for row in input_ids:
                text = tokenizer.decode(
                    row[prompt_len:].tolist(), skip_special_tokens=False
                )
                if not any(s in text for s in stop_strings):
                    return False
            return True

    return transformers.StoppingCriteriaList([_StopStrings()])


def _truncate_at_stop(
    tokenizer: Any, toks: list[int], stop_strings: tuple[str, ...]
) -> tuple[list[int], bool]:
    """Cut tokens at the earliest stop-string occurrence; whole-token granularity.

    Stop strings match on decoded text; a stop string that starts mid-token
    drops that whole token (vLLM convention). Returns (tokens, hit).
    """
    text = tokenizer.decode(toks, skip_special_tokens=False)
    hits = [p for p in (text.find(s) for s in stop_strings) if p >= 0]
    if not hits:
        return toks, False
    cut = min(hits)
    while toks and len(tokenizer.decode(toks, skip_special_tokens=False)) > cut:
        toks.pop()
    return toks, True


@dataclass
class _LocalSession:
    config: TrainConfig
    model: Any  # peft.PeftModel
    tokenizer: Any
    optimizer: Any | None = None
    step: int = 0
    pending_grad: bool = False
    # name → CPU copy of adapter state dict (true snapshot semantics)
    sampler_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    loaded_snapshot: str | None = None


class LocalBackend:
    """Single-host torch + PEFT backend implementing the Backend protocol."""

    name = "local"

    def __init__(
        self,
        *,
        device: str | None = None,
        root: str | Path | None = None,
        target_modules: Sequence[str] | None = None,
        allow_tiny_models: bool = False,
    ) -> None:
        torch, _, _ = _deps()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # fp32 on CPU; bf16 on GPU (LoRA params are small; base stays frozen)
        self.dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self._target_modules = list(target_modules) if target_modules else None
        self._allow_tiny_models = allow_tiny_models
        self._sessions: dict[str, _LocalSession] = {}
        self._base_models: dict[str, tuple[Any, Any]] = {}  # base_model → (model, tokenizer)
        self._root = Path(root) if root is not None else Path(
            os.environ.get("ANVIL_LOCAL_ROOT", ".anvil-local")
        )
        self._root.mkdir(parents=True, exist_ok=True)

    # --- session lifecycle ---------------------------------------------------

    def create_lora_session(self, config: TrainConfig) -> AdapterId:
        torch, transformers, peft = _deps()
        non_text = [m for m in config.modalities if m != "text"]
        if non_text:
            raise NotImplementedError(
                f"LocalBackend is text-only in Phase 1; modalities {non_text} "
                f"land with the Phase 3 VLM path"
            )

        tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.base_model, dtype=self.dtype
        )
        self._check_capacity(config.base_model, model.config)
        lora_cfg = peft.LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.effective_alpha(),
            lora_dropout=config.lora.dropout,
            task_type="CAUSAL_LM",
            # None → peft's per-architecture default (e.g. gpt2 → ["c_attn"])
            target_modules=self._target_modules,
        )
        model = peft.get_peft_model(model, lora_cfg)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if n_trainable == 0:
            raise ValueError(
                f"LoRA attached zero trainable params on {config.base_model} — "
                f"target_modules={self._target_modules!r} matched nothing. Pass "
                f"target_modules= to LocalBackend for this architecture."
            )
        model.to(self.device)
        model.train()

        aid = AdapterId(value=f"adapter-{uuid.uuid4().hex[:12]}")
        self._sessions[aid.value] = _LocalSession(
            config=config, model=model, tokenizer=tokenizer
        )
        return aid

    def _get(self, adapter_id: AdapterId) -> _LocalSession:
        try:
            return self._sessions[adapter_id.value]
        except KeyError as e:
            raise KeyError(f"unknown adapter: {adapter_id}") from e

    def _check_capacity(self, base_model: str, hf_config: Any) -> None:
        hidden = getattr(hf_config, "hidden_size", None) or getattr(
            hf_config, "n_embd", None
        )
        if hidden is None:
            return
        vocab = getattr(hf_config, "vocab_size", None)
        if hidden < _MIN_HIDDEN_FOR_LORA and not self._allow_tiny_models:
            raise ModelTooSmallError(
                f"{base_model} has hidden_size={hidden}; a LoRA adapter cannot "
                f"meaningfully shift a {vocab or '?'}-token output distribution "
                f"through a {hidden}-wide bottleneck (observed: loss pinned at "
                f"chance regardless of LR). Use a model with hidden_size >= "
                f"{_MIN_HIDDEN_FOR_LORA} (smoke tests: "
                f"hf-internal-testing/tiny-random-gpt2), or pass "
                f"allow_tiny_models=True to LocalBackend if you really mean it."
            )
        if hidden < _WARN_HIDDEN_FOR_LORA:
            warnings.warn(
                f"{base_model} has hidden_size={hidden} — LoRA learning will be "
                f"glacial; prefer hidden_size >= {_WARN_HIDDEN_FOR_LORA}",
                stacklevel=2,
            )

    # --- core verbs ------------------------------------------------------------

    def forward_backward(
        self,
        adapter_id: AdapterId,
        data: Sequence[Datum],
        loss_fn: LossFn | str,
    ) -> ForwardBackwardOutput:
        torch, _, _ = _deps()
        sess = self._get(adapter_id)
        name = _normalize_loss(loss_fn)
        if name not in _SUPPORTED_LOSSES:
            raise NotImplementedError(
                f"loss {name!r} not supported by LocalBackend v0 — the IS/PPO/DPO "
                f"family lands with the Phase 2 RL workers"
            )
        if not data:
            raise ValueError("forward_backward requires non-empty data")

        pad_id = sess.tokenizer.pad_token_id
        rows: list[tuple[list[int], list[int], list[float]]] = []
        for datum in data:
            ids = datum.model_input.token_ids()
            targets = [int(t) for t in datum.loss_fn_inputs.get("target_tokens", [])]
            weights = [float(w) for w in datum.loss_fn_inputs.get("weights", [])]
            if not (len(ids) == len(targets) == len(weights)):
                raise ValueError(
                    f"input/target/weights length mismatch: "
                    f"{len(ids)}/{len(targets)}/{len(weights)} — build data via a "
                    f"renderer (render_example_for_sft) so the causal shift aligns"
                )
            if ids:
                rows.append((ids, targets, weights))
        if not rows:
            raise ValueError("all examples are empty")

        width = max(len(r[0]) for r in rows)
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        weights_t = torch.zeros((len(rows), width), dtype=torch.float32)
        attn = torch.zeros((len(rows), width), dtype=torch.long)
        for i, (ids, targets, weights) in enumerate(rows):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            labels[i, :n] = torch.tensor(targets, dtype=torch.long)
            weights_t[i, :n] = torch.tensor(weights, dtype=torch.float32)
            attn[i, :n] = 1
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        weights_t = weights_t.to(self.device)
        attn = attn.to(self.device)

        # logits[i] predicts labels[i] directly: renderers emit input=ids[:-1],
        # targets=ids[1:], so no extra shift here.
        logits = sess.model(input_ids=input_ids, attention_mask=attn).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        gather_idx = labels.clamp_min(0).unsqueeze(-1)
        tok_lp = logp.gather(-1, gather_idx).squeeze(-1)
        supervised = labels != -100
        w = weights_t * supervised
        loss = -(tok_lp * w).sum() / w.sum().clamp_min(1e-8)
        loss.backward()
        sess.pending_grad = True

        return ForwardBackwardOutput(
            loss=float(loss.detach().cpu()),
            metrics={
                "n_tokens": float(w.sum().item()),
                "n_examples": float(len(rows)),
            },
        )

    def optim_step(self, adapter_id: AdapterId, adam: AdamParams) -> OptimStepOutput:
        torch, _, _ = _deps()
        sess = self._get(adapter_id)
        if sess.optimizer is None:
            params = [p for p in sess.model.parameters() if p.requires_grad]
            sess.optimizer = torch.optim.AdamW(params, lr=adam.learning_rate)
        opt = sess.optimizer
        assert opt is not None
        group = opt.param_groups[0]
        group["lr"] = adam.learning_rate
        group["betas"] = (adam.beta1, adam.beta2)
        group["eps"] = adam.eps
        group["weight_decay"] = adam.weight_decay
        # No pending grad is allowed (matches FakeBackend): a zero-grad step
        # still advances the counter so client loops stay simple.
        opt.step()
        opt.zero_grad(set_to_none=True)
        sess.pending_grad = False
        sess.step += 1
        return OptimStepOutput(step=sess.step, metrics={"lr": adam.learning_rate})

    # --- checkpoints -------------------------------------------------------------

    def save_state(self, adapter_id: AdapterId, name: str) -> CheckpointRef:
        torch, _, _ = _deps()
        sess = self._get(adapter_id)
        path = self._root / adapter_id.value / "train_state" / name
        path.mkdir(parents=True, exist_ok=True)
        sess.model.save_pretrained(path)
        if sess.optimizer is not None:
            torch.save(sess.optimizer.state_dict(), path / "optimizer.pt")
        meta = {
            "step": sess.step,
            "base_model": sess.config.base_model,
            "rank": sess.config.lora.rank,
        }
        (path / "anvil_state.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return CheckpointRef(name=name, path=str(path), kind="train_state")

    def load_state(self, adapter_id: AdapterId, ref: CheckpointRef | str) -> None:
        torch, _, peft = _deps()
        sess = self._get(adapter_id)
        name = ref.name if isinstance(ref, CheckpointRef) else ref
        path = Path(ref.path) if isinstance(ref, CheckpointRef) else (
            self._root / adapter_id.value / "train_state" / name
        )
        from safetensors.torch import load_file

        sd = load_file(str(path / "adapter_model.safetensors"))
        peft.set_peft_model_state_dict(sess.model, sd)
        opt_path = path / "optimizer.pt"
        if opt_path.is_file():
            if sess.optimizer is None:
                params = [p for p in sess.model.parameters() if p.requires_grad]
                sess.optimizer = torch.optim.AdamW(params)
            opt = sess.optimizer
            assert opt is not None
            opt.load_state_dict(
                torch.load(opt_path, map_location=self.device, weights_only=True)
            )
        meta = json.loads((path / "anvil_state.json").read_text(encoding="utf-8"))
        sess.step = int(meta["step"])
        sess.pending_grad = False
        sess.loaded_snapshot = None  # weights changed; snapshot cache invalid

    # --- sampling ----------------------------------------------------------------

    def snapshot_for_sample(self, adapter_id: AdapterId, name: str) -> CheckpointRef:
        _, _, peft = _deps()
        sess = self._get(adapter_id)
        sd = peft.get_peft_model_state_dict(sess.model)
        sess.sampler_snapshots[name] = {
            k: v.detach().cpu().clone() for k, v in sd.items()
        }
        path = self._root / adapter_id.value / "sampler_weights" / name
        path.mkdir(parents=True, exist_ok=True)
        sess.model.save_pretrained(path)
        return CheckpointRef(name=name, path=str(path), kind="sampler")

    def _model_for_sample(
        self, base_model: str, adapter_id: AdapterId | None
    ) -> tuple[Any, Any]:
        torch, transformers, peft = _deps()
        if adapter_id is None:
            if base_model not in self._base_models:
                tok = transformers.AutoTokenizer.from_pretrained(base_model)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                raw = transformers.AutoModelForCausalLM.from_pretrained(
                    base_model, dtype=self.dtype
                )
                raw.to(self.device)
                raw.eval()
                self._base_models[base_model] = (raw, tok)
            return self._base_models[base_model]

        sess = self._get(adapter_id)
        if sess.sampler_snapshots:
            latest = next(reversed(sess.sampler_snapshots))
            if sess.loaded_snapshot != latest:
                peft.set_peft_model_state_dict(
                    sess.model, sess.sampler_snapshots[latest]
                )
                sess.loaded_snapshot = latest
        return sess.model, sess.tokenizer

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
        torch, transformers, _ = _deps()
        stop_strings = tuple(s for s in (sampling_params.stop or ()) if s)
        model, tokenizer = self._model_for_sample(base_model, adapter_id)

        prompt_ids = prompt.token_ids()
        if not prompt_ids:
            raise ValueError("empty prompt")
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt_len = len(prompt_ids)

        if sampling_params.seed is not None:
            torch.manual_seed(sampling_params.seed)

        do_sample = sampling_params.temperature > 0
        if not do_sample and num_samples != 1:
            raise ValueError(
                "greedy sampling (temperature<=0) returns a single sequence; "
                "set temperature>0 for num_samples>1"
            )
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": sampling_params.max_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num_samples,
            "output_scores": True,
            "return_dict_in_generate": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = sampling_params.temperature
            if sampling_params.top_p < 1.0:
                gen_kwargs["top_p"] = sampling_params.top_p
            if sampling_params.top_k is not None:
                gen_kwargs["top_k"] = sampling_params.top_k
        if stop_strings:
            gen_kwargs["stopping_criteria"] = _stop_string_criteria(
                transformers, tokenizer, prompt_len, stop_strings
            )

        was_training = model.training
        model.eval()
        with torch.no_grad():
            out = model.generate(input_ids, **gen_kwargs)
            prompt_lps = None
            if include_prompt_logprobs:
                prompt_lps = self._prompt_logprobs(model, input_ids)
        if was_training:
            model.train()

        eos_id = tokenizer.eos_token_id
        sequences: list[SampledSequence] = []
        for i in range(out.sequences.shape[0]):
            new_tokens = out.sequences[i][prompt_len:]
            toks = [int(t) for t in new_tokens]
            lps: list[float] = []
            if out.scores:
                for t, score in enumerate(out.scores):
                    if t >= len(toks):
                        break
                    lp = torch.log_softmax(score[i].float(), dim=-1)
                    lps.append(float(lp[toks[t]]))
            stop_reason = "length"
            if stop_strings:
                toks, hit_stop = _truncate_at_stop(tokenizer, toks, stop_strings)
                lps = lps[: len(toks)]
                if hit_stop:
                    stop_reason = "stop"
            if (
                stop_reason == "length"
                and eos_id is not None
                and toks
                and toks[-1] == eos_id
            ):
                stop_reason = "eos"
            sequences.append(
                SampledSequence(
                    tokens=tuple(toks), logprobs=tuple(lps), stop_reason=stop_reason
                )
            )
        return SampleResult(sequences=tuple(sequences), prompt_logprobs=prompt_lps)

    def _prompt_logprobs(self, model: Any, input_ids: Any) -> tuple[float | None, ...]:
        torch, _, _ = _deps()
        logits = model(input_ids).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        out: list[float | None] = [None]
        for i in range(1, input_ids.shape[1]):
            out.append(float(logp[i - 1, input_ids[0, i]]))
        return tuple(out)

    def compute_logprobs(
        self,
        *,
        base_model: str,
        adapter_id: AdapterId | None,
        prompt: ModelInput,
    ) -> list[float | None]:
        model, _ = self._model_for_sample(base_model, adapter_id)
        prompt_ids = prompt.token_ids()
        if not prompt_ids:
            raise ValueError("empty prompt")
        torch, _, _ = _deps()
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            lps = self._prompt_logprobs(model, input_ids)
        if was_training:
            model.train()
        return list(lps)

    # --- export ------------------------------------------------------------------

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
            sess.model.save_pretrained(out)  # real PEFT dir: adapter_config + safetensors
        elif format == ExportFormat.MERGED_HF:
            merged = sess.model.merge_and_unload()
            merged.save_pretrained(out)
            sess.tokenizer.save_pretrained(out)
        else:
            raise NotImplementedError(
                f"export format {format.value!r} not supported by LocalBackend v0; "
                f"GGUF/ONNX/TRT land with the Phase 4 edge path"
            )
        return ExportResult(format=format, path=str(out), adapter_id=adapter_id.value)
