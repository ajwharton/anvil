"""Dedicated vLLM sample worker (Phase 2 sample/train split).

A Backend implementation that serves ONLY the sampling verbs on top of a
vLLM offline engine. Training verbs raise NotImplementedError — the train
side lives on the other node (LocalBackend today).

Run on the sample host::

    anvil serve --backend vllm-sample --model Qwen/Qwen2.5-1.5B-Instruct \
        --host 0.0.0.0 --port 8741

Adapter hot-swap: the train side POSTs a PEFT snapshot dir to
``/v1/adapters/{adapter_id}/load_snapshot``; the worker registers it as a
fresh LoRA request (new int id every load, so vLLM never serves a stale
cached adapter) and subsequent ``sample`` calls with that adapter_id pick
up the new weights without reloading the base model.
"""

from __future__ import annotations

import itertools
import os
from typing import Any, Iterator, Sequence

from anvil.backends.base import Backend
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
    SampleResult,
    SampledSequence,
    SamplingParams,
    TrainConfig,
)

WORKER_ROLE = "sample"

_TRAIN_VERB_MSG = (
    "sample worker serves sampling verbs only; "
    "training verbs belong to the train node (e.g. LocalBackend)"
)


def _vllm() -> Any:
    try:
        import vllm
    except ImportError as e:  # pragma: no cover - exercised on GPU hosts only
        raise ImportError(
            "VLLMSampleBackend requires vllm on the sample host: "
            "pip install vllm (validated with vllm 0.25.1)"
        ) from e
    return vllm


def _chosen_logprob(pos: dict[Any, Any], token_id: int) -> float:
    """Logprob of the sampled token at one position.

    vLLM returns the top-k logprob dict per position (plus the sampled token
    when it falls outside top-k). The sampled token is always present when
    logprobs are requested; fall back to the single entry defensively.
    """
    lp = pos.get(token_id)
    if lp is None:
        lp = next(iter(pos.values()))
    return float(lp.logprob if hasattr(lp, "logprob") else lp)


def _prompt_logprob_series(
    plps: list[Any] | None, token_ids: Any
) -> list[float | None]:
    """Align vLLM prompt_logprobs to the prompt token ids.

    vLLM may append the sampled continuation tokens' logprobs after the
    prompt positions; trim to prompt length (pad with None if short).
    """
    ids = list(token_ids)
    rows = (list(plps) + [None] * len(ids))[: len(ids)] if plps else [None] * len(ids)
    return [
        None if pos is None else _chosen_logprob(dict(pos), tid)
        for pos, tid in zip(rows, ids, strict=True)
    ]


class VLLMSampleBackend(Backend):
    """Sampling-only backend over a vLLM offline engine (hammer's role)."""

    name = "vllm-sample"

    def __init__(
        self,
        *,
        model: str,
        root: str | None = None,
        max_loras: int = 4,
        **engine_args: Any,
    ) -> None:
        vllm = _vllm()
        self._vllm = vllm
        self.model = model
        self.root = root
        engine_args.setdefault("enable_lora", True)
        engine_args.setdefault("max_loras", max_loras)
        self._llm = vllm.LLM(model=model, **engine_args)
        # adapter_id -> LoRARequest; int id bumps on EVERY load so a re-pushed
        # snapshot can never be served stale from vLLM's LoRA cache.
        self._loras: dict[str, Any] = {}
        self._lora_ids: Iterator[int] = itertools.count(1)

    # -- sampling verbs -------------------------------------------------

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
        del base_model  # one engine per worker; the CLI pins the base model
        sp = self._vllm_params(sampling_params, num_samples)
        # RL needs true-policy logprobs of sampled tokens (IS old-policy side)
        sp.logprobs = 1
        sp.prompt_logprobs = 1 if include_prompt_logprobs else None

        req = self._vllm.TokensPrompt(prompt_token_ids=prompt.token_ids())
        outs = self._llm.generate(
            [req], sp, lora_request=self._lora_request(adapter_id)
        )
        sequences: list[SampledSequence] = []
        prompt_lps: tuple[float | None, ...] | None = None
        for out in outs[0].outputs:
            token_ids = tuple(int(t) for t in out.token_ids)
            logprobs = tuple(
                _chosen_logprob(dict(pos), tid)
                for pos, tid in zip(out.logprobs or [], token_ids, strict=True)
            )
            sequences.append(
                SampledSequence(
                    tokens=token_ids,
                    logprobs=logprobs,
                    stop_reason="length" if out.finish_reason == "length" else "stop",
                )
            )
        if include_prompt_logprobs and outs[0].prompt_logprobs is not None:
            prompt_lps = tuple(
                _prompt_logprob_series(
                    outs[0].prompt_logprobs, req["prompt_token_ids"]
                )
            )
        return SampleResult(sequences=tuple(sequences), prompt_logprobs=prompt_lps)

    def compute_logprobs(
        self,
        *,
        base_model: str,
        adapter_id: AdapterId | None,
        prompt: ModelInput,
    ) -> list[float | None]:
        del base_model
        import copy

        sp = copy.copy(self._vllm_params(SamplingParams(max_tokens=1), 1))
        sp.prompt_logprobs = 1
        sp.max_tokens = 1
        sp.logprobs = 0
        req = self._vllm.TokensPrompt(prompt_token_ids=prompt.token_ids())
        outs = self._llm.generate(
            [req], sp, lora_request=self._lora_request(adapter_id)
        )
        return _prompt_logprob_series(
            outs[0].prompt_logprobs, req["prompt_token_ids"]
        )

    # -- adapter hot-swap -------------------------------------------------

    def load_snapshot(self, adapter_id: AdapterId, path: str) -> None:
        """Register a PEFT snapshot dir as the live LoRA for adapter_id."""
        if not os.path.isdir(path):
            raise FileNotFoundError(f"adapter snapshot dir not found: {path}")
        # vllm 0.25 does NOT re-export LoRARequest at top level; deep-import
        # (verified against the forge engine).
        from vllm.lora.request import LoRARequest

        key = str(adapter_id)
        self._loras[key] = LoRARequest(
            lora_name=key,
            lora_int_id=next(self._lora_ids),
            lora_path=path,
        )

    def _lora_request(self, adapter_id: AdapterId | None) -> Any | None:
        if adapter_id is None:
            return None
        key = str(adapter_id)
        try:
            return self._loras[key]
        except KeyError:
            raise KeyError(
                f"no adapter snapshot loaded for {key!r}; push one via "
                f"POST /v1/adapters/{key}/load_snapshot first"
            ) from None

    def _vllm_params(self, params: SamplingParams, num_samples: int) -> Any:
        return self._vllm.SamplingParams(
            n=num_samples,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            # vLLM 0.25 disables top-k with 0 (older releases used -1);
            # verified by the forge smoke test.
            top_k=0 if params.top_k is None else params.top_k,
            stop=list(params.stop),
            seed=params.seed,
        )

    # -- training verbs: not this node's job ------------------------------

    def create_lora_session(self, config: TrainConfig) -> AdapterId:
        raise NotImplementedError(_TRAIN_VERB_MSG)

    def forward_backward(
        self, adapter_id: AdapterId, data: Sequence[Datum], loss_fn: LossFn | str
    ) -> ForwardBackwardOutput:
        raise NotImplementedError(_TRAIN_VERB_MSG)

    def optim_step(self, adapter_id: AdapterId, adam: AdamParams) -> OptimStepOutput:
        raise NotImplementedError(_TRAIN_VERB_MSG)

    def save_state(self, adapter_id: AdapterId, name: str) -> CheckpointRef:
        raise NotImplementedError(_TRAIN_VERB_MSG)

    def load_state(self, adapter_id: AdapterId, ref: CheckpointRef | str) -> None:
        raise NotImplementedError(_TRAIN_VERB_MSG)

    def snapshot_for_sample(self, adapter_id: AdapterId, name: str) -> CheckpointRef:
        raise NotImplementedError(_TRAIN_VERB_MSG)

    def export_adapter(
        self, adapter_id: AdapterId, format: ExportFormat, path: str
    ) -> ExportResult:
        raise NotImplementedError(_TRAIN_VERB_MSG)
