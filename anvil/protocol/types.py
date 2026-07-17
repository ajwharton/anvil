"""Wire-facing types for the Anvil client contract.

ServiceClient → TrainingClient / SamplingClient, Datum, ModelInput, and the
four verbs form the stable product surface. No third-party deps — dataclasses
only for Phase 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdapterId:
    """Stable id for a LoRA adapter bound to a base model."""

    value: str

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class SessionId:
    value: str

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    """Reference to a saved train or sampler checkpoint."""

    name: str
    path: str
    kind: str = "weights"  # "weights" | "train_state" | "sampler"


# ---------------------------------------------------------------------------
# Model input (token + multimodal chunks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EncodedTextChunk:
    """Pre-tokenized text."""

    tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, tuple):
            object.__setattr__(self, "tokens", tuple(self.tokens))


@dataclass(frozen=True, slots=True)
class ImageRefChunk:
    """Vision-first: content-addressed media ref (preferred over inline bytes)."""

    ref: str
    detail: str = "auto"  # "auto" | "low" | "high"


@dataclass(frozen=True, slots=True)
class ImageChunk:
    """Inline image bytes — ok for demos; prefer ImageRefChunk in real loops."""

    data: bytes
    format: str = "png"  # "png" | "jpeg" | "webp" | …


Chunk = EncodedTextChunk | ImageRefChunk | ImageChunk


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Token / multimodal sequence for train or sample.

    Build from ints or from an ordered chunk list.
    """

    chunks: tuple[Chunk, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.chunks, tuple):
            object.__setattr__(self, "chunks", tuple(self.chunks))

    @classmethod
    def from_ints(cls, tokens: Sequence[int]) -> ModelInput:
        return cls(chunks=(EncodedTextChunk(tokens=tuple(int(t) for t in tokens)),))

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk]) -> ModelInput:
        return cls(chunks=tuple(chunks))

    def token_ids(self) -> list[int]:
        """Flatten text token ids; image chunks contribute no token ids here.

        Real renderers expand image refs into model-specific placeholder tokens
        before this is used for CE. Fake backend uses text tokens only.
        """
        out: list[int] = []
        for c in self.chunks:
            if isinstance(c, EncodedTextChunk):
                out.extend(c.tokens)
        return out

    def __len__(self) -> int:
        return len(self.token_ids())


# ---------------------------------------------------------------------------
# Training datum + losses
# ---------------------------------------------------------------------------


class LossFn(str, Enum):
    """Named losses the worker understands. No arbitrary remote code in v0."""

    CROSS_ENTROPY = "cross_entropy"
    IMPORTANCE_SAMPLING = "importance_sampling"
    PPO = "ppo"
    CISPO = "cispo"
    DRO = "dro"
    DPO = "dpo"


# loss_fn_inputs keys used by stock losses (tensors as plain lists for now):
#   cross_entropy:        target_tokens, weights
#   importance_sampling:  target_tokens, weights, logprobs, advantages
#   ppo / cispo / dro:    same as IS (+ optional clip params later)
#   dpo:                  preferred / rejected handled via paired data later


@dataclass(slots=True)
class Datum:
    """Single training example (model_input + loss_fn_inputs).

    model_input: full sequence fed to the model
    loss_fn_inputs: tensors the named loss needs (targets, weights, logprobs, …)
    """

    model_input: ModelInput
    loss_fn_inputs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Optimizer + sampling params
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdamParams:
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass(frozen=True, slots=True)
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stop, tuple):
            object.__setattr__(self, "stop", tuple(self.stop))


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForwardBackwardOutput:
    """Result of one forward_backward call."""

    loss: float
    metrics: Mapping[str, float] = field(default_factory=dict)
    # Optional per-token logprobs under current policy (RL)
    logprobs: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class OptimStepOutput:
    step: int
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SampledSequence:
    tokens: tuple[int, ...]
    logprobs: tuple[float, ...] | None = None
    stop_reason: str = "length"  # "length" | "stop" | "eos"

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, tuple):
            object.__setattr__(self, "tokens", tuple(self.tokens))
        if self.logprobs is not None and not isinstance(self.logprobs, tuple):
            object.__setattr__(self, "logprobs", tuple(self.logprobs))


@dataclass(frozen=True, slots=True)
class SampleResult:
    sequences: tuple[SampledSequence, ...]
    prompt_logprobs: tuple[float | None, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sequences, tuple):
            object.__setattr__(self, "sequences", tuple(self.sequences))


# ---------------------------------------------------------------------------
# LoRA / session config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoraTargets:
    """Which submodules get LoRA. Vision knobs are first-class."""

    language: bool = True
    vision_encoder: bool = False
    mm_projector: bool = True


@dataclass(frozen=True, slots=True)
class LoraConfig:
    rank: int = 32
    alpha: int | None = None  # default 2 * rank when None
    dropout: float = 0.0
    targets: LoraTargets = field(default_factory=LoraTargets)

    def effective_alpha(self) -> int:
        return self.alpha if self.alpha is not None else 2 * self.rank


@dataclass(frozen=True, slots=True)
class TrainConfig:
    base_model: str
    lora: LoraConfig = field(default_factory=LoraConfig)
    modalities: tuple[str, ...] = ("text",)

    def __post_init__(self) -> None:
        if not isinstance(self.modalities, tuple):
            object.__setattr__(self, "modalities", tuple(self.modalities))


class ExportFormat(str, Enum):
    PEFT = "peft"  # HF PEFT adapter dir
    MERGED_HF = "merged_hf"
    GGUF = "gguf"
    ONNX = "onnx"
    TRT = "trt"


@dataclass(frozen=True, slots=True)
class ExportResult:
    format: ExportFormat
    path: str
    adapter_id: str
