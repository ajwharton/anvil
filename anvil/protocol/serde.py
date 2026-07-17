"""Wire codec for the four verbs — dependency-free JSON shapes.

Server (anvil.serve) and remote client (anvil.client.remote) share this so the
HTTP boundary cannot drift from the in-process contract. Dataclasses encode to
plain dicts; unions/enums/bytes get explicit tags. Anything not listed here is
a bug — the wire surface is deliberately closed.

All shapes are JSON-serializable: tuples → lists, enums → values, bytes →
base64 strings, tagged unions carry ``kind``.
"""

from __future__ import annotations

import base64
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Mapping

from anvil.protocol.types import (
    AdamParams,
    CheckpointRef,
    Datum,
    EncodedTextChunk,
    ExportFormat,
    ExportResult,
    ForwardBackwardOutput,
    ImageChunk,
    ImageRefChunk,
    LoraConfig,
    LoraTargets,
    ModelInput,
    OptimStepOutput,
    SampledSequence,
    SampleResult,
    SamplingParams,
    TrainConfig,
)

# ---------------------------------------------------------------------------
# to_wire
# ---------------------------------------------------------------------------


def to_wire(obj: Any) -> Any:
    """Encode protocol dataclasses/enums/containers to JSON-safe shapes."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, bytes):
        return {"$bytes": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, EncodedTextChunk):
        return {"kind": "text", "tokens": list(obj.tokens)}
    if isinstance(obj, ImageRefChunk):
        return {"kind": "image_ref", "ref": obj.ref, "detail": obj.detail}
    if isinstance(obj, ImageChunk):
        return {
            "kind": "image",
            "data": base64.b64encode(obj.data).decode("ascii"),
            "format": obj.format,
        }
    if isinstance(obj, Mapping):
        return {str(k): to_wire(v) for k, v in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_wire(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [to_wire(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# from_wire — explicit reconstructors (closed surface)
# ---------------------------------------------------------------------------


def _chunk_from_wire(d: Mapping[str, Any]) -> EncodedTextChunk | ImageRefChunk | ImageChunk:
    kind = d.get("kind")
    if kind == "text":
        return EncodedTextChunk(tokens=tuple(int(t) for t in d["tokens"]))
    if kind == "image_ref":
        return ImageRefChunk(ref=str(d["ref"]), detail=str(d.get("detail", "auto")))
    if kind == "image":
        return ImageChunk(
            data=base64.b64decode(d["data"]),
            format=str(d.get("format", "png")),
        )
    raise ValueError(f"unknown chunk kind {kind!r}")


def model_input_from_wire(d: Mapping[str, Any]) -> ModelInput:
    return ModelInput(chunks=tuple(_chunk_from_wire(c) for c in d.get("chunks", ())))


def datum_from_wire(d: Mapping[str, Any]) -> Datum:
    return Datum(
        model_input=model_input_from_wire(d["model_input"]),
        loss_fn_inputs=dict(d.get("loss_fn_inputs") or {}),
    )


def train_config_from_wire(d: Mapping[str, Any]) -> TrainConfig:
    lora = d.get("lora") or {}
    targets = lora.get("targets") or {}
    return TrainConfig(
        base_model=str(d["base_model"]),
        lora=LoraConfig(
            rank=int(lora.get("rank", 32)),
            alpha=lora.get("alpha"),
            dropout=float(lora.get("dropout", 0.0)),
            targets=LoraTargets(
                language=bool(targets.get("language", True)),
                vision_encoder=bool(targets.get("vision_encoder", False)),
                mm_projector=bool(targets.get("mm_projector", True)),
            ),
        ),
        modalities=tuple(d.get("modalities", ("text",))),
    )


def adam_from_wire(d: Mapping[str, Any]) -> AdamParams:
    return AdamParams(
        learning_rate=float(d.get("learning_rate", 1e-4)),
        beta1=float(d.get("beta1", 0.9)),
        beta2=float(d.get("beta2", 0.95)),
        eps=float(d.get("eps", 1e-8)),
        weight_decay=float(d.get("weight_decay", 0.0)),
    )


def sampling_params_from_wire(d: Mapping[str, Any]) -> SamplingParams:
    return SamplingParams(
        max_tokens=int(d.get("max_tokens", 64)),
        temperature=float(d.get("temperature", 1.0)),
        top_p=float(d.get("top_p", 1.0)),
        top_k=d.get("top_k"),
        stop=tuple(d.get("stop", ())),
        seed=d.get("seed"),
    )


def checkpoint_ref_from_wire(d: Mapping[str, Any] | str) -> CheckpointRef:
    if isinstance(d, str):
        return CheckpointRef(name=d, path=d)
    return CheckpointRef(
        name=str(d["name"]), path=str(d["path"]), kind=str(d.get("kind", "weights"))
    )


def forward_backward_output_from_wire(d: Mapping[str, Any]) -> ForwardBackwardOutput:
    lp = d.get("logprobs")
    return ForwardBackwardOutput(
        loss=float(d["loss"]),
        metrics={str(k): float(v) for k, v in (d.get("metrics") or {}).items()},
        logprobs=tuple(float(x) for x in lp) if lp is not None else None,
    )


def optim_step_output_from_wire(d: Mapping[str, Any]) -> OptimStepOutput:
    return OptimStepOutput(
        step=int(d["step"]),
        metrics={str(k): float(v) for k, v in (d.get("metrics") or {}).items()},
    )


def sample_result_from_wire(d: Mapping[str, Any]) -> SampleResult:
    seqs = []
    for s in d.get("sequences", ()):
        lp = s.get("logprobs")
        seqs.append(
            SampledSequence(
                tokens=tuple(int(t) for t in s["tokens"]),
                logprobs=tuple(float(x) for x in lp) if lp is not None else None,
                stop_reason=str(s.get("stop_reason", "length")),
            )
        )
    plp = d.get("prompt_logprobs")
    return SampleResult(
        sequences=tuple(seqs),
        prompt_logprobs=(
            tuple(None if x is None else float(x) for x in plp) if plp is not None else None
        ),
    )


def export_result_from_wire(d: Mapping[str, Any]) -> ExportResult:
    return ExportResult(
        format=ExportFormat(str(d["format"])),
        path=str(d["path"]),
        adapter_id=str(d["adapter_id"]),
    )


def logprobs_from_wire(d: Any) -> list[float | None]:
    return [None if x is None else float(x) for x in d]
