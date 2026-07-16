"""Export format helpers (stubs until Phase 1 PEFT path is real)."""

from __future__ import annotations

from anvil.protocol.types import ExportFormat

SUPPORTED_EXPORTS: tuple[ExportFormat, ...] = (
    ExportFormat.PEFT,
    ExportFormat.MERGED_HF,
    ExportFormat.GGUF,
    ExportFormat.ONNX,
    ExportFormat.TRT,
)


def describe_export(fmt: ExportFormat) -> str:
    return {
        ExportFormat.PEFT: "HuggingFace PEFT adapter directory (adapter_config + weights).",
        ExportFormat.MERGED_HF: "Merged full weights in HF format.",
        ExportFormat.GGUF: "GGUF for llama.cpp / edge CPU paths.",
        ExportFormat.ONNX: "ONNX graph for portable inference.",
        ExportFormat.TRT: "TensorRT engine (Jetson / NVIDIA edge).",
    }[fmt]
