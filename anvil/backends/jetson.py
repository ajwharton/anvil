"""Jetson / edge sample stub (Phase 4.C).

Does **not** train on-device. Implements a thin **sample** path against a
remote Ollama-compatible HTTP endpoint (smolvlm-256m on Orin) so Anvil can
probe the same adapter-shaped policy language as lab without shipping a full
local GPU stack to the robot.

Train still happens on forge/lab; export PEFT/GGUF, then sample here.

Environment
-----------
- ``ANVIL_JETSON_URL`` — default ``http://127.0.0.1:11434`` (Ollama)
- ``ANVIL_JETSON_MODEL`` — default ``smolvlm-256m``
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anvil.protocol.types import (
    AdapterId,
    ExportFormat,
    ExportResult,
    ForwardBackwardOutput,
    ModelInput,
    OptimStepOutput,
    SampledSequence,
    SampleResult,
    SamplingParams,
    TrainConfig,
)


@dataclass
class JetsonSampleConfig:
    url: str = field(
        default_factory=lambda: os.environ.get("ANVIL_JETSON_URL", "http://127.0.0.1:11434")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("ANVIL_JETSON_MODEL", "smolvlm-256m")
    )
    timeout: float = 60.0
    dry_run: bool = False
    """If True, never hit the network — return deterministic stub samples (CI)."""


class JetsonSampleBackend:
    """Sample-only backend for edge / Jetson (Ollama-compatible).

    Training verbs raise ``NotImplementedError`` — use lab backends to train,
    then export and sample here.
    """

    name = "jetson"

    def __init__(self, config: JetsonSampleConfig | None = None) -> None:
        self.config = config or JetsonSampleConfig()
        self._adapters: dict[str, dict[str, Any]] = {}
        self.last_text: str = ""

    def create_lora_session(self, config: TrainConfig) -> AdapterId:
        aid = AdapterId(f"jetson-{len(self._adapters)+1:04d}")
        self._adapters[aid.value] = {
            "base_model": config.base_model,
            "step": 0,
            "config": config,
        }
        return aid

    def forward_backward(
        self, adapter_id: AdapterId, *args: Any, **kwargs: Any
    ) -> ForwardBackwardOutput:
        raise NotImplementedError(
            "JetsonSampleBackend is sample-only; train on lab and export to edge"
        )

    def optim_step(self, adapter_id: AdapterId, *args: Any, **kwargs: Any) -> OptimStepOutput:
        raise NotImplementedError(
            "JetsonSampleBackend is sample-only; train on lab and export to edge"
        )

    def save_state(self, adapter_id: AdapterId, name: str) -> Any:
        raise NotImplementedError("JetsonSampleBackend is sample-only")

    def export_adapter(
        self, adapter_id: AdapterId, format: ExportFormat, path: str
    ) -> ExportResult:
        # Re-export is a no-op package pointing at the edge model tag.
        from anvil.export.edge import package_edge_export

        sess = self._get(adapter_id)

        def _peft(dest: Path) -> None:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "edge_model.json").write_text(
                json.dumps(
                    {
                        "backend": "jetson",
                        "model": self.config.model,
                        "url": self.config.url,
                        "base_model": sess.get("base_model"),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        bundle = package_edge_export(
            fmt=format if format != ExportFormat.PEFT else ExportFormat.PEFT,
            root=Path(path),
            adapter_id=adapter_id.value,
            base_model=str(sess.get("base_model") or self.config.model),
            save_peft=_peft,
            save_merged=None,
            extra_notes=[
                f"sample endpoint {self.config.url} model={self.config.model}",
                "No weights written — edge serves Ollama/llama.cpp copy.",
            ],
        )
        return bundle.result

    def sample(
        self,
        *,
        base_model: str,
        adapter_id: AdapterId | None,
        prompt: ModelInput | str,
        sampling_params: SamplingParams | None = None,
        num_samples: int = 1,
        include_prompt_logprobs: bool = False,
        **kwargs: Any,
    ) -> SampleResult:
        if adapter_id is not None:
            self._get(adapter_id)
        _ = base_model
        _ = include_prompt_logprobs
        text_prompt = _prompt_from_input(prompt)
        images = _images_from_input(prompt)
        params = sampling_params or SamplingParams()
        sequences: list[SampledSequence] = []
        for _ in range(max(1, num_samples)):
            text = self._generate(text_prompt, images=images, params=params)
            # Edge samples are text; encode as byte-ish token stubs for protocol parity.
            toks = tuple(min(255, ord(c)) for c in text[: params.max_tokens or 64])
            sequences.append(
                SampledSequence(tokens=toks or (0,), logprobs=None, stop_reason="stop")
            )
        self.last_text = text if sequences else ""
        return SampleResult(sequences=tuple(sequences))

    def _get(self, adapter_id: AdapterId) -> dict[str, Any]:
        if adapter_id.value not in self._adapters:
            raise KeyError(f"unknown adapter {adapter_id.value}")
        return self._adapters[adapter_id.value]

    def _generate(
        self,
        prompt: str,
        *,
        images: list[str],
        params: SamplingParams | None,
    ) -> str:
        if self.config.dry_run:
            return f"[jetson-dry-run model={self.config.model}] {prompt[:80]}"

        url = self.config.url.rstrip("/") + "/api/generate"
        body: dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
        }
        if params is not None:
            body["options"] = {
                "temperature": getattr(params, "temperature", 0.2) or 0.2,
                "num_predict": getattr(params, "max_tokens", 64) or 64,
            }
        if images:
            body["images"] = images  # Ollama expects base64 strings

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"jetson sample HTTP {e.code}: {e.read()[:300]!r}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"jetson unreachable {self.config.url}: {e} "
                "(set ANVIL_JETSON_URL or JetsonSampleConfig.dry_run=True)"
            ) from e
        return str(data.get("response") or data.get("text") or "")


def _prompt_from_input(model_input: Any) -> str:
    if isinstance(model_input, str):
        return model_input
    if hasattr(model_input, "text") and model_input.text:
        return str(model_input.text)
    if hasattr(model_input, "prompt"):
        return str(model_input.prompt)
    # chunks / tokens — best-effort stringify
    return str(model_input)


def _images_from_input(model_input: Any) -> list[str]:
    """Return base64 strings if ModelInput carries raw image bytes/paths."""
    out: list[str] = []
    chunks = getattr(model_input, "chunks", None) or getattr(model_input, "content", None)
    if not chunks:
        return out
    for ch in chunks:
        raw = getattr(ch, "data", None) or getattr(ch, "bytes", None)
        if isinstance(raw, (bytes, bytearray)):
            out.append(base64.b64encode(bytes(raw)).decode("ascii"))
            continue
        path = getattr(ch, "path", None)
        if path and Path(path).is_file():
            out.append(base64.b64encode(Path(path).read_bytes()).decode("ascii"))
    return out
