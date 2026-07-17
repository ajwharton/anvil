"""RemoteBackend — the Backend protocol over HTTP.

Client half of `anvil serve`. Core-package dependency-free: the default
transport is stdlib urllib; tests inject a transport callable instead.

    svc = ServiceClient("http://forge.local:8741")   # remote worker
    svc = ServiceClient("local://")                  # in-process torch+PEFT

Both give the same four verbs; only latency and GPU locality differ.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping, Protocol, Sequence

from anvil.protocol.serde import (
    checkpoint_ref_from_wire,
    export_result_from_wire,
    forward_backward_output_from_wire,
    logprobs_from_wire,
    optim_step_output_from_wire,
    sample_result_from_wire,
    to_wire,
)
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
    SamplingParams,
    TrainConfig,
)

API = "/v1"


class Transport(Protocol):
    """(method, path, json_body) -> decoded json. Raises on transport error."""

    def __call__(self, method: str, path: str, body: Mapping[str, Any] | None) -> Any: ...


class UrllibTransport:
    def __init__(self, base_url: str, *, token: str | None = None, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def __call__(self, method: str, path: str, body: Mapping[str, Any] | None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode("utf-8"))
            except Exception:
                detail = {"error": "http_error", "detail": str(e)}
            raise RemoteBackendError(e.code, detail) from e
        except urllib.error.URLError as e:
            raise RemoteBackendError(None, {"error": "unreachable", "detail": str(e)}) from e


class RemoteBackendError(RuntimeError):
    def __init__(self, status: int | None, payload: Mapping[str, Any]):
        self.status = status
        self.payload = dict(payload)
        super().__init__(
            f"remote backend error (status={status}): "
            f"{payload.get('error', '?')}: {payload.get('detail', '')}"
        )


class RemoteBackend:
    """Backend implementation that forwards verbs to `anvil serve`."""

    name = "remote"

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._t: Transport = transport or UrllibTransport(base_url, token=token)

    # -- protocol ----------------------------------------------------------

    def create_lora_session(self, config: TrainConfig) -> AdapterId:
        out = self._t("POST", f"{API}/sessions", {"config": to_wire(config)})
        return AdapterId(str(out["adapter_id"]))

    def forward_backward(
        self,
        adapter_id: AdapterId,
        data: Sequence[Datum],
        loss_fn: LossFn | str,
    ) -> ForwardBackwardOutput:
        out = self._t(
            "POST",
            f"{API}/sessions/{adapter_id}/forward_backward",
            {
                "data": [to_wire(d) for d in data],
                "loss_fn": loss_fn.value if isinstance(loss_fn, LossFn) else str(loss_fn),
            },
        )
        return forward_backward_output_from_wire(out)

    def optim_step(self, adapter_id: AdapterId, adam: AdamParams) -> OptimStepOutput:
        out = self._t(
            "POST", f"{API}/sessions/{adapter_id}/optim_step", {"adam": to_wire(adam)}
        )
        return optim_step_output_from_wire(out)

    def save_state(self, adapter_id: AdapterId, name: str) -> CheckpointRef:
        out = self._t("POST", f"{API}/sessions/{adapter_id}/save_state", {"name": name})
        return checkpoint_ref_from_wire(out)

    def load_state(self, adapter_id: AdapterId, ref: CheckpointRef | str) -> None:
        wire = ref if isinstance(ref, str) else to_wire(ref)
        self._t("POST", f"{API}/sessions/{adapter_id}/load_state", {"ref": wire})

    def snapshot_for_sample(self, adapter_id: AdapterId, name: str) -> CheckpointRef:
        out = self._t(
            "POST", f"{API}/sessions/{adapter_id}/snapshot_for_sample", {"name": name}
        )
        return checkpoint_ref_from_wire(out)

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
        out = self._t(
            "POST",
            f"{API}/sample",
            {
                "base_model": base_model,
                "adapter_id": str(adapter_id) if adapter_id is not None else None,
                "prompt": to_wire(prompt),
                "sampling_params": to_wire(sampling_params),
                "num_samples": num_samples,
                "include_prompt_logprobs": include_prompt_logprobs,
            },
        )
        return sample_result_from_wire(out)

    def compute_logprobs(
        self,
        *,
        base_model: str,
        adapter_id: AdapterId | None,
        prompt: ModelInput,
    ) -> list[float | None]:
        out = self._t(
            "POST",
            f"{API}/compute_logprobs",
            {
                "base_model": base_model,
                "adapter_id": str(adapter_id) if adapter_id is not None else None,
                "prompt": to_wire(prompt),
            },
        )
        return logprobs_from_wire(out["logprobs"])

    def export_adapter(
        self,
        adapter_id: AdapterId,
        format: ExportFormat,
        path: str,
    ) -> ExportResult:
        out = self._t(
            "POST",
            f"{API}/sessions/{adapter_id}/export",
            {"format": format.value, "path": path},
        )
        return export_result_from_wire(out)
