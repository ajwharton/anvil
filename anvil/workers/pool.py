"""Multi-worker sample pool (Expert-v2).

When generation is the wall, run **N sample workers** (vLLM ``anvil serve
--backend vllm-sample``) and one train process. This pool:

- ``load_snapshot`` → **fan-out** to every worker (same adapter id / path)
- ``sample`` / ``compute_logprobs`` → **round-robin** across workers

Train stays single-process LocalBackend (or ``anvil serve --backend local``).
That matches Spark dual-node and multi-vLLM lab layouts without inventing
full data-parallel train shards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from anvil.backends.base import Backend, SnapshotLoader
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

_TRAIN_MSG = (
    "SampleWorkerPool is sample-only; training verbs run on the train node "
    "(LocalBackend / anvil serve --backend local)"
)


@dataclass
class SampleWorkerPool:
    """Round-robin sample fan-out + snapshot broadcast.

    Implements the Backend methods GRPO needs on the sample side plus
    :class:`SnapshotLoader`. Not a full train backend.
    """

    backends: list[Backend]
    label: str = "sample-pool"
    _rr: int = field(default=0, init=False, repr=False)
    sample_calls: int = field(default=0, init=False)
    sync_calls: int = field(default=0, init=False)
    last_worker_index: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.backends:
            raise ValueError("SampleWorkerPool requires at least one backend")
        for i, b in enumerate(self.backends):
            if not isinstance(b, SnapshotLoader):
                raise TypeError(
                    f"pool worker[{i}] {type(b).__name__!r} is not a SnapshotLoader "
                    f"(need load_snapshot for adapter sync)"
                )

    @property
    def n_workers(self) -> int:
        return len(self.backends)

    def _next(self) -> tuple[int, Backend]:
        ix = self._rr % len(self.backends)
        self._rr += 1
        self.last_worker_index = ix
        return ix, self.backends[ix]

    # --- SnapshotLoader ------------------------------------------------------

    def load_snapshot(self, adapter_id: AdapterId, path: str) -> None:
        """Push the same PEFT snapshot to every sample worker."""
        errors: list[str] = []
        for i, b in enumerate(self.backends):
            try:
                b.load_snapshot(adapter_id, path)  # type: ignore[attr-defined]
            except Exception as e:  # noqa: BLE001 — collect all failures
                errors.append(f"worker[{i}]: {type(e).__name__}: {e}")
        self.sync_calls += 1
        if errors:
            raise RuntimeError(
                "load_snapshot failed on one or more sample workers: "
                + "; ".join(errors)
            )

    # --- sampling ------------------------------------------------------------

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
        self.sample_calls += 1
        _, b = self._next()
        return b.sample(
            base_model=base_model,
            adapter_id=adapter_id,
            prompt=prompt,
            sampling_params=sampling_params,
            num_samples=num_samples,
            include_prompt_logprobs=include_prompt_logprobs,
        )

    def compute_logprobs(
        self,
        *,
        base_model: str,
        adapter_id: AdapterId | None,
        prompt: ModelInput,
    ) -> list[float | None]:
        self.sample_calls += 1
        _, b = self._next()
        return b.compute_logprobs(
            base_model=base_model, adapter_id=adapter_id, prompt=prompt
        )

    # --- train verbs: hard fail ----------------------------------------------

    def create_lora_session(self, config: TrainConfig) -> AdapterId:
        raise NotImplementedError(_TRAIN_MSG)

    def forward_backward(
        self,
        adapter_id: AdapterId,
        data: Sequence[Datum],
        loss_fn: LossFn | str,
    ) -> ForwardBackwardOutput:
        raise NotImplementedError(_TRAIN_MSG)

    def optim_step(self, adapter_id: AdapterId, adam: AdamParams) -> OptimStepOutput:
        raise NotImplementedError(_TRAIN_MSG)

    def save_state(self, adapter_id: AdapterId, name: str) -> CheckpointRef:
        raise NotImplementedError(_TRAIN_MSG)

    def load_state(self, adapter_id: AdapterId, checkpoint: CheckpointRef) -> None:
        raise NotImplementedError(_TRAIN_MSG)

    def save_weights_for_sampler(
        self, adapter_id: AdapterId, name: str
    ) -> CheckpointRef:
        raise NotImplementedError(_TRAIN_MSG)

    def export_adapter(
        self,
        adapter_id: AdapterId,
        format: ExportFormat,
        path: str,
    ) -> ExportResult:
        raise NotImplementedError(_TRAIN_MSG)

    def stats(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_workers": self.n_workers,
            "sample_calls": self.sample_calls,
            "sync_calls": self.sync_calls,
            "last_worker_index": self.last_worker_index,
        }


def build_sample_pool(
    endpoints: Sequence[str],
    *,
    label: str | None = None,
    token: str | None = None,
) -> SampleWorkerPool:
    """Connect to remote sample workers (``http://host:port``) as a pool."""
    from anvil.client.service import ServiceClient

    backends: list[Backend] = []
    clients: list[ServiceClient] = []
    for ep in endpoints:
        ep_s = str(ep).strip()
        if not ep_s:
            continue
        svc = ServiceClient(endpoint=ep_s, queue=False, token=token)
        clients.append(svc)
        backends.append(svc.backend)
    if not backends:
        raise ValueError("build_sample_pool requires at least one non-empty endpoint")
    pool = SampleWorkerPool(
        backends=backends,
        label=label or f"pool[{len(backends)}]",
    )
    # Stash clients so callers can close them (optional attribute)
    pool._service_clients = clients  # type: ignore[attr-defined]
    return pool


def close_sample_pool(pool: SampleWorkerPool) -> None:
    """Close any ServiceClients owned by :func:`build_sample_pool`."""
    for svc in getattr(pool, "_service_clients", []) or []:
        try:
            svc.close()
        except Exception:  # noqa: BLE001
            pass


@dataclass(frozen=True, slots=True)
class MultiWorkerLayout:
    """Declarative train + sample layout for ops docs / CLI."""

    train_endpoint: str
    sample_endpoints: tuple[str, ...]
    notes: str = ""

    def to_public(self) -> dict[str, Any]:
        return {
            "train_endpoint": self.train_endpoint,
            "sample_endpoints": list(self.sample_endpoints),
            "n_sample_workers": len(self.sample_endpoints),
            "notes": self.notes,
        }

    def launch_hints(self) -> list[str]:
        """Shell-ish hints (not executed) for dual-host labs."""
        lines = [
            "# train node — four verbs (LocalBackend)",
            "anvil serve --backend local --host 0.0.0.0 --port "
            f"{_port_of(self.train_endpoint) or 8740}",
        ]
        for i, ep in enumerate(self.sample_endpoints):
            port = _port_of(ep) or (8741 + i)
            lines.append(
                f"# sample worker {i} — vLLM sample only\n"
                f"anvil serve --backend vllm-sample --model $MODEL "
                f"--host 0.0.0.0 --port {port}"
            )
        lines.append(
            "# GRPO on the orchestrator / train box:\n"
            f"run_grpo(endpoint={self.train_endpoint!r}, "
            f"sample_endpoints={list(self.sample_endpoints)!r}, sync_every=1, ...)"
        )
        return lines


def _port_of(endpoint: str) -> int | None:
    # http://host:8741 or host:8741
    s = endpoint.rsplit(":", 1)
    if len(s) != 2:
        return None
    try:
        return int(s[1].split("/")[0])
    except ValueError:
        return None


__all__ = [
    "MultiWorkerLayout",
    "SampleWorkerPool",
    "build_sample_pool",
    "close_sample_pool",
]
