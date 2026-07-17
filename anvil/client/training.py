"""TrainingClient — forward_backward, optim_step, save_state, export."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from anvil.client.futures import AnvilFuture, VerbQueue, completed
from anvil.protocol.types import (
    AdamParams,
    AdapterId,
    CheckpointRef,
    Datum,
    ExportFormat,
    ExportResult,
    ForwardBackwardOutput,
    LossFn,
    LoraConfig,
    OptimStepOutput,
    TrainConfig,
)

if TYPE_CHECKING:
    from anvil.backends.base import Backend
    from anvil.client.sampling import SamplingClient
    from anvil.control.registry import AdapterRegistry


class TrainingClient:
    """LoRA training handle bound to one adapter + base model."""

    def __init__(
        self,
        *,
        backend: Backend,
        adapter_id: AdapterId,
        config: TrainConfig,
        registry: AdapterRegistry,
        queue: VerbQueue | None = None,
    ) -> None:
        self._backend = backend
        self.adapter_id = adapter_id
        self.config = config
        self._registry = registry
        self._queue = queue

    @property
    def base_model(self) -> str:
        return self.config.base_model

    @property
    def lora(self) -> LoraConfig:
        return self.config.lora

    # --- core verbs --------------------------------------------------------

    def forward_backward(
        self,
        data: Sequence[Datum],
        loss_fn: LossFn | str = LossFn.CROSS_ENTROPY,
    ) -> AnvilFuture[ForwardBackwardOutput]:
        """Compute loss + accumulate LoRA grads. Returns a future (pipelinable)."""
        if self._queue is not None:
            return self._queue.submit(
                self._backend.forward_backward, self.adapter_id, data, loss_fn
            )
        out = self._backend.forward_backward(self.adapter_id, data, loss_fn)
        return completed(out)

    async def forward_backward_async(
        self,
        data: Sequence[Datum],
        loss_fn: LossFn | str = LossFn.CROSS_ENTROPY,
    ) -> AnvilFuture[ForwardBackwardOutput]:
        return self.forward_backward(data, loss_fn)

    def optim_step(self, adam_params: AdamParams | None = None) -> AnvilFuture[OptimStepOutput]:
        adam = adam_params if adam_params is not None else AdamParams()
        if self._queue is not None:
            return self._queue.submit(self._backend.optim_step, self.adapter_id, adam)
        out = self._backend.optim_step(self.adapter_id, adam)
        return completed(out)

    async def optim_step_async(
        self, adam_params: AdamParams | None = None
    ) -> AnvilFuture[OptimStepOutput]:
        return self.optim_step(adam_params)

    def save_state(self, name: str) -> CheckpointRef:
        """Save adapter weights + optimizer bookkeeping."""
        if self._queue is not None:
            return self._queue.run(self._backend.save_state, self.adapter_id, name)
        return self._backend.save_state(self.adapter_id, name)

    def load_state(self, ref: CheckpointRef | str) -> None:
        if self._queue is not None:
            return self._queue.run(self._backend.load_state, self.adapter_id, ref)
        self._backend.load_state(self.adapter_id, ref)

    def save_weights_and_get_sampling_client(self, name: str = "latest") -> SamplingClient:
        """Snapshot adapter for on-policy sample (RL) or eval."""
        from anvil.client.sampling import SamplingClient

        ref = (
            self._queue.run(self._backend.snapshot_for_sample, self.adapter_id, name)
            if self._queue is not None
            else self._backend.snapshot_for_sample(self.adapter_id, name)
        )
        return SamplingClient(
            backend=self._backend,
            base_model=self.base_model,
            adapter_id=self.adapter_id,
            checkpoint=ref,
            queue=self._queue,
        )

    def export_adapter(
        self,
        path: str,
        format: ExportFormat | str = ExportFormat.PEFT,
    ) -> ExportResult:
        fmt = ExportFormat(format) if isinstance(format, str) else format
        if self._queue is not None:
            return self._queue.run(self._backend.export_adapter, self.adapter_id, fmt, path)
        return self._backend.export_adapter(self.adapter_id, fmt, path)

    def get_tokenizer(self) -> None:
        """Phase 1: return HF tokenizer for base_model. Stub returns None."""
        return None
