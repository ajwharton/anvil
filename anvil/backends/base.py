"""Backend protocol — pluggable train/sample implementations."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

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


@runtime_checkable
class Backend(Protocol):
    """What a ServiceClient talks to. HTTP later; in-process for fake/local."""

    name: str

    def create_lora_session(self, config: TrainConfig) -> AdapterId: ...

    def forward_backward(
        self,
        adapter_id: AdapterId,
        data: Sequence[Datum],
        loss_fn: LossFn | str,
    ) -> ForwardBackwardOutput: ...

    def optim_step(
        self,
        adapter_id: AdapterId,
        adam: AdamParams,
    ) -> OptimStepOutput: ...

    def save_state(self, adapter_id: AdapterId, name: str) -> CheckpointRef: ...

    def load_state(self, adapter_id: AdapterId, ref: CheckpointRef | str) -> None: ...

    def snapshot_for_sample(self, adapter_id: AdapterId, name: str) -> CheckpointRef: ...

    def sample(
        self,
        *,
        base_model: str,
        adapter_id: AdapterId | None,
        prompt: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int = 1,
        include_prompt_logprobs: bool = False,
    ) -> SampleResult: ...

    def compute_logprobs(
        self,
        *,
        base_model: str,
        adapter_id: AdapterId | None,
        prompt: ModelInput,
    ) -> list[float | None]: ...

    def export_adapter(
        self,
        adapter_id: AdapterId,
        format: ExportFormat,
        path: str,
    ) -> ExportResult: ...
