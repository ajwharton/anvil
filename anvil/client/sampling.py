"""SamplingClient — sample + compute_logprobs against base (+ optional adapter)."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from anvil.client.futures import AnvilFuture, VerbQueue, completed
from anvil.protocol.types import (
    AdapterId,
    CheckpointRef,
    ModelInput,
    SampleResult,
    SamplingParams,
)

if TYPE_CHECKING:
    from anvil.backends.base import Backend


class SamplingClient:
    """Generation handle. Tinker-shaped: sample / compute_logprobs."""

    def __init__(
        self,
        *,
        backend: Backend,
        base_model: str,
        adapter_id: AdapterId | None = None,
        checkpoint: CheckpointRef | None = None,
        queue: VerbQueue | None = None,
    ) -> None:
        self._backend = backend
        self.base_model = base_model
        self.adapter_id = adapter_id
        self.checkpoint = checkpoint
        self._queue = queue

    def sample(
        self,
        prompt: ModelInput,
        sampling_params: SamplingParams | None = None,
        *,
        num_samples: int = 1,
        include_prompt_logprobs: bool = False,
    ) -> AnvilFuture[SampleResult]:
        params = sampling_params if sampling_params is not None else SamplingParams()
        call = partial(
            self._backend.sample,
            base_model=self.base_model,
            adapter_id=self.adapter_id,
            prompt=prompt,
            sampling_params=params,
            num_samples=num_samples,
            include_prompt_logprobs=include_prompt_logprobs,
        )
        if self._queue is not None:
            return self._queue.submit(call)
        return completed(call())

    async def sample_async(
        self,
        prompt: ModelInput,
        sampling_params: SamplingParams | None = None,
        *,
        num_samples: int = 1,
        include_prompt_logprobs: bool = False,
    ) -> SampleResult:
        return self.sample(
            prompt,
            sampling_params,
            num_samples=num_samples,
            include_prompt_logprobs=include_prompt_logprobs,
        ).result()

    def compute_logprobs(self, prompt: ModelInput) -> AnvilFuture[list[float | None]]:
        call = partial(
            self._backend.compute_logprobs,
            base_model=self.base_model,
            adapter_id=self.adapter_id,
            prompt=prompt,
        )
        if self._queue is not None:
            return self._queue.submit(call)
        return completed(call())

    async def compute_logprobs_async(self, prompt: ModelInput) -> list[float | None]:
        return self.compute_logprobs(prompt).result()
