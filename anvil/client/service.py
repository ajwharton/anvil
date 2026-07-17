"""ServiceClient — entry point (Tinker-shaped)."""

from __future__ import annotations

from typing import Sequence

from anvil.backends.base import Backend
from anvil.backends.fake import FakeBackend
from anvil.client.futures import VerbQueue
from anvil.client.sampling import SamplingClient
from anvil.client.training import TrainingClient
from anvil.control.session import Session
from anvil.protocol.types import LoraConfig, LoraTargets, TrainConfig


def _parse_endpoint(endpoint: str) -> tuple[str, str]:
    """Return (scheme, rest)."""
    if "://" in endpoint:
        scheme, rest = endpoint.split("://", 1)
        return scheme.lower(), rest
    return "local", endpoint


def resolve_backend(endpoint: str, backend: Backend | None = None) -> Backend:
    if backend is not None:
        return backend
    scheme, rest = _parse_endpoint(endpoint)
    if scheme == "fake":
        # fake:// → default root; fake://path/to/dir → root there
        return FakeBackend(root=rest or None)
    if scheme == "memory" and rest in {"", "fake", "memory"}:
        return FakeBackend()
    if scheme == "local":
        if rest in {"fake", "memory"}:
            # local://fake — explicit in-process fake
            return FakeBackend()
        # local:// — real in-process torch+PEFT backend (Phase 1)
        from anvil.backends.local import LocalBackend

        return LocalBackend(root=rest or None)
    if scheme in {"http", "https"}:
        # http(s)://host:port — remote worker running `anvil serve` (Phase 2)
        import os

        from anvil.client.remote import RemoteBackend

        return RemoteBackend(endpoint, token=os.environ.get("ANVIL_TOKEN"))
    raise ValueError(
        f"unsupported endpoint {endpoint!r}; supported: 'fake://', "
        f"'local://' (torch+PEFT), 'local://fake', 'http(s)://host:port' "
        f"(anvil serve), or pass backend= explicitly"
    )


class ServiceClient:
    """Root client. Create training / sampling clients against a backend.

    Examples::

        svc = anvil.ServiceClient()  # fake:// in-process
        tc = svc.create_lora_training_client(base_model="sshleifer/tiny-gpt2", rank=8)
        fut = tc.forward_backward([datum], loss_fn="cross_entropy")
        loss = fut.result().loss
    """

    def __init__(
        self,
        endpoint: str = "fake://",
        *,
        backend: Backend | None = None,
        queue: bool = True,
    ) -> None:
        self.endpoint = endpoint
        self._backend = resolve_backend(endpoint, backend)
        self._session = Session.create(endpoint=endpoint)
        # P2 (design §4.3): verbs run on a single-worker FIFO so returned
        # futures are genuinely non-blocking and backend state is serialized.
        self._queue = VerbQueue() if queue else None

    @property
    def session(self) -> Session:
        return self._session

    @property
    def backend(self) -> Backend:
        return self._backend

    def close(self) -> None:
        if self._queue is not None:
            self._queue.shutdown()

    def __enter__(self) -> ServiceClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        *,
        alpha: int | None = None,
        dropout: float = 0.0,
        modalities: Sequence[str] = ("text",),
        lora_targets: LoraTargets | None = None,
    ) -> TrainingClient:
        config = TrainConfig(
            base_model=base_model,
            lora=LoraConfig(
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                targets=lora_targets if lora_targets is not None else LoraTargets(),
            ),
            modalities=tuple(modalities),
        )
        adapter_id = (
            self._queue.run(self._backend.create_lora_session, config)
            if self._queue is not None
            else self._backend.create_lora_session(config)
        )
        self._session.registry.register(adapter_id, config)
        return TrainingClient(
            backend=self._backend,
            adapter_id=adapter_id,
            config=config,
            registry=self._session.registry,
            queue=self._queue,
        )

    def create_sampling_client(
        self,
        base_model: str,
        *,
        adapter_id=None,
    ) -> SamplingClient:
        return SamplingClient(
            backend=self._backend,
            base_model=base_model,
            adapter_id=adapter_id,
            queue=self._queue,
        )
