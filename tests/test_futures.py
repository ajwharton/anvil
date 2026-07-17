"""P2 futures/queue: futures are genuinely non-blocking, FIFO-ordered,
and API-compatible with the inline path."""

from __future__ import annotations

import asyncio
import threading
import time

from anvil.backends.fake import FakeBackend
from anvil.client.futures import AnvilFuture
from anvil.client.service import ServiceClient
from anvil.protocol.types import Datum, LossFn, ModelInput


class GatedBackend(FakeBackend):
    """FakeBackend whose verbs record call order and can be held open."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.gate.set()
        self.calls: list[str] = []

    def _record(self, name: str) -> None:
        self.calls.append(name)
        self.gate.wait(timeout=5.0)

    def forward_backward(self, adapter_id, data, loss_fn):  # type: ignore[override]
        self._record("fb")
        return super().forward_backward(adapter_id, data, loss_fn)

    def optim_step(self, adapter_id, adam):  # type: ignore[override]
        self._record("optim")
        return super().optim_step(adapter_id, adam)


def _datum() -> Datum:
    return Datum(
        model_input=ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={"target_tokens": [2, 3], "weights": [1.0, 1.0]},
    )


def test_futures_are_nonblocking_and_pipelinable():
    backend = GatedBackend()
    backend.gate.clear()  # hold the first verb open
    svc = ServiceClient(backend=backend)
    try:
        tc = svc.create_lora_training_client(base_model="toy/lm", rank=4)
        t0 = time.monotonic()
        fut = tc.forward_backward([_datum()], loss_fn=LossFn.CROSS_ENTROPY)
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, f"caller blocked for {elapsed:.2f}s"
        assert not fut.done()
        # pipelining: a second verb queues behind the first without blocking
        fut2 = tc.optim_step()
        assert not fut2.done()
        backend.gate.set()
        out = fut.result(timeout=5.0)
        assert out.loss == out.loss  # finite
        fut2.result(timeout=5.0)
    finally:
        svc.close()


def test_queue_is_fifo():
    backend = GatedBackend()
    backend.gate.clear()
    svc = ServiceClient(backend=backend)
    try:
        tc = svc.create_lora_training_client(base_model="toy/lm", rank=4)
        f1 = tc.forward_backward([_datum()])
        f2 = tc.optim_step()
        f3 = tc.forward_backward([_datum()])
        backend.gate.set()
        for f in (f1, f2, f3):
            f.result(timeout=5.0)
        assert backend.calls == ["fb", "optim", "fb"]
    finally:
        svc.close()


class RaisingBackend(GatedBackend):
    def forward_backward(self, adapter_id, data, loss_fn):  # type: ignore[override]
        self._record("fb")
        raise ValueError("boom")


def test_queue_propagates_exceptions():
    backend = RaisingBackend()
    svc = ServiceClient(backend=backend)
    try:
        tc = svc.create_lora_training_client(base_model="toy/lm", rank=4)
        fut = tc.forward_backward([_datum()])
        assert isinstance(fut.exception(timeout=5.0), ValueError)
        try:
            fut.result(timeout=1.0)
            raise AssertionError("should have raised")
        except AssertionError:
            raise
        except Exception:
            pass
    finally:
        svc.close()


def test_awaitable_and_async_result():
    backend = GatedBackend()
    svc = ServiceClient(backend=backend)
    try:
        tc = svc.create_lora_training_client(base_model="toy/lm", rank=4)
        fut = tc.optim_step()

        async def get():
            return await fut

        out = asyncio.run(get())
        assert out.metrics is not None
    finally:
        svc.close()


def test_queue_false_keeps_inline_behavior():
    backend = GatedBackend()
    svc = ServiceClient(backend=backend, queue=False)
    try:
        tc = svc.create_lora_training_client(base_model="toy/lm", rank=4)
        fut = tc.forward_backward([_datum()])
        assert isinstance(fut, AnvilFuture)
        assert fut.done()  # resolved inline, like before P2
        assert backend.calls == ["fb"]
    finally:
        svc.close()
