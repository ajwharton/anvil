"""Future handles so clients can pipeline ops (Tinker-shaped).

Local/fake backends resolve immediately; real GPU backends will queue work
and keep the same ``result()`` / ``result_async()`` surface.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class AnvilFuture(Generic[T]):
    """Thin wrapper over a concurrent.futures.Future."""

    __slots__ = ("_fut",)

    def __init__(self, fut: ConcurrentFuture[T]) -> None:
        self._fut = fut

    def result(self, timeout: float | None = None) -> T:
        return self._fut.result(timeout=timeout)

    def done(self) -> bool:
        return self._fut.done()

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._fut.exception(timeout=timeout)

    async def result_async(self, timeout: float | None = None) -> T:
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, lambda: self._fut.result()),
            timeout=timeout,
        )

    def __await__(self):
        return self.result_async().__await__()


def completed(value: T) -> AnvilFuture[T]:
    """Return an already-resolved future (fake/local path)."""
    fut: ConcurrentFuture[T] = ConcurrentFuture()
    fut.set_result(value)
    return AnvilFuture(fut)


def failed(exc: BaseException) -> AnvilFuture[T]:
    fut: ConcurrentFuture[T] = ConcurrentFuture()
    fut.set_exception(exc)
    return AnvilFuture(fut)


class VerbQueue:
    """Single-worker FIFO queue behind one backend — the P2 async path.

    One thread means every verb submitted through a ServiceClient executes in
    submission order (forward_backward before the optim_step that follows it,
    per adapter and globally) and backend state is never touched concurrently.
    On a GPU backend this is the "true async GPU queue" of design §4.3; the
    caller-facing surface (AnvilFuture.result/done/exception/await) is
    identical to the inline path, so clients and recipes don't care which
    side of the queue a backend sits on.
    """

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anvil-verbs")

    def submit(self, fn: Callable[..., T], *args, **kwargs) -> AnvilFuture[T]:
        return AnvilFuture(self._pool.submit(fn, *args, **kwargs))

    def run(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Submit and block — for verbs whose API returns a plain value."""
        return self.submit(fn, *args, **kwargs).result()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
