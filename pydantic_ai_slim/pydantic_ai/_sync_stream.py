"""Bridge async streaming context managers to synchronous code via a dedicated event-loop thread.

The synchronous streaming wrappers (`Agent.run_stream_sync` and `direct.model_request_stream_sync`) need
to drive an async stream from sync code. Pumping via repeated `loop.run_until_complete(anext(...))` runs
each step in a *different* asyncio task, so any cancel scope the async code enters and exits per step (e.g.
the agent graph's per-node scopes, or `group_by_temporal`'s debouncer) straddles tasks and raises
`RuntimeError: Attempted to exit cancel scope in a different task than it was entered in`. It also leaves
OpenTelemetry spans dangling, since the run span never closes in the task that opened it.

`SyncStreamBridge` instead runs the whole stream on a single dedicated event-loop thread (an
[`anyio` blocking portal][anyio.from_thread.BlockingPortal]): the async context manager is entered and
exited in the same portal task, and each streaming pass runs its entire `async for` in one task.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator
from contextlib import AbstractAsyncContextManager, AbstractContextManager, asynccontextmanager, suppress
from types import TracebackType
from typing import Any, Generic, cast

import anyio
import anyio.streams.memory
from anyio.from_thread import BlockingPortal, start_blocking_portal
from opentelemetry import context as otel_context
from typing_extensions import TypeVar, TypeVarTuple, Unpack

from . import _utils

T = TypeVar('T')
StreamT = TypeVar('StreamT')
_PosArgsT = TypeVarTuple('_PosArgsT')

_ExcInfo = tuple[type[BaseException] | None, BaseException | None, TracebackType | None]


@asynccontextmanager
async def _capture_otel_context(
    cm: AbstractAsyncContextManager[StreamT], captured: list[otel_context.Context]
) -> AsyncGenerator[StreamT]:
    """Enter `cm` and capture the OTel context that's active at its yield point.

    `cm` opens its run/request span in the task it's entered in. We capture that task's OTel context so
    that operations we later run in *other* portal tasks (streaming, `get_output`, ...) can re-attach it,
    keeping their child spans parented under the run span and the run span's attributes complete.
    """
    async with cm as stream:
        captured.append(otel_context.get_current())
        yield stream


def _shutdown_portal(
    portal_cm: AbstractContextManager[BlockingPortal],
    stream_cm: AbstractContextManager[Any],
    exc_info: _ExcInfo,
) -> None:
    """Exit the (portal-held) stream context manager, then stop the portal thread."""
    try:
        stream_cm.__exit__(*exc_info)
    finally:
        portal_cm.__exit__(None, None, None)


class SyncStreamBridge(Generic[StreamT]):
    """Runs an async streaming context manager on a dedicated event-loop thread and bridges it to sync.

    Constructing the bridge enters `cm` inside an [`anyio` blocking portal][anyio.from_thread.BlockingPortal]
    and exposes the yielded object as [`stream`][pydantic_ai._sync_stream.SyncStreamBridge.stream]. Cancel
    scopes entered and exited by the async code never straddle tasks, and OpenTelemetry spans stay correctly
    nested. The owning sync wrapper calls [`shutdown`][pydantic_ai._sync_stream.SyncStreamBridge.shutdown]
    (from its own `__exit__`) to exit the stream and stop the portal thread; a `weakref.finalize` fallback
    does the same at garbage collection if the wrapper is dropped without being closed.
    """

    stream: StreamT
    """The object yielded by the async context manager (entered in the portal task)."""

    def __init__(self, cm: AbstractAsyncContextManager[StreamT], *, async_alternative: str) -> None:
        """Enter `cm` on a fresh portal thread, capturing its OTel context.

        Args:
            cm: The async streaming context manager to run on the portal thread.
            async_alternative: How to name the async counterpart in error messages (e.g. `run_stream`).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                f'Cannot use a synchronous streaming method from within an async context or a running '
                f'event loop; use {async_alternative} instead.'
            )
        if _utils._disable_threads.get():  # pyright: ignore[reportPrivateUsage]
            raise RuntimeError(
                f'Synchronous streaming runs on a dedicated event-loop thread, which is unavailable in this '
                f'environment (e.g. emscripten or a Temporal workflow); use {async_alternative} instead.'
            )

        captured: list[otel_context.Context] = []
        portal_cm = start_blocking_portal()
        portal = portal_cm.__enter__()
        try:
            stream_cm = portal.wrap_async_context_manager(_capture_otel_context(cm, captured))
            stream = stream_cm.__enter__()
        except BaseException as exc:
            portal_cm.__exit__(type(exc), exc, exc.__traceback__)
            raise

        self.stream = stream
        self._portal = portal
        self._portal_cm = portal_cm
        self._stream_cm = stream_cm
        # The run span's OTel context, captured in the portal task that holds `cm` open.
        self._otel_context = captured[0]
        # Clean up if the caller never uses the `with` block: exit the stream and stop the portal at GC.
        self._finalizer = weakref.finalize(self, _shutdown_portal, portal_cm, stream_cm, (None, None, None))

    def shutdown(self, exc_info: _ExcInfo = (None, None, None)) -> None:
        """Exit the stream context manager and stop the portal thread, at most once.

        `detach()` disarms the finalizer (returning true iff it was still live), guarding against a double
        shutdown from the owning wrapper's `__exit__`, a Ctrl-C teardown, and a later GC. The exception is
        propagated into the stream context manager so it can tear the stream down correctly.
        """
        if self._finalizer.detach() is not None:
            _shutdown_portal(self._portal_cm, self._stream_cm, exc_info)

    def call(self, func: Callable[[Unpack[_PosArgsT]], Awaitable[T] | T], *args: Unpack[_PosArgsT]) -> T:
        """Run `func` in the portal task, tearing the run down if the caller is interrupted while it blocks.

        Without this, a `KeyboardInterrupt` (Ctrl-C) or `SystemExit` landing while we're blocked on the
        portal would unwind the caller while leaving the async code's pending tasks and open sockets
        running on the portal thread until garbage collection. See #5975.
        """
        try:
            # `portal.call` accepts sync or async `func` (`Awaitable[T] | T`); the union defeats the
            # return-type inference, so we narrow it back the same way anyio does internally.
            return cast(T, self._portal.call(func, *args))
        except (KeyboardInterrupt, SystemExit) as exc:
            self.shutdown((type(exc), exc, exc.__traceback__))
            raise

    def call_with_otel_context(self, func: Callable[[], Awaitable[T]]) -> T:
        """Like [`call`][pydantic_ai._sync_stream.SyncStreamBridge.call], with the run span's OTel context attached."""
        return self.call(self._run_with_otel_context, func)

    async def _run_with_otel_context(self, func: Callable[[], Awaitable[T]]) -> T:
        token = otel_context.attach(self._otel_context)
        try:
            return await func()
        finally:
            otel_context.detach(token)

    async def _pump_to_stream(
        self, make_aiter: Callable[[], AsyncIterator[T]], send_stream: anyio.streams.memory.MemoryObjectSendStream[T]
    ) -> None:
        """Drive `make_aiter()` to completion in a single portal task, forwarding items to `send_stream`.

        Running the whole `async for` in one task keeps the source iterator's cancel scopes (e.g.
        `group_by_temporal`'s) from being entered and exited in different tasks.
        """
        token = otel_context.attach(self._otel_context)
        try:
            async with send_stream:
                aiter = make_aiter()
                try:
                    async for item in aiter:
                        try:
                            await send_stream.send(item)
                        except anyio.BrokenResourceError:
                            # The consumer stopped iterating early and closed the receive end.
                            return
                finally:
                    # The source iterators are async generators at runtime even though they're typed as
                    # `AsyncIterator`, so this narrows to the closable case.
                    if isinstance(aiter, AsyncGenerator):  # pragma: no branch
                        await aiter.aclose()
        finally:
            otel_context.detach(token)

    def stream_sync(self, make_aiter: Callable[[], AsyncIterator[T]]) -> Iterator[T]:
        """Synchronously iterate the items produced by `make_aiter()`, driven on the portal thread."""
        send_stream, receive_stream = anyio.create_memory_object_stream[T](max_buffer_size=0)
        future = self._portal.start_task_soon(self._pump_to_stream, make_aiter, send_stream)
        try:
            while True:
                try:
                    yield self.call(receive_stream.receive)
                except anyio.EndOfStream:
                    break
            # Stream exhausted normally: surface any error raised inside the pump task.
            future.result()
        finally:
            # Unblock the pump (in case of early exit), then wait for it to finish its own cleanup
            # (closing the source iterator). Suppress errors raised purely during that teardown
            # (including the portal already being shut down), matching the async iterator's `aclose`
            # behavior.
            with suppress(BaseException):
                self._portal.call(receive_stream.aclose)
            with suppress(BaseException):
                future.result()
            # If the portal was already torn down (e.g. by a `KeyboardInterrupt`), the on-loop `aclose`
            # above was suppressed and the receive stream is still open; close it synchronously so it
            # isn't leaked. Safe and idempotent here: the pump is no longer running, so there are no
            # waiting senders to wake across threads.
            receive_stream.close()
