"""Bridge async streaming context managers to synchronous code on the caller's event loop.

The synchronous streaming wrappers (`Agent.run_stream_sync` and `direct.model_request_stream_sync`) need
to drive an async stream from sync code. Pumping via repeated `loop.run_until_complete(anext(...))` runs
each step in a *different* asyncio task, so any cancel scope the async code enters and exits per step (e.g.
the agent graph's per-node scopes, or `group_by_temporal`'s debouncer) straddles tasks and raises
`RuntimeError: Attempted to exit cancel scope in a different task than it was entered in`. It also leaves
OpenTelemetry spans dangling, since the run span never closes in the task that opened it.

`SyncStreamBridge` instead keeps a long-lived task holding the async context manager open, and each
streaming pass runs its entire `async for` in another long-lived task. All tasks run on the caller's event
loop, preserving the event-loop affinity of async clients and other resources reused across sync calls.

Which loop that is depends on where the caller is: sync code normally owns the loop and drives it itself,
but a synchronous callback of an enclosing run reaches back to the loop that run is already using. See
`_StreamLoop`.
"""

from __future__ import annotations

import asyncio
import inspect
import weakref
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator
from contextlib import AbstractAsyncContextManager, suppress
from contextvars import Context, copy_context
from dataclasses import dataclass
from threading import get_ident
from types import TracebackType
from typing import Generic

import anyio
import anyio.streams.memory
from typing_extensions import TypeIs, TypeVar, TypeVarTuple, Unpack

from . import _utils

T = TypeVar('T')
StreamT = TypeVar('StreamT')
_PosArgsT = TypeVarTuple('_PosArgsT')

_ExitInfo = tuple[type[BaseException] | None, BaseException | None, TracebackType | None]


def _is_awaitable(value: T | Awaitable[T]) -> TypeIs[Awaitable[T]]:
    """Narrow an optionally awaitable result without losing its generic return type."""
    return inspect.isawaitable(value)


async def _ensure_task(awaitable: Awaitable[T], context: Context) -> asyncio.Task[T]:
    return context.run(asyncio.ensure_future, awaitable)


async def _await_future(future: asyncio.Future[T]) -> T:
    return await future


async def _call_now(func: Callable[[], object]) -> None:
    func()


@dataclass
class _StreamLoop:
    """How synchronous code reaches the event loop its stream runs on.

    Sync code normally owns its event loop and drives it with `run_until_complete()`. Inside a synchronous
    tool or output function of an enclosing run, the loop that run uses is already running on another thread,
    so work is submitted to it and this thread blocks on the result instead. Driving a second loop there
    would strand the enclosing run's async resources -- like a provider's HTTP connection pool -- on the
    first one, which is the same failure the module docstring's affinity guarantee exists to prevent.
    """

    loop: asyncio.AbstractEventLoop
    """The event loop every task of this stream runs on."""

    host: _utils.SyncCallbackHost | None
    """Set when `loop` belongs to an enclosing run, and this thread must hand work to it rather than drive it."""

    def ensure_task(self, awaitable: Awaitable[T], context: Context) -> asyncio.Task[T]:
        """Schedule `awaitable` on the loop as a task running in `context`."""
        if self.host is None:
            return context.run(asyncio.ensure_future, awaitable, loop=self.loop)
        return self.host.run(_ensure_task(awaitable, context))

    def wait(self, future: asyncio.Future[T]) -> T:
        """Block until `future` completes on the loop, and return its result."""
        if self.host is None:
            return self.loop.run_until_complete(future)
        return self.host.run(_await_future(future))

    def run_now(self, func: Callable[[], object]) -> None:
        """Run `func` on the thread that drives the loop, and wait for it to finish.

        Cancelling a task, closing a memory stream and resolving a future all schedule loop callbacks, so
        they have to happen on that thread rather than on whichever thread the synchronous caller is.
        """
        if self.host is None:
            func()
        else:
            self.host.run(_call_now(func))


async def _hold_context_manager(
    cm: AbstractAsyncContextManager[StreamT],
    entered: asyncio.Future[tuple[StreamT, Context]],
    exit_requested: asyncio.Future[_ExitInfo],
) -> None:
    """Enter and exit `cm` in one task, remaining parked while sync code uses the yielded stream."""
    try:
        stream = await cm.__aenter__()
    except (KeyboardInterrupt, SystemExit):
        # Futures deliberately do not stop `run_until_complete()` for these exceptions because tasks
        # normally re-raise them directly from the event loop. Preserve that behavior instead of
        # forwarding them through `entered`, which would leave the loop running indefinitely.
        raise
    except BaseException as exc:
        entered.set_exception(exc)
        return

    # Context changes made by `__aenter__()` stay in this owner task, so return its snapshot alongside
    # the stream for child call and pump tasks to inherit.
    entered.set_result((stream, copy_context()))
    try:
        exit_info = await exit_requested
    except BaseException as exc:
        if not await cm.__aexit__(type(exc), exc, exc.__traceback__):
            raise
    else:
        # The synchronous wrappers do not support exception suppression, and the context managers used here
        # do not suppress.
        await cm.__aexit__(*exit_info)


async def _wait_for_task(task: asyncio.Task[None]) -> None:
    """Wait for a task, then yield once so queued loop-stop callbacks run before this waiter completes."""
    if not task.done():
        await asyncio.wait((task,))
    await asyncio.sleep(0)


def _run_task_to_completion(stream_loop: _StreamLoop, task: asyncio.Task[None]) -> None:
    """Drive a task to completion despite stale `run_until_complete()` stop callbacks."""
    if stream_loop.host is not None:
        # The loop is driven by its own thread, so there are no stop callbacks of ours to work around.
        stream_loop.wait(task)
        return

    loop = stream_loop.loop
    waiter = loop.create_task(_wait_for_task(task))
    try:
        while not waiter.done():
            try:
                loop.run_until_complete(waiter)
            except RuntimeError:
                # `_wait_for_task()` cannot raise `RuntimeError`, so retry only while its waiter remains
                # pending and this thread can still drive the loop.
                if loop.is_closed() or loop.is_running() or waiter.done():
                    raise
    except BaseException:
        # Interrupts and other base exceptions must not strand either task. Finish best-effort cleanup,
        # retrieve the original task's exception, then re-raise the exception that interrupted this drive.
        with suppress(BaseException):
            loop.run_until_complete(waiter)
        with suppress(BaseException):
            task.exception()
        raise
    task.result()


def _shutdown_loop(
    stream_loop: _StreamLoop,
    owner_task: asyncio.Task[None],
    exit_requested: asyncio.Future[_ExitInfo],
    pump_tasks: set[asyncio.Task[None]],
    exit_info: _ExitInfo,
) -> None:
    """Tell the owner task to exit the stream context manager, then drive its cleanup to completion."""
    tasks = tuple(pump_tasks)
    for task in tasks:
        stream_loop.run_now(task.cancel)
    for task in tasks:
        with suppress(BaseException):
            _run_task_to_completion(stream_loop, task)
    pump_tasks.clear()

    def resolve_exit() -> None:
        if not exit_requested.done():
            exit_requested.set_result(exit_info)

    # Resolving the future schedules the owner task's continuation, so it belongs on the loop's own thread.
    stream_loop.run_now(resolve_exit)
    _run_task_to_completion(stream_loop, owner_task)


async def _request_exit(
    owner_task: asyncio.Task[None],
    exit_requested: asyncio.Future[_ExitInfo],
    pump_tasks: set[asyncio.Task[None]],
) -> None:
    """Cancel active stream pumps before allowing the context-manager owner to exit."""
    tasks = tuple(pump_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    pump_tasks.clear()
    if not exit_requested.done():
        exit_requested.set_result((None, None, None))
    # Garbage-collection cleanup is best effort, but the owner task's exception must still be retrieved
    # so an error from `__aexit__` is not reported later as "Task exception was never retrieved".
    with suppress(BaseException):
        await owner_task


def _finalize_loop(
    stream_loop: _StreamLoop,
    owner_task: asyncio.Task[None],
    exit_requested: asyncio.Future[_ExitInfo],
    pump_tasks: set[asyncio.Task[None]],
    owner_thread_id: int,
) -> None:
    """Best-effort finalizer for callers that do not close the synchronous wrapper explicitly."""
    loop = stream_loop.loop
    if loop.is_closed() or owner_task.done():
        return

    def request_exit() -> None:
        loop.create_task(_request_exit(owner_task, exit_requested, pump_tasks))

    # A loop borrowed from an enclosing run is never ours to drive, even once that run has stopped driving it.
    if stream_loop.host is None and get_ident() == owner_thread_id and not loop.is_running():
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            with suppress(BaseException):
                _shutdown_loop(stream_loop, owner_task, exit_requested, pump_tasks, (None, None, None))
            return

    # The owner loop is running, this is a foreign thread, or another loop is active in this thread.
    with suppress(RuntimeError):
        loop.call_soon_threadsafe(request_exit)


async def _receive_one(receive_stream: anyio.streams.memory.MemoryObjectReceiveStream[T]) -> T | _utils.Unset:
    """Receive one item without leaking `EndOfStream` through an asyncio task traceback."""
    try:
        return await receive_stream.receive()
    except anyio.EndOfStream:
        return _utils.UNSET


class SyncStreamBridge(Generic[StreamT]):
    """Runs an async streaming context manager on the caller's event loop and bridges it to sync.

    Constructing the bridge enters `cm` in a long-lived owner task and exposes the yielded object as
    [`stream`][pydantic_ai._sync_stream.SyncStreamBridge.stream]. Cancel scopes entered and exited by the
    async code never straddle tasks, OpenTelemetry spans stay correctly nested, and async resources remain
    on the same event loop across synchronous calls. The owning sync wrapper calls
    [`shutdown`][pydantic_ai._sync_stream.SyncStreamBridge.shutdown] (from its own `__exit__`) to exit the
    stream. A `weakref.finalize` fallback requests the same cleanup if the wrapper is dropped without
    being closed, but callers should use the wrapper as a context manager for deterministic cleanup.
    """

    stream: StreamT
    """The object yielded by the async context manager."""

    def __init__(self, cm: AbstractAsyncContextManager[StreamT], *, async_alternative: str) -> None:
        """Enter `cm` in a persistent task on the caller's event loop, capturing its context variables.

        Args:
            cm: The async streaming context manager to run on the caller's event loop.
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

        host = _utils.sync_callback_host()
        stream_loop = _StreamLoop(host.loop if host is not None else _utils.get_event_loop(), host)
        loop = stream_loop.loop
        caller_context = copy_context()
        entered: asyncio.Future[tuple[StreamT, Context]] = loop.create_future()
        exit_requested: asyncio.Future[_ExitInfo] = loop.create_future()
        owner_task = stream_loop.ensure_task(_hold_context_manager(cm, entered, exit_requested), copy_context())
        try:
            stream, run_context = stream_loop.wait(entered)
        except BaseException:
            if not owner_task.done():
                stream_loop.run_now(owner_task.cancel)
            with suppress(BaseException):
                _run_task_to_completion(stream_loop, owner_task)
            # If cancellation reached `cm.__aenter__()`, the owner task forwarded it to `entered`.
            # Retrieve it so the abandoned future cannot report an unhandled exception later.
            with suppress(BaseException):
                entered.result()
            raise

        self.stream = stream
        self._stream_loop = stream_loop
        self._owner_task = owner_task
        self._exit_requested = exit_requested
        self._caller_context = caller_context
        self._run_context = run_context
        self._owner_thread_id = get_ident()
        self._pump_tasks: set[asyncio.Task[None]] = set()
        # Clean up if the caller never uses the `with` block: exit the stream at GC.
        self._finalizer = weakref.finalize(
            self, _finalize_loop, stream_loop, owner_task, exit_requested, self._pump_tasks, self._owner_thread_id
        )

    def _task_context(self) -> Context:
        """Merge run-owned context changes into the sync caller's current context."""
        context = copy_context()
        for var in self._run_context:
            value = self._run_context[var]
            if var not in self._caller_context or self._caller_context[var] is not value:
                context.run(var.set, value)
        return context

    def _check_owner_thread(self) -> None:
        if get_ident() != self._owner_thread_id:
            raise RuntimeError('A synchronous stream must be used and closed on the thread where it was created.')

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError('A synchronous stream cannot be used or closed while an event loop is running.')

    def shutdown(self, exit_info: _ExitInfo = (None, None, None)) -> None:
        """Exit the stream context manager, at most once.

        `detach()` disarms the finalizer (returning true iff it was still live), guarding against a double
        shutdown from the owning wrapper's `__exit__`, a Ctrl-C teardown, and a later GC. The `__exit__`
        arguments are passed to the stream context manager so it can tear the stream down correctly.
        """
        self._check_owner_thread()
        if self._finalizer.detach() is not None:
            _shutdown_loop(self._stream_loop, self._owner_task, self._exit_requested, self._pump_tasks, exit_info)

    def _run(self, awaitable: Awaitable[T]) -> T:
        """Run `awaitable` on the bridge's event loop and clean up its task if the caller interrupts."""
        task = self._stream_loop.ensure_task(awaitable, self._task_context())
        # `run_until_complete()` deliberately does not stop for a Future that already holds
        # `KeyboardInterrupt` or `SystemExit`. Read completed tasks directly so teardown cannot hang.
        if task.done():
            return task.result()
        try:
            return self._stream_loop.wait(task)
        except BaseException:
            if not task.done():
                self._stream_loop.run_now(task.cancel)
                with suppress(BaseException):
                    self._stream_loop.wait(task)
            raise

    async def _call(self, func: Callable[[Unpack[_PosArgsT]], Awaitable[T] | T], *args: Unpack[_PosArgsT]) -> T:
        result = func(*args)
        if _is_awaitable(result):
            return await result
        return result

    def call(self, func: Callable[[Unpack[_PosArgsT]], Awaitable[T] | T], *args: Unpack[_PosArgsT]) -> T:
        """Run `func` on the bridge's event loop, tearing the run down if the caller is interrupted.

        Without this, a `KeyboardInterrupt` (Ctrl-C) or `SystemExit` landing while we're blocked on the
        event loop would unwind the caller while leaving the async code's pending tasks and open sockets
        until garbage collection. See https://github.com/pydantic/pydantic-ai/issues/5975.
        """
        if not self._finalizer.alive:
            raise RuntimeError('This synchronous stream is already closed.')
        self._check_owner_thread()
        try:
            return self._run(self._call(func, *args))
        except (KeyboardInterrupt, SystemExit) as exc:
            self.shutdown((type(exc), exc, exc.__traceback__))
            raise

    @staticmethod
    async def _pump_to_stream(
        make_aiter: Callable[[], AsyncIterator[T]], send_stream: anyio.streams.memory.MemoryObjectSendStream[T]
    ) -> None:
        """Drive `make_aiter()` to completion in one task, forwarding items to `send_stream`.

        Running the whole `async for` in one task keeps the source iterator's cancel scopes (e.g.
        `group_by_temporal`'s) from being entered and exited in different tasks.
        """
        async with send_stream:
            aiter = make_aiter()
            try:
                async for item in aiter:
                    await send_stream.send(item)
            finally:
                # The source iterators are async generators at runtime even though they're typed as
                # `AsyncIterator`, so this narrows to the closable case.
                if isinstance(aiter, AsyncGenerator):  # pragma: no branch
                    await aiter.aclose()

    def stream_sync(self, make_aiter: Callable[[], AsyncIterator[T]]) -> Iterator[T]:
        """Synchronously iterate the items produced by `make_aiter()` on the bridge's event loop."""
        if not self._finalizer.alive:
            raise RuntimeError('This synchronous stream is already closed.')
        self._check_owner_thread()
        send_stream, receive_stream = anyio.create_memory_object_stream[T](max_buffer_size=0)
        pump_task = self._stream_loop.ensure_task(self._pump_to_stream(make_aiter, send_stream), self._task_context())
        pump_tasks = self._pump_tasks
        pump_tasks.add(pump_task)

        def discard_pump(task: asyncio.Task[None]) -> None:
            pump_tasks.discard(task)
            # A deferred close may finish without the sync iterator being resumed. Retrieve any error
            # here so it cannot be reported later as "Task exception was never retrieved".
            with suppress(BaseException):
                task.exception()

        pump_task.add_done_callback(discard_pump)

        def cancel_pump() -> None:
            receive_stream.close()
            if not pump_task.done():
                pump_task.cancel()

        def defer_pump_cleanup() -> None:
            if pump_task.done():
                # The send side has already exited, so closing the receive side cannot wake loop-bound
                # senders and is safe even if shutdown completed before foreign-thread iterator GC.
                receive_stream.close()
            else:
                with suppress(RuntimeError):
                    self._stream_loop.loop.call_soon_threadsafe(cancel_pump)

        cleanup_deferred = False
        try:
            while True:
                received = self.call(_receive_one, receive_stream)
                if not _utils.is_set(received):
                    break
                yield received
            # Stream exhausted normally: surface any error raised inside the pump task.
            self._run(pump_task)
        except GeneratorExit:
            # Explicit close and CPython's implicit close during GC both inject `GeneratorExit`. If that
            # happens off the owner thread, queue cleanup without raising an unraisable exception from GC.
            try:
                self._check_owner_thread()
            except RuntimeError:
                defer_pump_cleanup()
                cleanup_deferred = True
                return
            raise
        finally:
            if not cleanup_deferred:
                # Resuming iteration can also reach this block on another thread. Check again before
                # touching the receive stream or pump task so cleanup cannot move the caller-owned loop.
                try:
                    self._check_owner_thread()
                except RuntimeError:
                    # Queue cleanup for the owner loop. It will run when the owner next drives that loop,
                    # including during `shutdown()`, without touching loop-bound state from this thread.
                    defer_pump_cleanup()
                    raise
                # Closing and cancelling unblocks the pump whether it is waiting to send or waiting on the
                # source iterator. Then drive its task so the source iterator closes in that same task.
                self._stream_loop.run_now(cancel_pump)
                with suppress(BaseException):
                    self._run(pump_task)
                self._pump_tasks.discard(pump_task)
