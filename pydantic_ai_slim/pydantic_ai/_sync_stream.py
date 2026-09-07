"""Bridge async streaming context managers to synchronous code on the caller's event loop.

The synchronous streaming wrappers (`Agent.run_stream_sync` and `direct.model_request_stream_sync`) need
to drive an async stream from sync code. Pumping via repeated `loop.run_until_complete(anext(...))` runs
each step in a *different* asyncio task, so any cancel scope the async code enters and exits per step (e.g.
the agent graph's per-node scopes, or `group_by_temporal`'s debouncer) straddles tasks and raises
`RuntimeError: Attempted to exit cancel scope in a different task than it was entered in`. It also leaves
OpenTelemetry spans dangling, since the run span never closes in the task that opened it.

`SyncStreamBridge` instead keeps a long-lived task holding the async context manager open, and each
streaming pass runs its entire `async for` in one child task. An AnyIO task group inside the owner
contains every call and pump, and exits before the stream context. The asyncio adapter drives these
tasks on the caller's loop to preserve the affinity of clients reused across synchronous calls.
"""

from __future__ import annotations

import asyncio
import inspect
import weakref
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator
from contextlib import AbstractAsyncContextManager, suppress
from contextvars import Context, copy_context
from dataclasses import dataclass
from functools import partial
from threading import get_ident
from types import TracebackType
from typing import Generic

import anyio
import anyio.streams.memory
from anyio.abc import TaskGroup
from typing_extensions import TypeIs, TypeVar, TypeVarTuple, Unpack

from . import _utils

T = TypeVar('T')
StreamT = TypeVar('StreamT')
_PosArgsT = TypeVarTuple('_PosArgsT')

_ExitInfo = tuple[type[BaseException] | None, BaseException | None, TracebackType | None]


@dataclass(eq=False)
class _StreamTask(Generic[T]):
    """Forward a group child's outcome to the sync caller without changing its exception type."""

    future: asyncio.Future[T]
    task: asyncio.Task[None] | None = None
    cancel_requested: bool = False

    async def run(self, func: Callable[[], Awaitable[T]]) -> None:
        self.task = asyncio.current_task()
        assert self.task is not None
        if self.cancel_requested:
            self.task.cancel()
        # The owner forwards cancellation once so source cleanup can await.
        with anyio.CancelScope(shield=True):
            try:
                self.future.set_result(await func())
            except asyncio.CancelledError:
                self.future.cancel()
            except BaseException as exc:
                self.future.set_exception(exc)

    def cancel(self) -> None:
        # Preserve edge cancellation so existing source cleanup can await without a cancel scope.
        self.cancel_requested = True
        if self.task is not None:
            self.task.cancel()

    def done(self) -> bool:
        return self.future.done()

    def exception(self) -> BaseException | None:
        return self.future.exception()

    def result(self) -> T:
        return self.future.result()


def _is_awaitable(value: T | Awaitable[T]) -> TypeIs[Awaitable[T]]:
    """Narrow an optionally awaitable result without losing its generic return type."""
    return inspect.isawaitable(value)


async def _hold_context_manager(
    cm: AbstractAsyncContextManager[StreamT],
    entered: asyncio.Future[tuple[StreamT, Context, TaskGroup]],
    exit_requested: asyncio.Future[_ExitInfo],
    start_requests: asyncio.Queue[Callable[[], object]],
    task_cancellations: set[Callable[[], None]],
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
    exit_info: _ExitInfo = (None, None, None)
    try:
        async with anyio.create_task_group() as task_group:
            try:
                entered.set_result((stream, copy_context(), task_group))
                stop = exit_requested.done
                exit_requested.add_done_callback(lambda _: start_requests.put_nowait(stop))
                while (start := await start_requests.get()) is not stop:
                    start()
                    del start
                exit_info = exit_requested.result()
            finally:
                for cancel in tuple(task_cancellations):
                    cancel()
                task_group.cancel_scope.cancel()
    except BaseException as exc:
        if not await cm.__aexit__(type(exc), exc, exc.__traceback__):
            raise
    else:
        # The synchronous wrappers do not support exception suppression, and the context managers used here
        # do not suppress.
        await cm.__aexit__(*exit_info)
    finally:
        while not start_requests.empty():
            start_requests.get_nowait()()


async def _wait_for_task(task: asyncio.Task[None] | _StreamTask[None]) -> None:
    """Wait for a task, then yield once so queued loop-stop callbacks run before this waiter completes."""
    if not task.done():
        await asyncio.wait((task.future if isinstance(task, _StreamTask) else task,))
    await asyncio.sleep(0)


def _run_task_to_completion(loop: asyncio.AbstractEventLoop, task: asyncio.Task[None] | _StreamTask[None]) -> None:
    """Drive a task to completion despite stale `run_until_complete()` stop callbacks."""
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
    loop: asyncio.AbstractEventLoop,
    owner_task: asyncio.Task[None],
    exit_requested: asyncio.Future[_ExitInfo],
    pump_tasks: set[asyncio.Task[None]] | set[_StreamTask[None]],
    exit_info: _ExitInfo,
) -> None:
    """Tell the owner task to exit the stream context manager, then drive its cleanup to completion."""
    tasks = tuple(pump_tasks)
    for task in tasks:
        task.cancel()
    for task in tasks:
        with suppress(BaseException):
            _run_task_to_completion(loop, task)
    pump_tasks.clear()

    if not exit_requested.done():
        exit_requested.set_result(exit_info)
    _run_task_to_completion(loop, owner_task)


async def _request_exit(
    owner_task: asyncio.Task[None],
    exit_requested: asyncio.Future[_ExitInfo],
    pump_tasks: set[asyncio.Task[None]] | set[_StreamTask[None]],
) -> None:
    """Cancel active stream pumps before allowing the context-manager owner to exit."""
    tasks = tuple(pump_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(
            *(task.future if isinstance(task, _StreamTask) else task for task in tasks), return_exceptions=True
        )
    pump_tasks.clear()
    if not exit_requested.done():
        exit_requested.set_result((None, None, None))
    # Garbage-collection cleanup is best effort, but the owner task's exception must still be retrieved
    # so an error from `__aexit__` is not reported later as "Task exception was never retrieved".
    with suppress(BaseException):
        await owner_task


def _finalize_loop(
    loop: asyncio.AbstractEventLoop,
    owner_task: asyncio.Task[None],
    exit_requested: asyncio.Future[_ExitInfo],
    pump_tasks: set[asyncio.Task[None]] | set[_StreamTask[None]],
    owner_thread_id: int,
) -> None:
    """Best-effort finalizer for callers that do not close the synchronous wrapper explicitly."""
    if loop.is_closed() or owner_task.done():
        return

    def request_exit() -> None:
        loop.create_task(_request_exit(owner_task, exit_requested, pump_tasks))

    if get_ident() == owner_thread_id and not loop.is_running():
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            with suppress(BaseException):
                _shutdown_loop(loop, owner_task, exit_requested, pump_tasks, (None, None, None))
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

        loop = _utils.get_event_loop()
        caller_context = copy_context()
        entered: asyncio.Future[tuple[StreamT, Context, TaskGroup]] = loop.create_future()
        exit_requested: asyncio.Future[_ExitInfo] = loop.create_future()
        start_requests: asyncio.Queue[Callable[[], object]] = asyncio.Queue()
        task_cancellations: set[Callable[[], None]] = set()
        owner_task = loop.create_task(
            _hold_context_manager(cm, entered, exit_requested, start_requests, task_cancellations)
        )
        try:
            stream, run_context, task_group = loop.run_until_complete(entered)
        except BaseException:
            if not owner_task.done():
                owner_task.cancel()
            with suppress(BaseException):
                _run_task_to_completion(loop, owner_task)
            # If cancellation reached `cm.__aenter__()`, the owner task forwarded it to `entered`.
            # Retrieve it so the abandoned future cannot report an unhandled exception later.
            with suppress(BaseException):
                entered.result()
            raise

        self.stream = stream
        self._loop = loop
        self._owner_task = owner_task
        self._exit_requested = exit_requested
        self._caller_context = caller_context
        self._run_context = run_context
        self._task_group = task_group
        self._start_requests = start_requests
        self._task_cancellations = task_cancellations
        self._owner_thread_id = get_ident()
        self._pump_tasks: set[_StreamTask[None]] = set()
        # Clean up if the caller never uses the `with` block: exit the stream at GC.
        self._finalizer = weakref.finalize(
            self, _finalize_loop, loop, owner_task, exit_requested, self._pump_tasks, self._owner_thread_id
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
            _shutdown_loop(self._loop, self._owner_task, self._exit_requested, self._pump_tasks, exit_info)

    def _start_task(self, func: Callable[[], Awaitable[T]]) -> _StreamTask[T]:
        task = _StreamTask[T](self._loop.create_future())
        task_group = self._task_group
        task_cancellations = self._task_cancellations
        task_cancellations.add(task.cancel)
        task.future.add_done_callback(lambda _: task_cancellations.discard(task.cancel))

        def start() -> None:
            try:
                task_group.start_soon(task.run, func)
            except RuntimeError as exc:
                task.future.set_exception(exc)

        self._start_requests.put_nowait(partial(self._task_context().run, start))
        return task

    def _run(self, task: _StreamTask[T]) -> T:
        if task.done():
            return task.result()
        waiter = self._loop.create_task(asyncio.wait((task.future,)))
        try:
            self._loop.run_until_complete(waiter)
            return task.result()
        except BaseException:
            if not task.done():
                task.cancel()
            with suppress(BaseException):
                self._loop.run_until_complete(waiter)
            with suppress(BaseException):
                task.exception()
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
        if not self._finalizer.alive or self._owner_task.done():
            raise RuntimeError('This synchronous stream is already closed.')
        self._check_owner_thread()
        try:
            return self._run(self._start_task(partial(self._call, func, *args)))
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
        if not self._finalizer.alive or self._owner_task.done():
            raise RuntimeError('This synchronous stream is already closed.')
        self._check_owner_thread()
        send_stream, receive_stream = anyio.create_memory_object_stream[T](max_buffer_size=0)
        pump_task = self._start_task(partial(self._pump_to_stream, make_aiter, send_stream))
        pump_tasks = self._pump_tasks
        pump_tasks.add(pump_task)

        def discard_pump(future: asyncio.Future[None]) -> None:
            pump_tasks.discard(pump_task)
            # A deferred close may finish without the sync iterator being resumed. Retrieve any error
            # here so it cannot be reported later as "Task exception was never retrieved".
            with suppress(BaseException):
                future.exception()

        pump_task.future.add_done_callback(discard_pump)

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
                    self._loop.call_soon_threadsafe(cancel_pump)

        cleanup_deferred = False
        try:
            while True:
                received = self.call(_receive_one, receive_stream)
                if not _utils.is_set(received):
                    break
                yield received
            # Stream exhausted normally: surface any error raised inside the pump task.
            self._run(pump_task)
        except (KeyboardInterrupt, SystemExit) as exc:
            self.shutdown((type(exc), exc, exc.__traceback__))
            raise
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
                cancel_pump()
                with suppress(BaseException):
                    self._run(pump_task)
                self._pump_tasks.discard(pump_task)
