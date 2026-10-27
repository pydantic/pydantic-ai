from __future__ import annotations

import asyncio
import heapq
import sys
from asyncio.tasks import _enter_task, _leave_task  # pyright: ignore[reportPrivateUsage]
from collections import deque
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator
from contextlib import contextmanager
from contextvars import Context, ContextVar
from datetime import UTC, datetime, timedelta
from typing import Protocol, TypeVar, cast
from weakref import WeakSet

from durabletask import task

from pydantic_ai._utils import disable_threads
from pydantic_ai.exceptions import UserError

T = TypeVar('T')
__all__ = ('run_durable', 'current_loop')


class _RunnableHandle(Protocol):
    def _run(self) -> None: ...


def _as_task(value: task.Task[T]) -> task.Task[object]:
    # The SDK's Task is invariant; the trampoline only reads results from this heterogeneous queue.
    return cast(task.Task[object], value)


_CURRENT_LOOP: ContextVar[_TrampolineLoop | None] = ContextVar('durabletask_loop', default=None)


def current_loop() -> _TrampolineLoop | None:
    return _CURRENT_LOOP.get()


class _TrampolineLoop(asyncio.BaseEventLoop):
    """Drive async agent code on the SDK's orchestration thread for one dispatch.

    Only native tasks/timers cross a yield. Each dispatch reconstructs the async
    scopes, keeping async work local to this loop.
    Activities instead use the backend's separate worker-owned portal.
    """

    def __init__(self, context: task.OrchestrationContext) -> None:
        super().__init__()
        self._ready: deque[asyncio.Handle]
        self._scheduled: list[asyncio.TimerHandle]
        self._asyncgens: WeakSet[AsyncGenerator[object, object]]
        self.context = context
        self.pending: list[tuple[task.Task[object], asyncio.Future[object]]] = []
        self.timers: dict[float, task.Task[object]] = {}
        self.abandoned = False

    def time(self) -> float:
        # SDK versions returning naive UTC must not be interpreted in the worker's local timezone.
        return self.context.current_utc_datetime.replace(tzinfo=UTC).timestamp()

    async def wait(self, durable_task: task.Task[T]) -> T:
        future: asyncio.Future[object] = self.create_future()
        self.pending.append((_as_task(durable_task), future))
        return cast(T, await future)

    def call_soon_threadsafe(
        self, callback: Callable[..., object], *args: object, context: Context | None = None
    ) -> asyncio.Handle:
        raise UserError('Threads and external I/O are not supported inside a Durable Task orchestration.')

    @contextmanager
    def activated(self) -> Generator[None]:
        previous = asyncio._get_running_loop()
        previous_task = asyncio.current_task(previous) if previous is not None else None
        previous_hooks = sys.get_asyncgen_hooks()
        token = _CURRENT_LOOP.set(self)
        # Python 3.14 tracks the active task per thread, even when switching loops.
        # No public API exposes the same-thread task handoff needed during cleanup.
        if previous_task is not None:
            _leave_task(previous_task.get_loop(), previous_task)
        asyncio._set_running_loop(self)
        sys.set_asyncgen_hooks(firstiter=self._asyncgen_firstiter_hook, finalizer=self._asyncgen_finalizer_hook)
        try:
            with disable_threads():
                yield
        finally:
            sys.set_asyncgen_hooks(*previous_hooks)
            asyncio._set_running_loop(previous)
            _CURRENT_LOOP.reset(token)
            if previous_task is not None:
                _enter_task(previous_task.get_loop(), previous_task)

    def _asyncgen_finalizer_hook(self, generator: AsyncGenerator[object, object]) -> None:
        self._asyncgens.discard(generator)
        # Orchestrations run on the worker thread; no selector or thread-safe wakeup is needed.
        self.call_soon(self.create_task, generator.aclose())

    def _asyncgen_firstiter_hook(self, generator: AsyncGenerator[object, object]) -> None:
        self._asyncgens.add(generator)

    def drain(self) -> None:
        with self.activated():
            while self._scheduled and self._scheduled[0].when() <= self.time():
                self._ready.append(heapq.heappop(self._scheduled))
            # Defer newly scheduled callbacks so drive() can recheck root completion.
            for _ in range(len(self._ready)):
                handle = self._ready.popleft()
                if not handle.cancelled():
                    cast(_RunnableHandle, handle)._run()  # pyright: ignore[reportPrivateUsage]

    def drain_cleanup(self) -> None:
        self.drain()
        # Callback-only loops must not keep an otherwise completed cleanup alive.
        while self._ready and asyncio.all_tasks(self):
            self.drain()

    def next_task(self) -> task.Task[object]:
        self.pending = [(durable, future) for durable, future in self.pending if not future.done()]
        durable_tasks = [durable for durable, _ in self.pending]
        while self._scheduled and self._scheduled[0].cancelled():
            heapq.heappop(self._scheduled)
        timer: task.Task[object] | None = None
        if self._scheduled:
            when = self._scheduled[0].when()
            timer = self.timers.get(when)
            if timer is None:
                deadline = datetime.fromtimestamp(when, UTC)
                # A timer that completes before the asyncio deadline would be yielded repeatedly.
                if deadline.timestamp() < when:
                    deadline += timedelta(microseconds=1)
                timer = _as_task(self.context.create_timer(deadline.replace(tzinfo=None)))
                self.timers[when] = timer
        if timer is not None:
            durable_tasks.append(timer)
        if not durable_tasks:
            raise UserError(
                'Agent code awaited non-durable work inside a Durable Task orchestration. '
                'Move I/O into a tool or a `@durable_operation` capability method.'
            )
        # Asyncio, not the SDK join, owns gather/race semantics and sibling cancellation.
        return _as_task(task.when_any(durable_tasks)) if len(durable_tasks) > 1 else durable_tasks[0]

    def deliver(self) -> None:
        for durable, future in self.pending:
            if not future.done() and durable.is_complete:
                if durable.is_failed:
                    future.set_exception(durable.get_exception())
                else:
                    future.set_result(durable.get_result())

    def drive(self, coroutine: Coroutine[object, object, T]) -> Generator[task.Task[object], object, T]:
        with self.activated():
            root = self.create_task(coroutine, name='durabletask-agent')
        try:
            while True:
                self.deliver()
                self.drain()
                if root.done():
                    return root.result()
                if self._ready:
                    continue
                waiting = self.next_task()
                try:
                    # durabletask 1.12.0 leaves suspended dispatch generators open, delaying cleanup.
                    # Upstream fixes: microsoft/durabletask-python#300 and microsoft/durabletask-python#303.
                    yield waiting
                except Exception:
                    # Feed the SDK task's failure back into the async task that owns the operation.
                    # Unrelated pending activities are left alone, not failed with the same error.
                    if not waiting.is_failed:
                        raise
                    self.deliver()
        finally:
            # Cancel and drain local async scopes without scheduling new durable work.
            self.abandoned = True
            with self.activated():
                tasks = asyncio.all_tasks(self)
                for pending in tasks:
                    pending.cancel()
            self.drain_cleanup()
            with self.activated():
                shutdown = self.create_task(self.shutdown_asyncgens(), name='durabletask-asyncgen-shutdown')
            self.drain_cleanup()
            tasks.update(asyncio.all_tasks(self))
            tasks.add(shutdown)
            unfinished = [pending for pending in tasks if not pending.done()]
            if unfinished:
                with self.activated():
                    for pending in unfinished:
                        pending.cancel()
                self.drain_cleanup()
            for pending in tasks:
                if pending.done() and not pending.cancelled():
                    pending.exception()
            self.close()
            if unfinished:
                raise UserError('Agent cleanup awaited durable work after its orchestration was abandoned.')


def run_durable(
    context: task.OrchestrationContext,
    body: Callable[[], Coroutine[object, object, T]],
) -> Generator[task.Task[object], object, T]:
    """Run an async body from a Durable Task generator orchestrator.

    Use `yield from run_durable(context, body)`. Attach `AzureDurableTaskDurability` to
    agents called by the body, and register their activities before starting the worker.
    Model requests and tool calls yield activities; parallel operations resume as each activity finishes.
    Asyncio timers use the orchestration's durable clock and timers.

    Orchestrator-side hooks, instruction functions, output validators and inline tools
    must be deterministic and perform no I/O. Dependencies must be JSON serializable;
    the capability rebuilds them from the agent's `deps_type` inside activities.
    """
    if asyncio._get_running_loop() is not None:
        raise UserError('Drive a Durable Task orchestrator from a synchronous worker, not a running event loop.')
    loop = _TrampolineLoop(context)
    return (yield from loop.drive(body()))
