"""Tracking work that outlives the step that started it, through [`RunContext.in_flight`][pydantic_ai.tools.RunContext.in_flight].

See [Work that outlives a step](../capabilities/custom.md#in-flight-work).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias

import anyio
from typing_extensions import TypeVar

from pydantic_graph import End

from . import _durable_replay, _utils
from ._enqueue import run_queue
from .exceptions import UserError
from .messages import AgentStreamEvent, CapabilityEvent
from .output import PendingModelResponse
from .tools import DeferredToolRequests

if TYPE_CHECKING:
    from . import _agent_graph
    from ._run_context import RunContext
    from .result import FinalResult

__all__ = (
    'InFlightOnEnd',
    'InFlightWorkState',
    'InFlightWorkPhase',
    'InFlightWork',
    'InFlightWorkSnapshot',
    'InFlightWorkEvent',
    'InFlightRegistry',
)

T = TypeVar('T', default=Any)

InFlightOnEnd: TypeAlias = Literal['drain', 'cancel', 'detach']
"""What happens to live work when the run ends.

- `'drain'`: for work whose result belongs to this run. The run waits for it: when the run ends with output,
    until content has been queued or all draining work has finished, so the model gets another turn on what
    was queued; when the run pauses with [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] or a
    [`PendingModelResponse`][pydantic_ai.output.PendingModelResponse], until all of it has.
- `'cancel'`: for best-effort work that is worthless once the run is over. The run cancels it when it ends.
- `'detach'`: for work that lives in a resource that outlives the run, like a shell job, a process in a
    workspace that is kept, or a task an application-owned object runs. The run lists it while it lives, but
    never waits for it or cancels it.

A run that ends with an error, is cancelled, times out or hits a usage limit cancels all work except `'detach'`
work.
"""

InFlightWorkState: TypeAlias = Literal['running', 'succeeded', 'failed', 'cancelled']
"""Where a piece of in-flight work is in its life."""

InFlightWorkPhase: TypeAlias = Literal['started', 'settled', 'draining', 'drained']
"""What an [`InFlightWorkEvent`][pydantic_ai.in_flight.InFlightWorkEvent] reports.

- `'started'`: a piece of work was started or registered.
- `'settled'`: a piece of work finished, failed or was cancelled.
- `'draining'`: the run is about to end and is waiting for its `'drain'` work.
- `'drained'`: the run stopped waiting.
"""


@dataclass(frozen=True, kw_only=True)
class InFlightWorkSnapshot:
    """The state of a piece of in-flight work at the time an [`InFlightWorkEvent`][pydantic_ai.in_flight.InFlightWorkEvent] was emitted."""

    id: str
    """The work's ID, unique among the run's live work."""
    kind: str
    """The kind of work, such as `'background_tool'`."""
    name: str
    """A name for the work, for display."""
    on_end: InFlightOnEnd
    """What happens to the work when the run ends."""
    durable: bool
    """Whether a durable execution engine records the work's result on its own."""
    state: InFlightWorkState
    """Where the work was in its life."""


@dataclass(repr=False, kw_only=True)
class InFlightWorkEvent(CapabilityEvent, namespace='in_flight', name='work'):
    """Reports in-flight work starting and settling, and the run waiting for it before it ends.

    A UI can show that the agent is still working from `live`, without knowing which capability started the
    work. Emitted by the run itself, so its `capability_id` is `None`.
    """

    phase: InFlightWorkPhase
    """What happened."""
    work: tuple[InFlightWorkSnapshot, ...]
    """The work it happened to: the piece that started or settled, or the `'drain'` work the run waits for.

    For `'draining'`, the work the run starts waiting for; for `'drained'`, all the work it waited for, including
    draining work started while it waited.
    """
    live: int
    """How many pieces of work were live afterwards, of any `on_end`."""


class InFlightWork(Generic[T]):
    """A handle on a piece of work that outlives the step that started it.

    Returned by [`InFlightRegistry.start()`][pydantic_ai.in_flight.InFlightRegistry.start] and
    [`InFlightRegistry.register()`][pydantic_ai.in_flight.InFlightRegistry.register]; don't create one yourself.
    """

    def __init__(
        self,
        *,
        id: str,
        kind: str,
        name: str,
        on_end: InFlightOnEnd,
        durable: bool,
        task: asyncio.Task[T] | None,
        cancel: Callable[[], None] | None,
        on_settled: Callable[[InFlightWork[Any], BaseException | None], None],
    ) -> None:
        self.id = id
        """The work's ID, unique among the run's live work: usually the tool call or task ID it belongs to."""
        self.kind = kind
        """The kind of work, such as `'background_tool'`, to tell different producers' work apart."""
        self.name = name
        """A name for the work, for display."""
        self.on_end: InFlightOnEnd = on_end
        """What happens to the work when the run ends."""
        self.durable = durable
        """Whether a durable execution engine records the work's result on its own, as an activity, step or task.

        Recorded on the handle and in events; the run doesn't act on it yet.
        """
        self._state: InFlightWorkState = 'running'
        self._task = task
        self._cancel = cancel
        self._on_settled = on_settled
        self._done = asyncio.Event()
        self._value: _utils.Option[T] = None
        self._error: BaseException | None = None
        if task is not None:
            task.add_done_callback(self._task_done)

    @property
    def state(self) -> InFlightWorkState:
        """Where the work is in its life."""
        return self._state

    async def wait(self) -> None:
        """Wait until the work has settled: finished, failed or been cancelled. Never raises the work's error.

        Work started with [`start()`][pydantic_ai.in_flight.InFlightRegistry.start] that waits for itself never
        returns.
        """
        await self._done.wait()

    async def result(self) -> T:
        """Wait until the work has settled, and return its result.

        Work that was [registered][pydantic_ai.in_flight.InFlightRegistry.register] has no result, so this
        returns `None` for it.

        Raises:
            BaseException: The error the work failed with.
            asyncio.CancelledError: If the work was cancelled, as awaiting a cancelled task does.
        """
        await self._done.wait()
        if self._value is not None:
            return self._value.value
        if self._error is not None:
            raise self._error
        raise asyncio.CancelledError(f'In-flight work {self.id!r} was cancelled.')

    def cancel(self) -> None:
        """Cancel the work, if it's still running.

        Work run by [`start()`][pydantic_ai.in_flight.InFlightRegistry.start] has its task cancelled, and
        settles once the task has stopped. Registered work has the `cancel` callback it was registered with
        called, and settles as cancelled right away, even if the callback raises.
        """
        if self._state != 'running':
            return
        if self._task is not None:
            self._task.cancel()
            return
        try:
            if self._cancel is not None:
                self._cancel()
        finally:
            self._finish('cancelled')

    def _finish(
        self, state: InFlightWorkState, *, value: _utils.Option[T] = None, error: BaseException | None = None
    ) -> None:
        if self._state != 'running':
            return
        self._state = state
        self._value = value
        self._error = error
        self._task = None
        self._done.set()
        self._on_settled(self, error)

    def _task_done(self, task: asyncio.Task[T]) -> None:
        if task.cancelled():
            # Reading the outcome releases the `CancelledError` the task keeps, whose traceback holds its frames.
            with suppress(asyncio.CancelledError):
                task.exception()
            self._finish('cancelled')
        elif (error := task.exception()) is not None:
            self._finish('failed', error=error)
        else:
            self._finish('succeeded', value=_utils.Some(task.result()))


async def _call(fn: Callable[[], Awaitable[T]]) -> T:
    return await fn()


def _check_on_loop(method: str, agent_loop: asyncio.AbstractEventLoop | None) -> asyncio.AbstractEventLoop:
    """`agent_loop`, the loop running the agent, if it's running this call: the registry isn't safe to touch from elsewhere."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            anyio.get_current_task()
        except RuntimeError:
            pass
        else:
            # Running on another async backend, which the registry's tasks and events don't support.
            raise UserError(
                f'`ctx.in_flight.{method}()` requires the asyncio event loop and is not supported under other async '
                'backends, such as Trio.'
            ) from None
    else:
        if loop is agent_loop:
            return loop
    # A thread without an event loop, or another event loop, like one a sync tool started in its worker thread.
    raise UserError(
        f'`ctx.in_flight.{method}()` must be called from the event loop running the agent, such as from an async '
        'tool or a capability hook.'
    )


def _snapshot(work: InFlightWork[Any]) -> InFlightWorkSnapshot:
    return InFlightWorkSnapshot(
        id=work.id, kind=work.kind, name=work.name, on_end=work.on_end, durable=work.durable, state=work.state
    )


class InFlightRegistry:
    """The run's registry of work that outlives the step that started it, available as [`ctx.in_flight`][pydantic_ai.tools.RunContext.in_flight].

    Work goes in either way:

    - [`start()`][pydantic_ai.in_flight.InFlightRegistry.start] runs a coroutine as a task the run owns.
    - [`register()`][pydantic_ai.in_flight.InFlightRegistry.register] tracks work something else runs, like a
        process, until you [`settle()`][pydantic_ai.in_flight.InFlightRegistry.settle] it. By default the run
        only lists that work, since it usually outlives the run.

    Each run has its own registry, which a run started from inside it doesn't share. Call its methods from the
    event loop running the agent, such as from an async tool or a capability hook.
    """

    def __init__(self, event_stream_buffer: list[AgentStreamEvent]) -> None:
        """Created by the run; don't create one yourself."""
        self._event_stream_buffer = event_stream_buffer
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            # Not on asyncio, which runs need, so every call raises.
            self._loop = None
        self._live: dict[str, InFlightWork[Any]] = {}
        self._started: set[InFlightWork[Any]] = set()
        self._failures: list[BaseException] = []
        self._changed = asyncio.Event()
        self._closed = False
        self._waited: dict[InFlightWork[Any], None] | None = None
        """While the run waits to end, the draining work it waited for, including work started during the wait."""

    def start(
        self,
        fn: Callable[[], Awaitable[T]],
        /,
        *,
        kind: str,
        name: str,
        id: str,
        on_end: Literal['drain', 'cancel'] = 'drain',
        durable: bool = False,
    ) -> InFlightWork[T]:
        """Run `fn()` as a task the run owns, and track it.

        `fn` delivers its own results, usually by calling [`ctx.enqueue()`][pydantic_ai.tools.RunContext.enqueue].
        An error `fn` raises ends the run once the step that's running when it fails has finished, or while the
        run waits for the work to end, so catch the errors the model should hear about and enqueue them instead.
        The run's node error hooks don't see the error, and an error raised after the run's last step is not
        raised at all. If several pieces of work fail, only the first error is raised.

        Args:
            fn: The work to run, called with no arguments.
            kind: The kind of work, such as `'background_tool'`, to tell different producers' work apart.
            name: A name for the work, for display.
            id: The work's ID, unique among the run's live work. Make it deterministic, such as the ID of the
                tool call that starts the work, so the same work gets the same ID when a run is replayed.
            on_end: What happens to the work when the run ends: `'drain'` waits for it, `'cancel'` cancels it.
                Work that should outlive the run isn't the run's to own: run it yourself and
                [`register()`][pydantic_ai.in_flight.InFlightRegistry.register] it, which detaches it by default.
            durable: Whether a durable execution engine records the work's result on its own, as an activity,
                step or task. The run doesn't act on it yet.

        Returns:
            A handle on the work.

        Raises:
            UserError: If the run has ended, live work already has this `id`, `on_end` is `'detach'`, or this
                isn't called from the event loop running the agent.
        """
        if on_end not in ('drain', 'cancel'):
            raise UserError(
                f'`ctx.in_flight.start()` takes `on_end` of `drain` or `cancel`, not {on_end!r}: the run owns the work '
                'it starts. Run work that should outlive the run yourself, and `register()` it.'
            )
        self._check_can_add(id, 'start')
        task = _check_on_loop('start', self._loop).create_task(_call(fn), name=f'in_flight {kind} {id}')
        work = InFlightWork[T](
            id=id,
            kind=kind,
            name=name,
            on_end=on_end,
            durable=durable,
            task=task,
            cancel=None,
            on_settled=self._settled,
        )
        self._started.add(work)
        self._add(work)
        return work

    def register(
        self,
        *,
        kind: str,
        name: str,
        id: str,
        on_end: InFlightOnEnd = 'detach',
        durable: bool = False,
        cancel: Callable[[], None] | None = None,
    ) -> InFlightWork[None]:
        """Track work that something else runs, such as a process or a task another object owns.

        Call [`settle()`][pydantic_ai.in_flight.InFlightRegistry.settle] when it has finished. The work delivers
        its own results, as with [`start()`][pydantic_ai.in_flight.InFlightRegistry.start], but an error it
        settles with doesn't end the run.

        Work that something else owns usually lives in a resource that outlives the run, so by default the run
        only lists it. Pass `on_end='drain'` when its result belongs to this run, or `on_end='cancel'` when it's
        worthless once the run is over.

        Args:
            kind: The kind of work, such as `'shell_job'`, to tell different producers' work apart.
            name: A name for the work, for display.
            id: The work's ID, unique among the run's live work. Make it deterministic, such as the ID of the
                tool call that starts the work.
            on_end: What happens to the work when the run ends: by default nothing, as it's `'detach'`. See
                [`InFlightOnEnd`][pydantic_ai.in_flight.InFlightOnEnd].
            durable: Whether a durable execution engine records the work's result on its own, as an activity,
                step or task. The run doesn't act on it yet.
            cancel: Stops the work, when the run or a call to
                [`InFlightWork.cancel()`][pydantic_ai.in_flight.InFlightWork.cancel] cancels it. Without one,
                cancelling the work only stops tracking it. If it raises when the run cancels the work, the run
                still cancels its other work, then raises the error, unless the run is failing or being cancelled
                anyway, in which case the caller gets that instead.

        Returns:
            A handle on the work.

        Raises:
            UserError: If the run has ended, live work already has this `id`, or this isn't called from the event
                loop running the agent.
        """
        self._check_can_add(id, 'register')
        _check_on_loop('register', self._loop)
        work = InFlightWork[None](
            id=id,
            kind=kind,
            name=name,
            on_end=on_end,
            durable=durable,
            task=None,
            cancel=cancel,
            on_settled=self._settled,
        )
        self._add(work)
        return work

    def settle(self, work: InFlightWork[None], *, error: BaseException | None = None) -> None:
        """Record that [registered][pydantic_ai.in_flight.InFlightRegistry.register] work has finished.

        Settling work that has already settled, for example because it was cancelled, does nothing.

        Args:
            work: The handle `register()` returned.
            error: The error the work failed with, if it failed.

        Raises:
            UserError: If `work` was started with `start()`, which settles it when its coroutine finishes, or this
                isn't called from the event loop running the agent. From another thread, schedule the call on that
                loop: capture it with `asyncio.get_running_loop()` where you registered the work, and call
                `loop.call_soon_threadsafe(ctx.in_flight.settle, work)`.
        """
        _check_on_loop('settle', self._loop)
        if work in self._started:
            raise UserError(
                f'In-flight work {work.id!r} was started with `start()`, so it settles when its coroutine finishes.'
            )
        if error is None:
            work._finish('succeeded', value=_utils.Some(None))  # pyright: ignore[reportPrivateUsage]
        else:
            work._finish('failed', error=error)  # pyright: ignore[reportPrivateUsage]

    def live(self, *, kind: str | None = None) -> tuple[InFlightWork[Any], ...]:
        """The work that hasn't settled yet, of every `on_end`, oldest first.

        Args:
            kind: Only list work of this kind.

        Raises:
            UserError: If this isn't called from the event loop running the agent, which changes the live work.
        """
        _check_on_loop('live', self._loop)
        return tuple(work for work in self._live.values() if kind is None or work.kind == kind)

    async def wait_idle(self, *, kind: str | None = None) -> None:
        """Wait until all live work, except `'detach'` work, has settled. Never raises the work's errors.

        Called from work started with [`start()`][pydantic_ai.in_flight.InFlightRegistry.start] that this would
        wait for, it waits for itself and never returns: pass a `kind` that excludes it.

        Args:
            kind: Only wait for work of this kind.

        Raises:
            UserError: If this isn't called from the event loop running the agent.
        """
        _check_on_loop('wait_idle', self._loop)
        while any(work.on_end != 'detach' for work in self.live(kind=kind)):
            await self._changed.wait()

    def _check_can_add(self, id: str, method: str) -> None:
        if self._closed:
            raise UserError(f'`ctx.in_flight.{method}()` is not available because the agent run has ended.')
        if id in self._live:
            raise UserError(
                f'In-flight work {id!r} is already running in this run; give each piece of work its own `id`.'
            )

    def _add(self, work: InFlightWork[Any]) -> None:
        self._live[work.id] = work
        if self._waited is not None and work.on_end == 'drain':
            self._waited[work] = None
        self._emit('started', (work,))

    def _settled(self, work: InFlightWork[Any], error: BaseException | None) -> None:
        del self._live[work.id]
        if work in self._started:
            self._started.discard(work)
            if error is not None:
                self._failures.append(error)
        self._emit('settled', (work,))
        self._notify()

    def _notify_from(self, loop: asyncio.AbstractEventLoop) -> None:
        """`_notify` on `loop`, from wherever content was queued, as `enqueue` may be called from another thread."""
        try:
            on_loop = asyncio.get_running_loop() is loop
        except RuntimeError:
            # A thread with no event loop running, like a sync tool's worker thread.
            on_loop = False
        if on_loop:
            # Directly, since not every event loop has `call_soon_threadsafe`, and one that runs on a pool of
            # threads, like Temporal's workflow loop, can't be recognized by its thread.
            self._notify()
        else:
            loop.call_soon_threadsafe(self._notify)

    def _notify(self) -> None:
        """Wake whoever waits for work to settle or content to be queued, so they check again."""
        self._changed.set()
        self._changed = asyncio.Event()

    def _emit(self, phase: InFlightWorkPhase, works: Sequence[InFlightWork[Any]]) -> None:
        # Nothing streams events once the run has ended.
        if not self._closed:
            self._event_stream_buffer.append(
                InFlightWorkEvent(phase=phase, work=tuple(_snapshot(work) for work in works), live=len(self._live))
            )

    def _raise_failure(self) -> None:
        if self._failures:
            raise self._failures.pop(0)

    def _draining(self) -> list[InFlightWork[Any]]:
        return [work for work in self._live.values() if work.on_end == 'drain']

    async def _after_step(
        self,
        ctx: RunContext[Any],
        result: _agent_graph.AgentNode[Any, Any] | End[FinalResult[Any]],
    ) -> _agent_graph.AgentNode[Any, Any] | End[FinalResult[Any]]:
        """Raise the error of failed work, and apply the end rule when the step ends the run.

        Runs after every capability hook of the step and before the pending messages are drained, so the drain
        handles content the work queued while the run waited, as it does any content queued when a run ends.
        """
        self._raise_failure()
        if not isinstance(result, End) or not (waiting := self._draining()):
            return result
        # A pause hands control back to the caller without another model request, so no later step would take
        # a result that arrives after it. Ending with output, the model gets another turn as soon as content has
        # been queued, which work that keeps running after reporting may do. Work that settles without queuing
        # anything only wakes the wait: ending then would cancel the draining work that's still running.
        pause = isinstance(result.data.output, (DeferredToolRequests, PendingModelResponse))
        queue = run_queue(ctx)
        if not pause and queue:
            return result
        loop = asyncio.get_running_loop()
        with (
            _durable_replay.start_as_current_span(
                ctx.tracer,
                'in_flight drain',
                attributes={'in_flight.waiting': len(waiting), 'in_flight.until': 'all' if pause else 'queued'},
            ) as span,
            queue.notifying(lambda: self._notify_from(loop)),
        ):
            self._emit('draining', waiting)
            self._waited = waited = dict.fromkeys(waiting)
            try:
                while self._draining() and (pause or not queue):
                    await self._changed.wait()
                    self._raise_failure()
            finally:
                self._waited = None
            span.set_attribute('in_flight.waited', len(waited))
            self._emit('drained', tuple(waited))
        return result

    async def _close_on_exit(self, exc_type: type[BaseException] | None, *_: object) -> None:
        """Close the registry as the run exits, raising a `cancel` callback's error only if the run itself didn't fail.

        A failure of the run, or its cancellation, stays what the caller gets, so a cancellation is still classified as
        one.
        """
        await self._aclose(raise_callback_errors=exc_type is None)

    async def _aclose(self, *, raise_callback_errors: bool) -> None:
        """Cancel the work that doesn't outlive the run, and wait for it to stop."""
        self._closed = True
        ending = [work for work in self._live.values() if work.on_end != 'detach']
        if not ending:
            return
        # A failing `cancel` callback mustn't keep the rest of the work running: it's raised once all of it stopped.
        errors: list[Exception] = []
        for work in ending:
            try:
                work.cancel()
            except Exception as error:
                errors.append(error)
        # Teardown may already be under cancellation, which must not cut the wait for the work to stop.
        with anyio.CancelScope(shield=True):
            for work in ending:
                await work.wait()
        if errors and raise_callback_errors:
            raise errors[0]
