from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, cast

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from pydantic_ai import _utils
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import (
        EventStreamHandler as EventStreamHandlerFunc,
        EventStreamProcessor as EventStreamProcessorFunc,
    )


@dataclass
class ProcessEventStream(AbstractCapability[AgentDepsT]):
    """A capability that forwards the agent's event stream to a user-provided async handler.

    The handler receives the stream of [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent]s
    emitted during classic model streaming and tool execution, or the shared and realtime-only
    events emitted by a realtime session. Two forms are supported:

    - An [`EventStreamHandler`][pydantic_ai.agent.EventStreamHandler] — an `async def`
      returning `None`. Events are forwarded to the handler while also being passed
      through unchanged to the rest of the capability chain, so multiple handlers (and
      the top-level `event_stream_handler` argument) can all see the same stream without
      changing each other's view. A handler that returns early stops receiving events
      but does not affect downstream consumers; a handler that raises propagates the
      exception to the rest of the run. Events are delivered synchronously, so a slow
      handler back-pressures the rest of the stream.
    - An `EventStreamProcessor` — an async
      generator yielding [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent]s.
      The events it yields replace the inner stream for downstream wrappers and consumers,
      so it can modify, drop, or add events.

      This replacement is global, not a private view for event-stream handlers: the run has one
      event stream and a processor shapes all of it. Dropping or rewriting a
      [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] therefore also changes what
      [`stream_text()`][pydantic_ai.result.StreamedRunResult.stream_text] yields to a
      `run_stream()` caller.

      Some events are also control signals:
      [`FinalResultEvent`][pydantic_ai.messages.FinalResultEvent] is what tells
      [`agent.run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream] that the final output has
      started, so dropping it makes `run_stream()` wait for the whole model response before handing
      back the result instead of streaming it. Filter deliberately.

      None of this changes the run's output: the
      [`ModelResponse`][pydantic_ai.messages.ModelResponse] is accumulated from the raw model
      stream before a processor sees the events, so
      [`stream_output()`][pydantic_ai.result.StreamedRunResult.stream_output] and the final
      validated output are unaffected (dropping events can only change when a partial snapshot is
      emitted, not its content). Use the observer form if you only want to watch events.

      In a realtime session, this is likewise only a consumer-facing view. Transforming or dropping
      events does not affect session history or tool execution.

    Async generator processors are closed and observer handlers are awaited during stream teardown.
    Release resources in a `finally` block, using a shielded AnyIO cancel scope for asynchronous
    cleanup that must finish under cancellation. When the stream ends early (the consumer stops, an
    error is raised, or the run is cancelled), async work an observer is in the middle of can be
    interrupted even inside a shielded scope, such as cleanup after it stopped iterating. For
    cleanup to finish, keep iterating to the end of the stream and skip the events you don't need
    instead of breaking out of the loop, then run the shielded cleanup in `finally`. A shielded
    write in the loop can still be interrupted.

    When this capability is registered, `agent.run()` and
    [`AgentRun.next()`][pydantic_ai.run.AgentRun.next] automatically enable streaming so the
    handler fires without requiring an explicit `event_stream_handler` argument. The handler
    sees the same events however the run is driven, including under
    [`agent.iter()`][pydantic_ai.agent.Agent.iter] and when you stream a node yourself with
    `node.stream()`.

    !!! note "Durable execution"

        Under the durable-execution capabilities
        ([`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability],
        [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability],
        [`PrefectDurability`][pydantic_ai.durable_exec.prefect.PrefectDurability]),
        this capability's handler always runs in workflow or flow code and must be
        deterministic because it re-runs on workflow replay. Tool-call and final-output
        events arrive live; model events are replayed from the final response after each
        model-request activity, step, or task completes, one `PartStartEvent` per part
        rather than the streamed deltas. For handler I/O that must run
        exactly once inside a durable boundary, pass `event_stream_handler=` to the
        durability capability instead.
    """

    handler: EventStreamHandlerFunc[AgentDepsT] | EventStreamProcessorFunc[AgentDepsT]

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        # Probe the handler: the processor form returns an AsyncIterator directly, while
        # the observer form returns an awaitable. Introspecting the return is robust for
        # both plain functions and callable instances, unlike `inspect.isasyncgenfunction`.
        probe = self.handler(ctx, stream)
        if isinstance(probe, AsyncIterator):
            try:
                async for event in probe:
                    yield event
            finally:
                with anyio.CancelScope(shield=True):
                    await _utils.aclose_if_supported(probe)
            return

        # Observer: the probe is a coroutine we haven't awaited. Close it (nothing has
        # run yet) and re-invoke the handler with the teed receive stream.
        cast('Coroutine[Any, Any, None]', probe).close()

        # The handler runs in a plain `asyncio` task rather than an `anyio` task group held open
        # across the `yield`s below. A task group is bound to the task that entered it, which would
        # make this generator bound to that task too -- and the node stream it wraps is memoized, so
        # it can legitimately be resumed elsewhere (a `StreamedRunResult` consumed in another task,
        # or `CallToolsNode.run()` finalizing a stream whose consumer bailed out). Exiting the group
        # from a different task raises anyio's "cancel scope in a different task" error, replacing
        # whatever the caller was actually doing. A task has no such affinity.
        #
        # The flip side is that each pull below runs in a fresh task, so the stream being wrapped is
        # resumed from a different task on every event. Nothing upstream may hold an `anyio` cancel
        # scope or task group open across one of its `yield`s: entering and exiting it from
        # different tasks raises that same error.
        tee = _ObserverTee(cast('EventStreamHandlerFunc[AgentDepsT]', self.handler), ctx, stream)
        try:
            async with tee.send_stream:
                try:
                    while (event := await tee.pull()) is not None:
                        await tee.send(event)
                        yield event
                        tee.next_scope = anyio.CancelScope()
                except BaseException:
                    # Cancel the handler natively, once, before `send_stream` closes, so it sees a
                    # cancellation rather than a normal end of stream. A native cancel is edge-triggered,
                    # so the handler's `finally` can still await cleanup without shielding it. It also
                    # interrupts shielded work the handler is in the middle of, such as cleanup after it
                    # stopped iterating, so the docs tell handlers to keep iterating and run shielded cleanup
                    # in `finally`. Cancelling through `handler_scope` whenever the handler is outside
                    # `receive()` would protect that work, but leave a handler interrupted mid-loop unable to
                    # finish an unshielded `finally`. Deferring the native cancel to the handler's next
                    # `receive()` would hang teardown on a handler stuck mid-loop.
                    tee.handler_task.cancel()
                    tee.handler_cancel_requested = True
                    raise

            # A plain `await handler_task` would forward a cancel of this task into the handler as a
            # native cancel, which interrupts even its shielded cleanup. `handler_scope` below owns that.
            # `asyncio.shield` would report the handler's later exception to the loop, as above.
            await asyncio.wait((tee.handler_task,))
            tee.handler_task.result()
        except BaseException:
            # The consumer stopped early or the inner stream failed: tear the handler down rather
            # than leaving it parked on `receive`, and let the original exception propagate.
            # Unless the handler was already cancelled natively above, cancel it through `handler_scope`,
            # and the pull through `next_scope`: a scope cancel lets their shielded cleanup finish.
            # `_utils.cancel_and_drain` cancels natively and would interrupt that cleanup.
            with anyio.CancelScope(shield=True):
                if not tee.handler_cancel_requested:
                    tee.handler_scope.cancel()
                tee.next_scope.cancel()
                await asyncio.gather(tee.handler_task, *filter(None, (tee.next_task,)), return_exceptions=True)
                await _utils.aclose_if_supported(tee.stream_iterator)
            raise

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)

    @property
    def _emits_app_events(self) -> bool:
        return True


class _ObserverTee(Generic[AgentDepsT]):
    """The state `ProcessEventStream.wrap_run_event_stream` keeps across the events it tees to an observer handler.

    Creating it starts the handler task.
    """

    def __init__(
        self,
        observer: EventStreamHandlerFunc[AgentDepsT],
        ctx: RunContext[AgentDepsT],
        stream: AsyncIterable[AgentStreamEvent],
    ) -> None:
        self.send_stream: MemoryObjectSendStream[AgentStreamEvent]
        receive_stream: MemoryObjectReceiveStream[AgentStreamEvent]
        self.send_stream, receive_stream = anyio.create_memory_object_stream()
        self.handler_scope = anyio.CancelScope()
        self.handler_task = asyncio.create_task(_run_handler(self.handler_scope, observer, ctx, receive_stream))
        self.handler_cancel_requested = False
        self.handler_alive = True
        self.next_task: asyncio.Task[AgentStreamEvent] | None = None
        self.next_scope = anyio.CancelScope()
        self.stream_iterator = aiter(stream)

    async def pull(self) -> AgentStreamEvent | None:
        """Pull the next event in a fresh task, or `None` at the end of the stream.

        While the handler is alive, the pull races it: a handler that fails or is cancelled ends the stream
        with its error.
        """
        self.next_task = next_task = asyncio.create_task(_pull_next(self.stream_iterator, self.next_scope))
        handler_task = self.handler_task
        if self.handler_alive:
            await asyncio.wait(
                (next_task, handler_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if handler_task.done():
                if not handler_task.cancelled() and handler_task.exception() is None:
                    self.handler_alive = False
                else:
                    self.next_scope.cancel()
                    with anyio.CancelScope(shield=True):
                        await asyncio.gather(next_task, return_exceptions=True)
                    try:
                        await _utils.aclose_if_supported(self.stream_iterator)
                    finally:
                        await handler_task

        # A plain await would forward a cancel of this task into the pull natively, and
        # `asyncio.shield` reports the pull's later exception to the loop on CPython 3.14 and
        # 3.15 releases without the fix for https://github.com/python/cpython/issues/156321.
        # Teardown cancels the pull through `next_scope`, which its shielded cleanup survives.
        # `asyncio.wait` always suspends, so skip it once the pull is done.
        if not next_task.done():
            await asyncio.wait((next_task,))
        try:
            return next_task.result()
        except StopAsyncIteration:
            return None

    async def send(self, event: AgentStreamEvent) -> None:
        """Send `event` to the handler while it is still receiving."""
        if self.handler_alive:
            try:
                await self.send_stream.send(event)
            except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                # Handler bailed early; keep forwarding events downstream.
                self.handler_alive = False


async def _run_handler(
    handler_scope: anyio.CancelScope,
    observer: EventStreamHandlerFunc[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    receive_stream: MemoryObjectReceiveStream[AgentStreamEvent],
) -> None:
    with handler_scope:
        async with receive_stream:
            await observer(ctx, receive_stream)


async def _pull_next(stream_iterator: AsyncIterator[AgentStreamEvent], scope: anyio.CancelScope) -> AgentStreamEvent:
    with scope:
        return await anext(stream_iterator)
    # Reached only when teardown cancelled `scope`, which swallows that cancellation.
    raise StopAsyncIteration
