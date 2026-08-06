from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

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
        events arrive live; model events are the real captured events replayed after each
        model-request activity, step, or task completes. For handler I/O that must run
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
            async for event in probe:
                yield event
            return

        # Observer: the probe is a coroutine we haven't awaited. Close it (nothing has
        # run yet) and re-invoke the handler with the teed receive stream.
        cast('Coroutine[Any, Any, None]', probe).close()

        observer = cast('EventStreamHandlerFunc[AgentDepsT]', self.handler)
        send_stream: MemoryObjectSendStream[AgentStreamEvent]
        receive_stream: MemoryObjectReceiveStream[AgentStreamEvent]
        send_stream, receive_stream = anyio.create_memory_object_stream()

        async def run_handler() -> None:
            async with receive_stream:
                await observer(ctx, receive_stream)

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
        # different tasks raises that same error. Today's upstream frames all open and close their
        # scopes within a single `__anext__`, so this holds -- it is a constraint on what may be
        # wrapped, not a latent bug.
        handler_task = asyncio.create_task(run_handler())
        next_task: asyncio.Task[AgentStreamEvent] | None = None
        stream_iterator = aiter(stream)
        try:
            async with send_stream:
                handler_alive = True

                async def pull_next() -> AgentStreamEvent:
                    return await anext(stream_iterator)

                while True:
                    next_task = asyncio.create_task(pull_next())
                    if handler_alive:
                        await asyncio.wait(
                            (next_task, handler_task),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if handler_task.done():
                            if not handler_task.cancelled() and handler_task.exception() is None:
                                handler_alive = False
                            else:
                                await _utils.cancel_and_drain(next_task)
                                try:
                                    await _utils.aclose_if_supported(stream_iterator)
                                finally:
                                    await handler_task

                    try:
                        event = await next_task
                    except StopAsyncIteration:
                        break

                    if handler_alive:
                        try:
                            await send_stream.send(event)
                        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                            # Handler bailed early; keep forwarding events downstream.
                            handler_alive = False
                    yield event
        except BaseException:
            # The consumer stopped early or the inner stream failed: tear the handler down rather
            # than leaving it parked on `receive`, and let the original exception propagate.
            # The in-flight pull goes with it. Being cancelled while awaiting a task doesn't cancel
            # that task, so it would otherwise advance the source one step past our exit and could
            # still be inside `anext()` when someone else closes that same iterator.
            await _utils.cancel_and_drain(handler_task, *filter(None, (next_task,)))
            await _utils.aclose_if_supported(stream_iterator)
            raise

        # Closing `send_stream` ends the handler's iteration; awaiting it surfaces anything it raised.
        await handler_task

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)
