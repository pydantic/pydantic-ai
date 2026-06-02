from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import anyio

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
    emitted during model streaming and tool execution for each `ModelRequestNode` and
    `CallToolsNode`. Two forms are supported:

    - An [`EventStreamHandler`][pydantic_ai.agent.EventStreamHandler] — an `async def`
      returning `None`. Events are forwarded to the handler while also being passed
      through unchanged to the rest of the capability chain, so multiple handlers (and
      the top-level `event_stream_handler` argument) can all see the same stream without
      changing each other's view. A handler that returns early stops receiving events
      but does not affect downstream consumers; a handler that raises propagates the
      exception to the rest of the run. Events are delivered synchronously, so a slow
      handler back-pressures the rest of the stream.
    - An [`EventStreamProcessor`][pydantic_ai.agent.EventStreamProcessor] — an async
      generator yielding [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent]s.
      The events it yields replace the inner stream for downstream wrappers and consumers,
      so it can modify, drop, or add events.

    When this capability is registered, [`agent.run()`][pydantic_ai.Agent.run] automatically
    enables streaming so the handler fires without requiring an explicit `event_stream_handler`
    argument.

    !!! note "Durable execution"

        Under the current durable-execution integrations
        ([Temporal][pydantic_ai.durable_exec.temporal.TemporalAgent],
        [DBOS][pydantic_ai.durable_exec.dbos.DBOSAgent],
        [Prefect][pydantic_ai.durable_exec.prefect.PrefectAgent]), model streaming happens
        inside an activity/step rather than in the outer agent loop. This capability's
        `wrap_run_event_stream` hook fires for tool-call events and the final post-streaming
        batch, but it does **not** see individual model-response events live — the underlying
        durable model consumes those inside the activity before returning. The in-flight
        `event_stream_handler` parameter does still observe the live events; a future
        refactor threading the capability chain through the activity boundary is being
        explored in [#4977](https://github.com/pydantic/pydantic-ai/pull/4977).
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
        send_stream, receive_stream = anyio.create_memory_object_stream[AgentStreamEvent]()

        async def run_handler() -> None:
            async with receive_stream:
                await observer(ctx, receive_stream)

        async with anyio.create_task_group() as tg:
            tg.start_soon(run_handler)
            async with send_stream:
                handler_alive = True
                async for event in stream:
                    if handler_alive:
                        try:
                            await send_stream.send(event)
                        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                            # Handler bailed early; keep forwarding events downstream.
                            handler_alive = False
                    yield event

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)
