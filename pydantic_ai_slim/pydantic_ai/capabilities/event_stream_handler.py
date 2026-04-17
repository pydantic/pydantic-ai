from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import anyio

from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import EventStreamHandler as EventStreamHandlerFunc


@dataclass
class HandleEventStream(AbstractCapability[AgentDepsT]):
    """A capability that forwards the agent's event stream to a user-provided async handler.

    The handler receives the stream of [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent]s
    emitted during model streaming and tool execution for each `ModelRequestNode` and
    `CallToolsNode`. The handler can be either of the two forms supported by
    [`EventStreamHandler`][pydantic_ai.agent.EventStreamHandler]:

    - **Observer**: an `async def` returning `None`. Events are forwarded to the handler
      while also being passed through unchanged to the rest of the capability chain, so
      multiple observers (and the top-level `event_stream_handler` argument) can all see
      the same stream without interfering.
    - **Transformer**: an async generator yielding [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent]s.
      The events it yields replace the inner stream for downstream wrappers and consumers,
      so it can modify, drop, or add events.

    When this capability is registered, [`agent.run()`][pydantic_ai.Agent.run] automatically
    enables streaming so the handler fires without requiring an explicit `event_stream_handler`
    argument.
    """

    handler: EventStreamHandlerFunc[AgentDepsT]

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        if inspect.isasyncgenfunction(self.handler):
            async for event in self.handler(ctx, stream):
                yield event
            return

        observer = cast(
            'Callable[[RunContext[AgentDepsT], AsyncIterable[AgentStreamEvent]], Awaitable[None]]',
            self.handler,
        )
        send_stream, receive_stream = anyio.create_memory_object_stream[AgentStreamEvent]()

        async def run_handler() -> None:
            async with receive_stream:
                await observer(ctx, receive_stream)

        async with anyio.create_task_group() as tg:
            tg.start_soon(run_handler)
            async with send_stream:
                async for event in stream:
                    await send_stream.send(event)
                    yield event

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)
