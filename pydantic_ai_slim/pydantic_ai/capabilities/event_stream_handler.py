from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    `CallToolsNode`. Events are also passed through to the rest of the capability chain,
    so multiple `HandleEventStream` capabilities (and the top-level `event_stream_handler`
    argument) can observe the same stream without interfering with each other.

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
        send_stream, receive_stream = anyio.create_memory_object_stream[AgentStreamEvent]()

        async def run_handler() -> None:
            async with receive_stream:
                await self.handler(ctx, receive_stream)

        async with anyio.create_task_group() as tg:
            tg.start_soon(run_handler)
            async with send_stream:
                async for event in stream:
                    await send_stream.send(event)
                    yield event

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)
