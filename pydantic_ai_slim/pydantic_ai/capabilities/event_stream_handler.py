from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability, CapabilityOrdering

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import EventStreamHandler as EventStreamHandlerFunc


@dataclass
class HandleEventStream(AbstractCapability[AgentDepsT]):
    """A capability that consumes the agent's event stream via a user-provided async handler.

    The handler receives the stream of [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent]s
    emitted during model streaming and tool execution for each
    [`ModelRequestNode`][pydantic_ai._agent_graph.ModelRequestNode] and
    [`CallToolsNode`][pydantic_ai._agent_graph.CallToolsNode].

    When this capability is registered, [`agent.run()`][pydantic_ai.Agent.run] automatically
    enables streaming so the handler fires without requiring an explicit `event_stream_handler`
    argument. The capability is ordered `'outermost'` so the handler sees events after all
    other [`wrap_run_event_stream`][pydantic_ai.capabilities.AbstractCapability.wrap_run_event_stream]
    wrappers have transformed the stream.
    """

    handler: EventStreamHandlerFunc[AgentDepsT]

    def get_ordering(self) -> CapabilityOrdering | None:
        return CapabilityOrdering(position='outermost')

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        await self.handler(ctx, stream)
        return
        yield  # pragma: no cover

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)
