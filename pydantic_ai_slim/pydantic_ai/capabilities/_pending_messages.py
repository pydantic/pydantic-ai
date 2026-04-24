"""Auto-injected capability that drains the pending message queue at appropriate times."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import ModelRequest, PendingMessage, PendingMessagePriority
from pydantic_ai.tools import RunContext

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.result import FinalResult
    from pydantic_graph import End


def _drain_by_priority(
    queue: list[PendingMessage],
    priority: PendingMessagePriority,
) -> list[PendingMessage]:
    """Remove and return all messages with the given priority from the queue."""
    drained: list[PendingMessage] = []
    remaining: list[PendingMessage] = []
    for msg in queue:
        if msg.priority == priority:
            drained.append(msg)
        else:
            remaining.append(msg)
    queue[:] = remaining
    return drained


class PendingMessageDrainCapability(AbstractCapability[Any]):
    """Drains the pending message queue at appropriate times.

    - Steering messages are injected before each model request.
    - Follow-up messages are injected when the agent would otherwise end,
      redirecting to a new ModelRequestNode to continue the conversation.

    This capability is always auto-injected and placed outermost via
    [`CapabilityOrdering`][pydantic_ai.capabilities.abstract.CapabilityOrdering]
    so it wraps around other capabilities. This ensures steering messages are
    drained into the model request before user capabilities see it, and follow-up
    redirection runs after all other `after_node_run` hooks (which run in reverse).
    """

    def get_ordering(self) -> CapabilityOrdering:
        # Outermost so steering messages are drained into the request before other
        # capabilities see it, and follow-up redirection runs after all other
        # after_node_run hooks (which run in reverse order).
        return CapabilityOrdering(position='outermost')

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Drain steering messages into the model request.

        Appends to both `request_context.messages` (so the model sees them in this
        request) and `ctx.messages` (so they persist in the agent's message history).
        """
        drained = _drain_by_priority(ctx.pending_messages, 'steering')
        if drained:
            parts = [part for msg in drained for part in msg.parts]
            steering_request = ModelRequest(parts=parts)
            request_context.messages.append(steering_request)
            ctx.messages.append(steering_request)
        return request_context

    async def after_node_run(
        self,
        ctx: RunContext[Any],
        *,
        node: _agent_graph.AgentNode[Any, Any],
        result: _agent_graph.AgentNode[Any, Any] | End[FinalResult[Any]],
    ) -> _agent_graph.AgentNode[Any, Any] | End[FinalResult[Any]]:
        """Drain follow-up messages when the agent would otherwise end."""
        from pydantic_ai._agent_graph import ModelRequestNode
        from pydantic_graph import End

        if not isinstance(result, End):
            return result

        follow_ups = _drain_by_priority(ctx.pending_messages, 'follow_up')
        if not follow_ups:
            return result

        parts = [part for msg in follow_ups for part in msg.parts]
        request = ModelRequest(parts=parts)
        return ModelRequestNode(request=request)
