"""Auto-injected capability that drains the pending message queue at appropriate times."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.messages import ModelRequest, PendingMessage, PendingMessagePriority
from pydantic_ai.tools import RunContext

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.result import FinalResult
    from pydantic_graph import End


def _drain_by_priority(
    queue: deque[PendingMessage],
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
    queue.clear()
    queue.extend(remaining)
    return drained


class PendingMessageDrainCapability(AbstractCapability[Any]):
    """Drains the pending message queue at appropriate times.

    - Steering messages are injected before each model request.
    - Follow-up messages are injected when the agent would otherwise end,
      redirecting to a new ModelRequestNode to continue the conversation.

    This capability is always auto-injected and prepended to the capabilities
    list so that steering messages are drained first (before user capabilities
    see them) and follow-up drain runs last (after all other after_node_run hooks).
    """

    @classmethod
    def get_serialization_name(cls) -> None:
        return None

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Drain steering messages into the model request."""
        drained = _drain_by_priority(ctx.pending_messages, 'steering')
        if drained:
            parts = [part for msg in drained for part in msg.parts]
            request_context.messages.append(ModelRequest(parts=parts))
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
