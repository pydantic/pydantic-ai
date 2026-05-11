"""Auto-injected capability that drains the pending message queue at appropriate times."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai._utils import now_utc
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import PendingMessage, PendingMessagePriority
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

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # not spec-constructible (internal, auto-injected)

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Drain steering messages into the model request.

        Each pending steering message is emitted as its own [`ModelRequest`][pydantic_ai.messages.ModelRequest],
        appended to both `request_context.messages` (so the model sees them in this
        request) and `ctx.messages` (so they persist in the agent's message history).
        Consecutive `ModelRequest`s are merged on the wire by
        [`_clean_message_history`][pydantic_ai._utils._clean_message_history].
        """
        for msg in _drain_by_priority(ctx.pending_messages, 'steering'):
            # Stamp explicitly: ModelRequestNode.run() only stamps `self.request` (the
            # current node's request). The agent graph also fixes up `messages[-1]`
            # before calling the model, but relying on that is fragile — another
            # capability could append after us. Stamp at construction so the
            # persisted history always has timestamp/run_id, regardless of order.
            # Leave producer-supplied values alone (e.g. when enqueueing a full ModelRequest).
            request = msg.request
            if request.timestamp is None:
                request.timestamp = now_utc()
            if request.run_id is None:
                request.run_id = ctx.run_id
            request_context.messages.append(request)
            ctx.messages.append(request)
        return request_context

    async def after_node_run(
        self,
        ctx: RunContext[Any],
        *,
        node: _agent_graph.AgentNode[Any, Any],
        result: _agent_graph.AgentNode[Any, Any] | End[FinalResult[Any]],
    ) -> _agent_graph.AgentNode[Any, Any] | End[FinalResult[Any]]:
        """Drain follow-up messages when the agent would otherwise end.

        Each pending follow-up becomes its own [`ModelRequest`][pydantic_ai.messages.ModelRequest].
        The last one becomes the redirect [`ModelRequestNode`][pydantic_ai._agent_graph.ModelRequestNode]'s
        request; any earlier ones are appended to `ctx.messages` so they appear in
        history before the redirect.
        """
        from pydantic_ai._agent_graph import ModelRequestNode
        from pydantic_graph import End

        if not isinstance(result, End):
            return result

        follow_ups = _drain_by_priority(ctx.pending_messages, 'follow_up')
        if not follow_ups:
            return result

        *extras, final = follow_ups
        for extra in extras:
            request = extra.request
            if request.timestamp is None:
                request.timestamp = now_utc()
            if request.run_id is None:
                request.run_id = ctx.run_id
            ctx.messages.append(request)
        # No explicit stamping for `final.request`: `ModelRequestNode._prepare_request`
        # stamps `self.request` during the graph lifecycle (see `_agent_graph.py`
        # `self.request.timestamp = now_utc()` / `self.request.run_id = ...`).
        # The earlier-extras path stamps explicitly because it appends directly to
        # `messages` outside that lifecycle.
        return ModelRequestNode(request=final.request)
