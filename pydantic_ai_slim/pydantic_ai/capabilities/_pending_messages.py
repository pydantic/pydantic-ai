"""Auto-injected capability that drains the pending message queue at appropriate times."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai._utils import now_utc
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import ModelRequest, ModelRequestPart, PendingMessage, PendingMessagePriority
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


def _flatten_drained(
    drained: list[PendingMessage],
    *,
    fallback_run_id: str | None,
) -> list[ModelRequest]:
    """Flatten drained pending messages into a list of `ModelRequest`s.

    Adjacent parts-style payloads merge into one synthesized request (matching what
    the model sees on the wire); each passthrough `ModelRequest` payload becomes its
    own message. Synthesized requests are stamped with `now_utc()` / `fallback_run_id`;
    passthrough requests keep producer-supplied values, only filling in `timestamp` /
    `run_id` when unset.
    """
    requests: list[ModelRequest] = []
    pending_parts: list[ModelRequestPart] = []

    def flush_parts() -> None:
        if pending_parts:
            requests.append(ModelRequest(parts=pending_parts.copy(), timestamp=now_utc(), run_id=fallback_run_id))
            pending_parts.clear()

    for msg in drained:
        if isinstance(msg.payload, ModelRequest):
            flush_parts()
            request = msg.payload
            if request.timestamp is None:
                request.timestamp = now_utc()
            if request.run_id is None:
                request.run_id = fallback_run_id
            requests.append(request)
        else:
            pending_parts.extend(msg.payload)
    flush_parts()

    return requests


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

        Adjacent parts-style payloads merge into one synthesized
        [`ModelRequest`][pydantic_ai.messages.ModelRequest]; each passthrough
        `ModelRequest` payload becomes its own message. Each resulting request is
        appended to both `request_context.messages` (so the model sees it this
        step) and `ctx.messages` (so it persists in the agent's message history).
        """
        drained = _drain_by_priority(ctx.pending_messages, 'steering')
        # Stamp explicitly here: ModelRequestNode.run() only stamps `self.request`
        # (the current node's request). The agent graph fixes up `messages[-1]`
        # before calling the model, but relying on that is fragile — another
        # capability could append after us.
        for request in _flatten_drained(drained, fallback_run_id=ctx.run_id):
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

        Adjacent parts-style payloads merge into one synthesized
        [`ModelRequest`][pydantic_ai.messages.ModelRequest]; each passthrough
        `ModelRequest` payload becomes its own message. The last resulting request
        becomes the redirect [`ModelRequestNode`][pydantic_ai._agent_graph.ModelRequestNode]'s
        request; any earlier ones are appended to `ctx.messages` so they appear
        in history before the redirect.
        """
        from pydantic_ai._agent_graph import ModelRequestNode
        from pydantic_graph import End

        if not isinstance(result, End):
            return result

        follow_ups = _drain_by_priority(ctx.pending_messages, 'follow_up')
        if not follow_ups:
            return result

        requests = _flatten_drained(follow_ups, fallback_run_id=ctx.run_id)
        # `final` becomes the redirect node's request; `ModelRequestNode._prepare_request`
        # will re-stamp it during the graph lifecycle. `_flatten_drained` already
        # stamped it, which is harmless (the lifecycle stamp overwrites).
        *extras, final = requests
        ctx.messages.extend(extras)
        return ModelRequestNode(request=final)
