"""Auto-injected capability that drains the pending message queue at appropriate times."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai._agent_graph import ModelRequestNode
from pydantic_ai._enqueue import PendingMessage, PendingMessagePriority
from pydantic_ai._utils import fill_run_metadata
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.tools import RunContext
from pydantic_graph import End

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.result import FinalResult


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


def _stamped_messages(
    drained: list[PendingMessage],
    *,
    fallback_run_id: str | None,
    fallback_conversation_id: str | None,
) -> list[ModelMessage]:
    """Flatten drained pending messages, stamping `timestamp` / `run_id` / `conversation_id` where unset.

    Each [`PendingMessage`][pydantic_ai._enqueue.PendingMessage] carries one or more built
    [`ModelMessage`][pydantic_ai.messages.ModelMessage]s (assembled at enqueue time by
    [`PendingMessage.from_content`][pydantic_ai._enqueue.PendingMessage.from_content]); this only
    fills in framework-tracked metadata that the producer left unset, so producer-supplied values
    are preserved.
    """
    messages: list[ModelMessage] = []
    for pending in drained:
        for message in pending.messages:
            fill_run_metadata(message, run_id=fallback_run_id, conversation_id=fallback_conversation_id)
            messages.append(message)
    return messages


class PendingMessageDrainCapability(AbstractCapability[Any]):
    """Drains the pending message queue at appropriate times.

    - `'asap'` messages drain at the earliest opportunity: into the next
      [`ModelRequest`][pydantic_ai.messages.ModelRequest] via `before_model_request`,
      or — if the agent would otherwise terminate — redirected through a new
      `ModelRequestNode` from `after_node_run`.
    - `'when_idle'` messages drain only when the agent would otherwise terminate
      and no `'asap'` messages remain, after any `'asap'` redirect.

    This capability is always auto-injected and placed outermost via
    [`CapabilityOrdering`][pydantic_ai.capabilities.abstract.CapabilityOrdering]
    so it wraps around other capabilities. This ensures `'asap'` messages are
    drained into the model request before user capabilities see it, and the
    end-of-run redirection runs after all other `after_node_run` hooks (which
    run in reverse).
    """

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # not spec-constructible (internal, auto-injected)

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Drain `'asap'` messages into the upcoming model request.

        Each drained request is appended to both `request_context.messages` (so the model
        sees it this step) and `ctx.messages` (so it persists in the agent's message
        history). Stamps `timestamp`/`run_id`/`conversation_id` if the producer didn't —
        `ModelRequestNode.run()` only stamps `self.request` (the current node's request),
        and capabilities downstream of us might append more messages, so we can't rely on
        that fixup.
        """
        assert ctx.pending_messages is not None, 'drain runs during an agent run, which always has a queue'
        drained = _drain_by_priority(ctx.pending_messages, 'asap')
        for message in _stamped_messages(
            drained, fallback_run_id=ctx.run_id, fallback_conversation_id=ctx.conversation_id
        ):
            request_context.messages.append(message)
            ctx.messages.append(message)
        return request_context

    async def after_node_run(
        self,
        ctx: RunContext[Any],
        *,
        node: _agent_graph.AgentNode[Any, Any],
        result: _agent_graph.AgentNode[Any, Any] | End[FinalResult[Any]],
    ) -> _agent_graph.AgentNode[Any, Any] | End[FinalResult[Any]]:
        """Drain remaining `'asap'` and `'when_idle'` messages if the agent would terminate.

        If the run is about to end, drain `'asap'` messages first (anything that arrived
        after the most recent `before_model_request` and would otherwise be lost), then
        `'when_idle'` messages. Each priority is appended independently so the history
        keeps the priority split visible (matches pi-mono's separate steering / follow-up
        turns). On the wire, `_clean_message_history` re-merges adjacent requests with
        compatible instructions, so the model still sees one turn.

        The last resulting request becomes the redirect
        [`ModelRequestNode`][pydantic_ai._agent_graph.ModelRequestNode]'s request; any
        earlier ones are appended to `ctx.messages` so they appear in history before the
        redirect.
        """
        if not isinstance(result, End):
            return result

        assert ctx.pending_messages is not None, 'drain runs during an agent run, which always has a queue'
        # Pi-mono parity: drain `'asap'` first so anything that arrived during the
        # final step (e.g. a background task completing while the model produced
        # its final response) gets delivered before `'when_idle'` messages, and the
        # agent gets another turn rather than terminating with the message lost.
        leftover_asap = _drain_by_priority(ctx.pending_messages, 'asap')
        when_idle = _drain_by_priority(ctx.pending_messages, 'when_idle')
        if not leftover_asap and not when_idle:
            return result

        messages = [
            *_stamped_messages(leftover_asap, fallback_run_id=ctx.run_id, fallback_conversation_id=ctx.conversation_id),
            *_stamped_messages(when_idle, fallback_run_id=ctx.run_id, fallback_conversation_id=ctx.conversation_id),
        ]
        # `final` becomes the redirect node's request; `ModelRequestNode._prepare_request`
        # will re-stamp it during the graph lifecycle. `_stamped_messages` already
        # stamped it, which is harmless (the lifecycle stamp overwrites). `from_content`
        # guarantees each `PendingMessage` ends in a `ModelRequest`, but a producer can
        # construct `PendingMessage` (or mutate `RunContext.pending_messages`) directly, so
        # we check rather than assert. Any earlier responses/requests become `extras`
        # appended to history before the redirect.
        *extras, final = messages
        if not isinstance(final, ModelRequest):
            raise UserError(
                'Enqueued content must end with a `ModelRequest` so the agent has a request to respond to, '
                f'but the last queued message is a `{type(final).__name__}`.'
            )
        ctx.messages.extend(extras)
        return ModelRequestNode(request=final)
