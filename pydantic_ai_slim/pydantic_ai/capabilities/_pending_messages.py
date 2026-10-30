"""Auto-injected capability that drains the pending message queue at appropriate times."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai._agent_graph import ModelRequestNode
from pydantic_ai._enqueue import PendingMessage, carry_over, deliver_asap_messages, run_queue, stamped_messages
from pydantic_ai._run_context import dispatch_event_now
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import EnqueuedMessagesEvent, ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.output import PendingModelResponse
from pydantic_ai.tools import DeferredToolRequests, RunContext
from pydantic_graph import End

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.result import FinalResult


class PendingMessageDrainCapability(AbstractCapability[Any]):
    """Drains the pending message queue at appropriate times.

    - `'asap'` messages drain at the earliest opportunity: into the next
      [`ModelRequest`][pydantic_ai.messages.ModelRequest] via `before_model_request`,
      or — if the agent would otherwise terminate — redirected through a new
      `ModelRequestNode` at the end of the run.
    - `'when_idle'` messages drain only when the agent would otherwise terminate
      and no `'asap'` messages remain, after any `'asap'` redirect.

    This capability is always auto-injected and placed outermost via
    [`CapabilityOrdering`][pydantic_ai.capabilities.abstract.CapabilityOrdering].
    Final draining happens at the graph-advancement seam after the full capability
    hook chain, so user ordering constraints cannot make the queue close before a
    hook has had a chance to redirect an apparent [`End`][pydantic_graph.End].
    """

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # not spec-constructible (internal, auto-injected)

    def _realtime_uncalled_hooks(self) -> tuple[str, ...]:
        # A realtime session drains the queue itself, at its own turn boundaries.
        # Measured against this class, so a subclass that adds its own hooks still warns.
        return self._realtime_uncalled_hooks_overriding(PendingMessageDrainCapability)

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

        Emits one [`EnqueuedMessagesEvent`][pydantic_ai.messages.EnqueuedMessagesEvent] per drained
        [`enqueue`][pydantic_ai.tools.RunContext.enqueue] call, in enqueue order, describing the
        messages exactly as delivered here.
        """
        return deliver_asap_messages(ctx, request_context)


async def drain_pending_messages_at_end(
    ctx: RunContext[Any],
    result: _agent_graph.AgentNode[Any, Any] | End[FinalResult[Any]],
) -> _agent_graph.AgentNode[Any, Any] | End[FinalResult[Any]]:
    """Drain pending messages after all capability hooks if the agent would terminate.

    Drain `'asap'` messages first (anything that arrived after the most recent
    `before_model_request` and would otherwise be lost), then `'when_idle'` messages.
    Each priority is appended independently so the history keeps the priority split
    visible (matches pi-mono's separate steering / follow-up turns). On the wire,
    `_clean_message_history` re-merges adjacent requests with compatible instructions,
    so the model still sees one turn.

    The last resulting request becomes the redirect
    [`ModelRequestNode`][pydantic_ai._agent_graph.ModelRequestNode]'s request; any
    earlier ones are appended to `ctx.messages` so they appear in history before the
    redirect. A run that pauses is never redirected: see `_keep_at_pause`. Emits one
    [`EnqueuedMessagesEvent`][pydantic_ai.messages.EnqueuedMessagesEvent] per drained
    [`enqueue`][pydantic_ai.tools.RunContext.enqueue] call, in enqueue order.
    """
    if not isinstance(result, End):
        return result

    if isinstance(result.data.output, DeferredToolRequests | PendingModelResponse):
        await _keep_at_pause(ctx, run_queue(ctx).drain_at_pause())
        return result

    # Pi-mono parity: drain `'asap'` first so anything that arrived during the
    # final step (e.g. a background task completing while the model produced
    # its final response) gets delivered before `'when_idle'` messages, and the
    # agent gets another turn rather than terminating with the message lost.
    leftover_asap, when_idle = run_queue(ctx).drain_at_end()
    if not leftover_asap and not when_idle:
        return result

    drained = [*leftover_asap, *when_idle]
    stamped = [
        (
            pending,
            stamped_messages(pending, fallback_run_id=ctx.run_id, fallback_conversation_id=ctx.conversation_id),
        )
        for pending in drained
    ]
    messages = [message for _, pending_messages in stamped for message in pending_messages]
    # `final` becomes the redirect node's request; `ModelRequestNode._prepare_request`
    # will re-stamp it during the graph lifecycle. `stamped_messages` already
    # stamped it, which is harmless (the lifecycle stamp overwrites). `from_content`
    # guarantees each `PendingMessage` ends in a `ModelRequest`, but a producer can
    # construct `PendingMessage` (or mutate `RunContext.pending_messages`) directly, so
    # we check rather than assert. Every message except `final` is appended to history
    # before the redirect.
    final = messages[-1]
    if not isinstance(final, ModelRequest):
        raise UserError(
            'Enqueued content must end with a `ModelRequest` so the agent has a request to respond to, '
            f'but the last queued message is a `{type(final).__name__}`.'
        )
    for pending, pending_messages in stamped:
        for message in pending_messages:
            if message is not final:
                ctx.messages.append(message)
        _announce(ctx, pending, pending_messages)
    return ModelRequestNode(request=final)


async def _keep_at_pause(ctx: RunContext[Any], pending: list[PendingMessage]) -> None:
    """Keep the content still queued when the run pauses, without giving the model another turn.

    A pause (deferred tools, or a response the provider is still working on) hands control back to the caller,
    so a redirect would skip it. When the paused step's own request ends the history, the content goes after it,
    as it would have been delivered. Otherwise the history has to keep ending in the paused response, which is
    what a resumed run acts on, and so does a queued `ModelResponse`, which would read as the paused one: the
    content is carried on the last message instead, and the next run started from this history queues it again
    (see `take_carried_over`), so it reaches the model at that run's first request.

    Either way the content is now in the history, so its `EnqueuedMessagesEvent` is emitted here, and dispatched to
    capability listeners right away (one acknowledging deliveries, like `SubAgents`, would otherwise deliver it again):
    no node stream follows to drain it.
    """
    if not pending:
        return
    stamped = [
        (item, stamped_messages(item, fallback_run_id=ctx.run_id, fallback_conversation_id=ctx.conversation_id))
        for item in pending
    ]
    if isinstance(ctx.messages[-1], ModelRequest) and not any(
        isinstance(message, ModelResponse) for _, messages in stamped for message in messages
    ):
        for _, messages in stamped:
            ctx.messages.extend(messages)
    else:
        ctx.messages[-1] = carry_over(ctx.messages[-1], pending)
    for item, messages in stamped:
        if (event := _announce(ctx, item, messages)) is not None:
            await dispatch_event_now(ctx, event)


def _announce(
    ctx: RunContext[Any], pending: PendingMessage, messages: list[ModelMessage]
) -> EnqueuedMessagesEvent | None:
    """Emit the `EnqueuedMessagesEvent` for `pending`, unless it was emitted when an earlier run paused."""
    if pending._announced:  # pyright: ignore[reportPrivateUsage]
        return None
    event = EnqueuedMessagesEvent(enqueue_id=pending.enqueue_id, messages=tuple(messages))
    ctx._emit_event(event)  # pyright: ignore[reportPrivateUsage]
    return event
