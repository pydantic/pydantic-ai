"""Internal helpers for the `RunContext.enqueue` / `AgentRun.enqueue` APIs.

These types live here (rather than in `messages.py`) because they're internal runtime
state for the pending message queue, not part of the wire-serializable message history.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, SupportsIndex, TypeAlias

from ._utils import fill_run_metadata
from ._uuid import uuid7
from .exceptions import UserError
from .messages import (
    EnqueuedMessagesEvent,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    ToolAvailabilityDeltaPart,
    ToolReturnPart,
    ToolSearchReturnPart,
    UserPromptPart,
)

if TYPE_CHECKING:
    from ._run_context import RunContext
    from .messages import UserContent
    from .models import ModelRequestContext


PendingMessagePriority: TypeAlias = Literal['asap', 'when_idle']
"""When to deliver a pending message.

- `'asap'`: Delivered at the earliest opportunity — either prepended to the next
    [`ModelRequest`][pydantic_ai.messages.ModelRequest], or, if the agent would
    otherwise terminate before another request, used to redirect the run into one
    more request.
- `'when_idle'`: Delivered only when the agent would otherwise terminate, after
    any `'asap'` messages. Doesn't interrupt in-flight work.
"""


EnqueueContent: TypeAlias = 'UserContent | ModelRequestPart | ModelMessage'
"""A single item accepted by [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue]
and [`AgentRun.enqueue`][pydantic_ai.run.AgentRun.enqueue].

`enqueue` is variadic, so each item is one positional argument:

- [`UserContent`][pydantic_ai.messages.UserContent] (a `str` or a piece of multi-modal content
    like an [`ImageUrl`][pydantic_ai.messages.ImageUrl]): adjacent user content is gathered into a
    single [`UserPromptPart`][pydantic_ai.messages.UserPromptPart], so `enqueue('caption', image)`
    forms one user turn. To pass an existing list, spread it: `enqueue(*items)`.
- [`ModelRequestPart`][pydantic_ai.messages.ModelRequestPart] (e.g. a
    [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]): included verbatim.
- [`ModelMessage`][pydantic_ai.messages.ModelMessage] (a complete
    [`ModelRequest`][pydantic_ai.messages.ModelRequest] or
    [`ModelResponse`][pydantic_ai.messages.ModelResponse]): emitted as its own message.

Consecutive part-style items (user content and `ModelRequestPart`s) are coalesced into a single
`ModelRequest`; complete `ModelMessage`s stay separate. This lets one `enqueue` call inject an
interleaved exchange (e.g. a synthetic tool call + result — a `ModelResponse` followed by a
`ModelRequest`). The assembled sequence must end in a `ModelRequest` so the agent has something to
respond to.
"""


def _build_enqueue_messages(items: Sequence[EnqueueContent]) -> list[ModelMessage]:
    """Assemble enqueue items into a list of [`ModelMessage`][pydantic_ai.messages.ModelMessage]s.

    Adjacent [`UserContent`][pydantic_ai.messages.UserContent] items are gathered into one
    [`UserPromptPart`][pydantic_ai.messages.UserPromptPart], and part-style items (user content and
    [`ModelRequestPart`][pydantic_ai.messages.ModelRequestPart]s) are coalesced into a single
    [`ModelRequest`][pydantic_ai.messages.ModelRequest]; complete `ModelMessage`s are emitted as-is.
    Order is preserved, so a `ModelResponse` followed by part-style items produces the response then
    a request built from those parts.
    """
    messages: list[ModelMessage] = []
    parts: list[ModelRequestPart] = []
    content: list[UserContent] = []

    def flush_content() -> None:
        if content:
            # Collapse a lone string to `str` content, matching `Agent.run('...')`; anything else
            # (multiple items, or a single non-string like an image) becomes a content list.
            single = content[0] if len(content) == 1 and isinstance(content[0], str) else list(content)
            parts.append(UserPromptPart(content=single))
            content.clear()

    def flush_request() -> None:
        flush_content()
        if parts:
            messages.append(ModelRequest(parts=list(parts)))
            parts.clear()

    for item in items:
        if isinstance(item, (ModelRequest, ModelResponse)):
            flush_request()
            messages.append(item)
        elif isinstance(
            item,
            (
                SystemPromptPart,
                UserPromptPart,
                ToolReturnPart,
                RetryPromptPart,
                ToolSearchReturnPart,
                ToolAvailabilityDeltaPart,
                SpeechPart,
            ),
        ):
            flush_content()
            parts.append(item)
        else:
            content.append(item)
    flush_request()
    return messages


@dataclass
class PendingMessage:
    """One or more [`ModelMessage`][pydantic_ai.messages.ModelMessage]s queued for injection into the agent conversation.

    Enqueued via [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue] or
    [`AgentRun.enqueue`][pydantic_ai.run.AgentRun.enqueue] and automatically drained
    at the appropriate time during the agent run by the internal `PendingMessageDrainCapability`.
    """

    messages: list[ModelMessage]
    """The message(s) to inject, in order. Always ends in a
    [`ModelRequest`][pydantic_ai.messages.ModelRequest]."""

    priority: PendingMessagePriority = 'asap'
    """When to deliver these messages:

    - `'asap'`: at the earliest opportunity (next model request, or redirect if the agent
        would otherwise terminate).
    - `'when_idle'`: only when the agent would otherwise terminate, after `'asap'` messages.
    """

    enqueue_id: str = field(default_factory=lambda: str(uuid7()))
    """Unique identifier for this enqueue call, surfaced on the
    [`EnqueuedMessagesEvent`][pydantic_ai.messages.EnqueuedMessagesEvent] emitted when the messages
    are delivered, and returned by [`enqueue`][pydantic_ai.tools.RunContext.enqueue]."""

    tool_call_id: str | None = None
    """The tool call whose result this content is delivered next to, if any.

    Set when the content was enqueued with `priority='asap'` during a tool call (from the tool body or a
    capability hook wrapping it) and is made only of parts, so delivery puts those parts in the
    [`ModelRequest`][pydantic_ai.messages.ModelRequest] carrying that call's
    [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart], right after it. Content holding a complete
    `ModelRequest` or `ModelResponse` keeps its own message-level fields, so it's always delivered as its
    own messages.
    """

    @classmethod
    def from_content(
        cls,
        *content: EnqueueContent,
        priority: PendingMessagePriority = 'asap',
        tool_call_id: str | None = None,
    ) -> PendingMessage | None:
        """Build a `PendingMessage` from `enqueue` arguments, or `None` when there's nothing to send.

        Returns `None` for an empty call (enqueueing nothing is a no-op rather than an error).
        `tool_call_id` is the tool call being executed, if any; it's kept only when the content can be
        delivered next to that call's result (see `PendingMessage.tool_call_id`).

        Raises:
            UserError: If the assembled messages don't end in a
                [`ModelRequest`][pydantic_ai.messages.ModelRequest] — e.g. a lone `ModelResponse` —
                since the agent needs a request to respond to.
        """
        messages = _build_enqueue_messages(content)
        if not messages:
            return None
        if not isinstance(messages[-1], ModelRequest):
            raise UserError(
                'Enqueued content must end with a `ModelRequest` (or user content / `ModelRequestPart` '
                'items that form one), so the agent has a request to respond to.'
            )
        if priority != 'asap' or any(isinstance(item, ModelRequest | ModelResponse) for item in content):
            tool_call_id = None
        return cls(messages=messages, priority=priority, tool_call_id=tool_call_id)


class PendingMessageQueue(list[PendingMessage]):
    """A run's pending messages with thread-safe append and drain operations."""

    def __init__(self, messages: Iterable[PendingMessage] = ()) -> None:
        super().__init__(messages)
        self._closed = False
        self._lock = threading.Lock()

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[type[PendingMessageQueue], tuple[list[PendingMessage]]]:
        return PendingMessageQueue, (list(self),)

    def append(self, pending: PendingMessage) -> None:
        with self._lock:
            if self._closed:
                raise UserError('`enqueue` is not available because the agent run has ended.')
            super().append(pending)

    def pop_priority(self, priority: PendingMessagePriority) -> list[PendingMessage]:
        with self._lock:
            return self._pop_priority(priority)

    def drain_at_end(self) -> tuple[list[PendingMessage], list[PendingMessage]]:
        """Drain both priorities, or atomically close an empty queue."""
        with self._lock:
            asap = self._pop_priority('asap')
            when_idle = self._pop_priority('when_idle')
            if not asap and not when_idle:
                self._closed = True
            return asap, when_idle

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def _pop_priority(self, priority: PendingMessagePriority) -> list[PendingMessage]:
        selected = [pending for pending in self if pending.priority == priority]
        self[:] = [pending for pending in self if pending.priority != priority]
        return selected


def stamped_messages(
    pending: PendingMessage,
    *,
    fallback_run_id: str | None,
    fallback_conversation_id: str | None,
) -> list[ModelMessage]:
    """Stamp a pending message's messages' `timestamp` / `run_id` / `conversation_id` where unset.

    Each [`PendingMessage`][pydantic_ai._enqueue.PendingMessage] carries one or more built
    [`ModelMessage`][pydantic_ai.messages.ModelMessage]s (assembled at enqueue time by
    [`PendingMessage.from_content`][pydantic_ai._enqueue.PendingMessage.from_content]); this only
    fills in framework-tracked metadata that the producer left unset, so producer-supplied values
    are preserved.
    """
    messages: list[ModelMessage] = []
    for message in pending.messages:
        fill_run_metadata(message, run_id=fallback_run_id, conversation_id=fallback_conversation_id)
        messages.append(message)
    return messages


def run_queue(ctx: RunContext[Any]) -> PendingMessageQueue:
    """The run's pending-message queue.

    Graph runs always carry the thread-safe queue. Other `RunContext` producers (durable-exec guards,
    synthetic contexts) never reach the code that drains it.
    """
    queue = ctx.pending_messages
    assert isinstance(queue, PendingMessageQueue)
    return queue


def deliver_asap_messages(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
    """Deliver the queued `'asap'` messages into the request being prepared.

    Content tied to a tool call (see [`PendingMessage.tool_call_id`][pydantic_ai._enqueue.PendingMessage.tool_call_id])
    goes into the request that carries that call's [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart],
    right after it, when that request ends the history: several deliveries for the same call keep their
    enqueue order, whatever other calls' content arrives in between. Everything else is appended as its own
    messages. Either way it lands in both `request_context.messages` (so the model sees it this step) and
    `ctx.messages` (so it persists in the agent's message history).

    One [`EnqueuedMessagesEvent`][pydantic_ai.messages.EnqueuedMessagesEvent] is emitted per drained
    [`enqueue`][pydantic_ai.tools.RunContext.enqueue] call, in the order the content lands in history,
    describing it exactly as delivered. A UI adapter streams them in that order and its client loads them
    back in that order, so the client's history renders like the server's.
    """
    drained = run_queue(ctx).pop_priority('asap')
    if not drained:
        return request_context

    # Only the request about to be sent, so content arriving later (say, from a task the tool started)
    # never rewrites history the model has already seen. It's replaced rather than mutated, since
    # instrumentation caches each message's serialized form, so it must be in both lists to be found again.
    target = ctx.messages[-1] if ctx.messages else None
    answered_calls: set[str] = set()
    if isinstance(target, ModelRequest) and any(message is target for message in request_context.messages):
        answered_calls = {part.tool_call_id for part in target.parts if isinstance(part, ToolReturnPart)}

    spliced: dict[str, list[tuple[str, ModelRequest]]] = {}
    appended: list[tuple[str, list[ModelMessage]]] = []
    for pending in drained:
        messages = stamped_messages(pending, fallback_run_id=ctx.run_id, fallback_conversation_id=ctx.conversation_id)
        if pending.tool_call_id in answered_calls:
            # `from_content` only keeps `tool_call_id` for content made of parts, which is one request.
            (request,) = messages
            assert isinstance(request, ModelRequest)
            spliced.setdefault(pending.tool_call_id, []).append((pending.enqueue_id, request))
        else:
            appended.append((pending.enqueue_id, messages))

    events: list[EnqueuedMessagesEvent] = []
    if spliced:
        assert isinstance(target, ModelRequest)
        parts: list[ModelRequestPart] = []
        for part in target.parts:
            parts.append(part)
            if isinstance(part, ToolReturnPart):
                for enqueue_id, request in spliced.pop(part.tool_call_id, ()):
                    parts.extend(request.parts)
                    events.append(
                        EnqueuedMessagesEvent(enqueue_id=enqueue_id, messages=(request,), parts=tuple(request.parts))
                    )
        spliced_target = replace(target, parts=parts)
        ctx.messages[-1] = spliced_target
        request_context.messages = [
            spliced_target if message is target else message for message in request_context.messages
        ]
    for enqueue_id, messages in appended:
        request_context.messages = [*request_context.messages, *messages]
        ctx.messages.extend(messages)
        events.append(EnqueuedMessagesEvent(enqueue_id=enqueue_id, messages=tuple(messages)))
    for event in events:
        ctx._emit_event(event)  # pyright: ignore[reportPrivateUsage]
    return request_context
