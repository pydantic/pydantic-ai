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
    InstructionBaselinePart,
    InstructionDeltaPart,
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
                InstructionDeltaPart,
                InstructionBaselinePart,
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

    _tool_call: tuple[str, int] | None = field(default=None, repr=False, compare=False)
    """The `tool_call_id` and run step of the tool call whose result this content goes next to, if any.

    Set by `from_content` for parts enqueued with `priority='asap'` during a tool call (from the tool or a
    capability hook wrapping it). The step matters because a model may reuse a `tool_call_id` later, and
    content an earlier call enqueues late (say, from a task it started) mustn't go next to the later return.
    """

    @classmethod
    def from_content(
        cls,
        *content: EnqueueContent,
        priority: PendingMessagePriority = 'asap',
        _tool_call: tuple[str, int] | None = None,
    ) -> PendingMessage | None:
        """Build a `PendingMessage` from `enqueue` arguments, or `None` when there's nothing to send.

        Returns `None` for an empty call (enqueueing nothing is a no-op rather than an error).
        `_tool_call` is the `tool_call_id` and run step of the tool call being executed, if any. It's kept
        only for content made of parts, which can go next to that call's result.

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
            _tool_call = None
        return cls(messages=messages, priority=priority, _tool_call=_tool_call)


class PendingMessageQueue(list[PendingMessage]):
    """A run's pending messages with thread-safe append and drain operations."""

    def __init__(self, messages: Iterable[PendingMessage] = ()) -> None:
        super().__init__(messages)
        self._closed = False
        self._lock = threading.Lock()
        # The request tool-call content was last put into, as `(request it replaced first, current copy)`, and
        # the last part put next to each call's return in it, so a later delivery in the same step puts that
        # call's content after it.
        self._spliced_request: tuple[ModelRequest, ModelRequest] | None = None
        self._last_delivered_for_call: dict[str, ModelRequestPart] = {}

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

    def request_in_history(self, request: ModelRequest) -> ModelRequest:
        """The copy in history of `request`, if tool-call content was put into it, else `request` itself."""
        if self._spliced_request is not None and self._spliced_request[0] is request:
            return self._spliced_request[1]
        return request

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


def _splice_anchor(target: ModelRequest, tool_call_id: str, queue: PendingMessageQueue) -> ModelRequestPart:
    """The part in `target` that content for `tool_call_id` goes right after."""
    delivered = queue._last_delivered_for_call.get(tool_call_id)  # pyright: ignore[reportPrivateUsage]
    if delivered is not None and any(part is delivered for part in target.parts):
        return delivered
    return next(part for part in target.parts if isinstance(part, ToolReturnPart) and part.tool_call_id == tool_call_id)


def deliver_asap_messages(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
    """Deliver the queued `'asap'` messages into the request being prepared.

    Content tied to a tool call (see `PendingMessage._tool_call`) goes into the request that carries that call's [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart],
    right after it, when that request ends the history: several deliveries for the same call keep their
    enqueue order, whatever other calls' content arrives in between. Everything else is appended as its own
    messages, and so is anything drained after it, so content never lands ahead of what was enqueued first.
    Either way it lands in both `request_context.messages` (so the model sees it this step) and `ctx.messages`
    (so it persists in the agent's message history). The request the content went into is replaced by a copy,
    which `PendingMessageQueue.request_in_history` finds.

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
        tool_call = pending._tool_call  # pyright: ignore[reportPrivateUsage]
        if (
            tool_call is not None
            and tool_call[0] in answered_calls
            # The calls `target` answers ran in the step before this one, so an older call reused the id.
            and tool_call[1] >= ctx.run_step - 1
            # Content that came after something added as its own message stays after it.
            and not appended
            # `from_content` only keeps the call for content made of parts, which is one request.
            and len(messages) == 1
            and isinstance(request := messages[0], ModelRequest)
        ):
            spliced.setdefault(tool_call[0], []).append((pending.enqueue_id, request))
        else:
            appended.append((pending.enqueue_id, messages))

    events: list[EnqueuedMessagesEvent] = []
    if spliced:
        assert isinstance(target, ModelRequest)
        queue = run_queue(ctx)
        previous = queue._spliced_request  # pyright: ignore[reportPrivateUsage]
        if previous is not None and previous[1] is target:
            original = previous[0]
        else:
            original = target
            queue._last_delivered_for_call.clear()  # pyright: ignore[reportPrivateUsage]
        # Each call's content goes after its return, or after what an earlier delivery in this step put there.
        anchors = {id(_splice_anchor(target, call_id, queue)): call_id for call_id in spliced}
        parts: list[ModelRequestPart] = []
        for part in target.parts:
            parts.append(part)
            call_id = anchors.get(id(part))
            if call_id is None:
                continue
            for enqueue_id, request in spliced.pop(call_id, ()):
                parts.extend(request.parts)
                queue._last_delivered_for_call[call_id] = request.parts[-1]  # pyright: ignore[reportPrivateUsage]
                events.append(
                    EnqueuedMessagesEvent(enqueue_id=enqueue_id, messages=(request,), parts=tuple(request.parts))
                )
        spliced_target = replace(target, parts=parts)
        queue._spliced_request = (original, spliced_target)  # pyright: ignore[reportPrivateUsage]
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
