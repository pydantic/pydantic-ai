"""Internal helpers for the `RunContext.enqueue` / `AgentRun.enqueue` APIs.

These types live here (rather than in `messages.py`) because they're internal runtime
state for the pending message queue, not part of the wire-serializable message history.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, SupportsIndex, TypeAlias

from ._uuid import uuid7
from .exceptions import UserError
from .messages import (
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
    from .messages import UserContent


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

    @classmethod
    def from_content(cls, *content: EnqueueContent, priority: PendingMessagePriority = 'asap') -> PendingMessage | None:
        """Build a `PendingMessage` from `enqueue` arguments, or `None` when there's nothing to send.

        Returns `None` for an empty call (enqueueing nothing is a no-op rather than an error).

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
        return cls(messages=messages, priority=priority)


class PendingMessageQueue(list[PendingMessage]):
    """A run's pending messages with thread-safe append and drain operations."""

    def __init__(
        self, messages: Iterable[PendingMessage] = (), capability_loads: Iterable[PendingMessage] = ()
    ) -> None:
        super().__init__(messages)
        # Kept apart from the list items: a capability load rides along with the next model request
        # but must never make the run continue, and code that checks whether messages are pending
        # (e.g. to decide whether the run will continue) must not see it.
        self._capability_loads: list[PendingMessage] = list(capability_loads)
        self._closed = False
        self._lock = threading.Lock()

    def __reduce_ex__(
        self, protocol: SupportsIndex
    ) -> tuple[type[PendingMessageQueue], tuple[list[PendingMessage], list[PendingMessage]]]:
        return PendingMessageQueue, (list(self), list(self._capability_loads))

    def append(self, pending: PendingMessage) -> None:
        with self._lock:
            if self._closed:
                raise UserError('`enqueue` is not available because the agent run has ended.')
            super().append(pending)

    def pop_priority(self, priority: PendingMessagePriority) -> list[PendingMessage]:
        with self._lock:
            return self._pop_priority(priority)

    @property
    def capability_loads(self) -> tuple[PendingMessage, ...]:
        """Capability loads queued by `RunContext.load_capability` that haven't been delivered yet."""
        with self._lock:
            return tuple(self._capability_loads)

    def append_capability_load(self, pending: PendingMessage) -> None:
        with self._lock:
            if self._closed:
                raise UserError('`load_capability` is not available because the agent run has ended.')
            self._capability_loads.append(pending)

    def pop_capability_loads(self) -> list[PendingMessage]:
        with self._lock:
            loads, self._capability_loads = self._capability_loads, []
            return loads

    def drain_at_end(self) -> tuple[list[PendingMessage], list[PendingMessage]]:
        """Drain both priorities, or atomically close a queue with nothing that continues the run."""
        with self._lock:
            asap = self._pop_priority('asap')
            when_idle = self._pop_priority('when_idle')
            loads, self._capability_loads = self._capability_loads, []
            if not asap and not when_idle:
                # Capability loads alone never continue the run, so they're dropped with it.
                self._closed = True
                return [], []
            # Delivered first when other messages continue the run anyway.
            return [*loads, *asap], when_idle

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def _pop_priority(self, priority: PendingMessagePriority) -> list[PendingMessage]:
        selected = [pending for pending in self if pending.priority == priority]
        self[:] = [pending for pending in self if pending.priority != priority]
        return selected
