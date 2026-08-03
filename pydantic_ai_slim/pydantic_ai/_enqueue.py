"""Internal helpers for the `RunContext.enqueue` / `AgentRun.enqueue` APIs.

These types live here (rather than in `messages.py`) because they're internal runtime
state for the pending message queue, not part of the wire-serializable message history.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

from ._uuid import uuid7
from .exceptions import UserError
from .messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
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
            item, (SystemPromptPart, UserPromptPart, ToolReturnPart, RetryPromptPart, ToolSearchReturnPart)
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
