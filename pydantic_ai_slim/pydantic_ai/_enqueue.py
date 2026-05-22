"""Internal helpers for the `RunContext.enqueue` / `AgentRun.enqueue` APIs.

These types live here (rather than in `messages.py`) because they're internal runtime
state for the pending message queue, not part of the wire-serializable message history.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from .exceptions import UserError
from .messages import ModelMessage, ModelRequest, ModelRequestPart, ModelResponse, UserPromptPart

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


EnqueueContent: TypeAlias = 'str | Sequence[UserContent] | ModelRequestPart | ModelMessage'
"""A single item accepted by [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue]
and [`AgentRun.enqueue`][pydantic_ai.run.AgentRun.enqueue].

- `str` or `Sequence[UserContent]`: wrapped in a [`UserPromptPart`][pydantic_ai.messages.UserPromptPart]
    (matching the shape of `Agent.run(user_prompt=...)`).
- [`ModelRequestPart`][pydantic_ai.messages.ModelRequestPart] (e.g. a
    [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]): included verbatim.
- [`ModelMessage`][pydantic_ai.messages.ModelMessage] (a complete
    [`ModelRequest`][pydantic_ai.messages.ModelRequest] or
    [`ModelResponse`][pydantic_ai.messages.ModelResponse]): emitted as its own message.

Consecutive part-style items (`str` / `Sequence[UserContent]` / `ModelRequestPart`) are coalesced
into a single `ModelRequest`; complete `ModelMessage`s stay separate. This lets one `enqueue`
call inject an interleaved exchange (e.g. a synthetic tool-search call + result — a `ModelResponse`
followed by a `ModelRequest`). The assembled sequence must end in a `ModelRequest` so the agent has
something to respond to.
"""


def _build_enqueue_messages(items: Sequence[EnqueueContent]) -> list[ModelMessage]:
    """Assemble enqueue items into a list of [`ModelMessage`][pydantic_ai.messages.ModelMessage]s.

    Part-style items (`str` / `Sequence[UserContent]` / `ModelRequestPart`) are coalesced into a
    single [`ModelRequest`][pydantic_ai.messages.ModelRequest]; complete `ModelMessage`s are emitted
    as-is. Order is preserved, so a `ModelResponse` followed by part-style items produces the
    response then a request built from those parts.
    """
    messages: list[ModelMessage] = []
    loose_parts: list[ModelRequestPart] = []

    def flush() -> None:
        if loose_parts:
            messages.append(ModelRequest(parts=list(loose_parts)))
            loose_parts.clear()

    for item in items:
        if isinstance(item, (ModelRequest, ModelResponse)):
            flush()
            messages.append(item)
        elif isinstance(item, (str, Sequence)):
            loose_parts.append(UserPromptPart(content=item))
        else:
            loose_parts.append(item)
    flush()
    return messages


@dataclass
class PendingMessage:
    """One or more [`ModelMessage`][pydantic_ai.messages.ModelMessage]s queued for injection into the agent conversation.

    Enqueued via [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue] or
    [`AgentRun.enqueue`][pydantic_ai.run.AgentRun.enqueue] and automatically drained
    at the appropriate time during the agent run by
    [`PendingMessageDrainCapability`][pydantic_ai.capabilities._pending_messages.PendingMessageDrainCapability].
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
                'Enqueued content must end with a `ModelRequest` (or `str` / `Sequence[UserContent]` / '
                '`ModelRequestPart` items that form one), so the agent has a request to respond to.'
            )
        return cls(messages=messages, priority=priority)
