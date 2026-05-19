"""Internal helpers for the `RunContext.enqueue` / `AgentRun.enqueue` APIs.

These types live here (rather than in `messages.py`) because they're internal runtime
state for the pending message queue, not part of the wire-serializable message history.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from .messages import ModelRequest, UserPromptPart

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


EnqueueContent: TypeAlias = 'str | Sequence[UserContent] | ModelRequest'
"""A single item accepted by [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue]
and [`AgentRun.enqueue`][pydantic_ai.run.AgentRun.enqueue].

- `str` or `Sequence[UserContent]`: wrapped in a [`UserPromptPart`][pydantic_ai.messages.UserPromptPart]
    (matching the shape of `Agent.run(user_prompt=...)`).
- [`ModelRequest`][pydantic_ai.messages.ModelRequest]: emitted verbatim as its own message at
    drain time — pass a complete request when you need to control `instructions`, `metadata`,
    or other request-level fields, or when you need a part type other than
    [`UserPromptPart`][pydantic_ai.messages.UserPromptPart]. Must be the only positional argument to `enqueue`.

[`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart] is intentionally excluded:
Anthropic and Google hoist any `SystemPromptPart` (regardless of position) to their
top-level system parameter, which invalidates prefix cache and loses positional intent.
If you really need to inject a `SystemPromptPart` mid-run, wrap it in a `ModelRequest` and
pass that as the payload — but be aware of the cross-provider behavior; the framework-level
fix is tracked in [#5437](https://github.com/pydantic/pydantic-ai/issues/5437) and will let
us re-add direct `SystemPromptPart` / `Sequence[ModelRequestPart]` support to `EnqueueContent`.
"""


def build_enqueue_request(items: Sequence[EnqueueContent]) -> ModelRequest | None:
    """Build a single [`ModelRequest`][pydantic_ai.messages.ModelRequest] from a sequence of enqueue items.

    Returns the input [`ModelRequest`][pydantic_ai.messages.ModelRequest] unchanged when a single
    one was passed (passthrough — preserves `instructions`, `metadata`, and other request-level fields).
    Otherwise wraps each item in a [`UserPromptPart`][pydantic_ai.messages.UserPromptPart] and packs
    them into one new request.

    Returns `None` when there are no items (the caller can skip enqueueing — empty enqueues
    are a no-op rather than an error).

    Used internally by [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue] and
    [`AgentRun.enqueue`][pydantic_ai.run.AgentRun.enqueue].
    """
    if not items:
        return None
    if len(items) == 1 and isinstance(items[0], ModelRequest):
        return items[0]
    if any(isinstance(item, ModelRequest) for item in items):
        raise ValueError('ModelRequest must be enqueued alone, not mixed with strings or `Sequence[UserContent]` items')
    return ModelRequest(parts=[UserPromptPart(content=item) for item in items])  # type: ignore[arg-type]


@dataclass
class PendingMessage:
    """A [`ModelRequest`][pydantic_ai.messages.ModelRequest] queued for injection into the agent conversation.

    Enqueued via [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue] or
    [`AgentRun.enqueue`][pydantic_ai.run.AgentRun.enqueue] and automatically drained
    at the appropriate time during the agent run by
    [`PendingMessageDrainCapability`][pydantic_ai.capabilities._pending_messages.PendingMessageDrainCapability].
    """

    request: ModelRequest
    """The request to inject into the conversation."""

    priority: PendingMessagePriority = 'asap'
    """When to deliver this message:

    - `'asap'`: at the earliest opportunity (next model request, or redirect if the agent
        would otherwise terminate).
    - `'when_idle'`: only when the agent would otherwise terminate, after `'asap'` messages.
    """
