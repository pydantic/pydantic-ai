"""`SlidingWindowCompaction` -- zero-cost trimming of the oldest messages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.compaction._shared import (
    compact_with_span,
    context_for_request,
    exceeds,
    find_safe_cutoff,
    find_token_cutoff,
    prepend_first_user_message,
    resolve_token_trigger,
    validate_token_trigger,
)

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


@dataclass
class SlidingWindowCompaction(AbstractCapability[AgentDepsT]):
    """Zero-cost sliding-window trimmer.

    When the conversation exceeds a configurable threshold (message count or
    estimated token count), the oldest messages are discarded while preserving
    tool-call / tool-return pairs.  No LLM calls are made.

    Trimming happens in ``before_model_request`` so it is transparent to the
    rest of the agent run.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.compaction import SlidingWindowCompaction

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[SlidingWindowCompaction(max_messages=80, keep_messages=40)],
        )
        ```
    """

    max_messages: int | None = None
    """Trigger trimming when message count exceeds this value. ``None`` disables."""

    max_tokens: int | None = None
    """Trigger trimming when estimated token count exceeds this value. ``None`` disables."""

    max_fraction: float | None = field(default=None, kw_only=True)
    """Trigger when estimated tokens exceed this fraction of the model's context window.

    Resolved per request from the request's model, so one setting behaves correctly on any
    model. Mutually exclusive with `max_tokens`."""

    context_window: int | None = field(default=None, kw_only=True)
    """Window override in tokens. `None` resolves it from the request's model.

    Unlike `fallback_context_window`, this applies whether or not resolution succeeds. Reach
    for it when the registry is confidently wrong: a beta- or tier-gated window it records as
    the maximum, or a self-hosted endpoint whose model id describes someone else's
    deployment. Only consulted alongside `max_fraction`."""

    fallback_context_window: int = field(default=DEFAULT_CONTEXT_WINDOW, kw_only=True)
    """Window assumed when the request's model is not in the pricing registry.

    Only consulted alongside `max_fraction`. Supply the real number for a deployment the
    registry cannot resolve."""

    keep_messages: int = 40
    """Number of tail messages to retain after trimming (message-count trigger)."""

    keep_tokens: int | None = None
    """Target token budget after trimming (token-count trigger).

    When ``None``, falls back to ``keep_messages``.
    """

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer for accurate token counting.

    A callable that returns the token count for a given string.
    When ``None``, uses a ~4 characters-per-token heuristic.
    """

    preserve_first_user_message: bool = True
    """When ``True``, the first ``ModelRequest`` containing a ``UserPromptPart``
    is always kept after trimming, in addition to system prompts.
    """

    def __post_init__(self) -> None:
        if self.max_messages is None and self.max_tokens is None and self.max_fraction is None:
            raise ValueError('At least one of max_messages, max_tokens, or max_fraction must be set.')
        if self.max_messages is not None and self.max_messages < 1:
            raise ValueError('max_messages must be positive.')
        validate_token_trigger(self.max_tokens, self.max_fraction, self.fallback_context_window, self.context_window)
        if self.keep_messages < 0:
            raise ValueError('keep_messages must be non-negative.')
        if self.keep_tokens is not None and self.keep_tokens < 0:
            raise ValueError('keep_tokens must be non-negative.')

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Drop the oldest messages down to the configured tail."""
        if self.keep_tokens is not None:
            cutoff = find_token_cutoff(messages, self.keep_tokens, self.tokenizer)
        else:
            cutoff = find_safe_cutoff(messages, self.keep_messages)

        if cutoff <= 0:
            return messages

        trimmed = messages[cutoff:]
        if self.preserve_first_user_message:
            trimmed = prepend_first_user_message(messages, cutoff, trimmed)
        return trimmed

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Trim the message list if it exceeds the configured threshold."""
        messages: list[ModelMessage] = list(request_context.messages)
        request_ctx = context_for_request(ctx, request_context)
        token_trigger = resolve_token_trigger(
            self.max_tokens, self.max_fraction, request_ctx.model, self.fallback_context_window, self.context_window
        )
        if not exceeds(messages, self.max_messages, token_trigger, self.tokenizer):
            return request_context
        request_context.messages = await compact_with_span(
            request_ctx,
            strategy='SlidingWindowCompaction',
            messages=messages,
            compact=lambda: self.compact(messages, request_ctx),
            tokenizer=self.tokenizer,
        )
        return request_context
