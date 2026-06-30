"""`ClearToolResults` -- zero-cost in-place clearing of old tool results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.experimental.compaction._shared import (
    compact_with_span,
    estimate_token_count,
    exceeds,
    iter_tool_pairs,
    rebuild_with_cleared,
)

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


@dataclass
class ClearToolResults(AbstractCapability[AgentDepsT]):
    """Zero-cost in-place clearing of old tool results.

    Replaces the content of the oldest tool *results* with a short placeholder while
    keeping the most recent ``keep_pairs`` tool-call / tool-return pairs intact.  Tool
    calls remain paired with their (now-blanked) results, so the history stays valid.
    No LLM calls are made.

    This is the cheap first tier of compaction -- tool results typically dominate
    context, and the agent can re-run a tool if it needs the data again.

    Cache tradeoff: clearing rewrites message content, which invalidates the provider's
    prompt cache from the clear point onward (the next request pays a cache-write).  Use
    ``min_clear_tokens`` to skip clearing that reclaims too little to be worth busting the
    cache.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.experimental.compaction import ClearToolResults

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[ClearToolResults(max_tokens=100_000, keep_pairs=3)],
        )
        ```
    """

    max_messages: int | None = None
    """Trigger clearing when message count reaches this value. ``None`` disables."""

    max_tokens: int | None = None
    """Trigger clearing when estimated token count reaches this value. ``None`` disables."""

    keep_pairs: int = 3
    """Number of most-recent tool-call / tool-return pairs left untouched."""

    placeholder: str = '[tool result cleared]'
    """Replacement content for a cleared tool result."""

    exclude_tools: frozenset[str] = frozenset()
    """Tool names whose results are never cleared."""

    clear_tool_inputs: bool = False
    """When ``True``, also blank the arguments of the cleared tool calls."""

    min_clear_tokens: int | None = None
    """Only clear if doing so reclaims at least this many estimated tokens.

    Protects the prompt cache from being invalidated for a trivial gain. ``None`` always clears.
    """

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer for accurate token counting.

    A callable that returns the token count for a given string.
    When ``None``, uses a ~4 characters-per-token heuristic.
    """

    def __post_init__(self) -> None:
        if self.max_messages is None and self.max_tokens is None:
            raise ValueError('At least one of max_messages or max_tokens must be set.')
        if self.max_messages is not None and self.max_messages < 1:
            raise ValueError('max_messages must be positive.')
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError('max_tokens must be positive.')
        if self.keep_pairs < 0:
            raise ValueError('keep_pairs must be non-negative.')
        if self.min_clear_tokens is not None and self.min_clear_tokens < 0:
            raise ValueError('min_clear_tokens must be non-negative.')

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Blank the oldest tool results beyond the most recent ``keep_pairs``."""
        pairs = iter_tool_pairs(messages)
        clearable = pairs[: max(0, len(pairs) - self.keep_pairs)]

        clear_return_ids: set[str] = set()
        clear_input_ids: set[str] = set()
        for pair in clearable:
            if pair.tool_name in self.exclude_tools:
                continue
            clear_return_ids.add(pair.tool_call_id)
            if self.clear_tool_inputs:
                clear_input_ids.add(pair.tool_call_id)

        if not clear_return_ids:
            return messages

        cleared = rebuild_with_cleared(messages, clear_return_ids, clear_input_ids, self.placeholder)
        if self.min_clear_tokens is not None:
            reclaimed = estimate_token_count(messages, self.tokenizer) - estimate_token_count(cleared, self.tokenizer)
            if reclaimed < self.min_clear_tokens:
                return messages
        return cleared

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Clear old tool results if the conversation exceeds the configured threshold."""
        messages: list[ModelMessage] = list(request_context.messages)
        if not exceeds(messages, self.max_messages, self.max_tokens, self.tokenizer):
            return request_context
        request_context.messages = await compact_with_span(
            ctx,
            strategy='ClearToolResults',
            messages=messages,
            compact=lambda: self.compact(messages, ctx),
            tokenizer=self.tokenizer,
        )
        return request_context
