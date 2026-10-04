"""Minimum-token suffix selection shared by summarizing and sliding-window compaction."""

from __future__ import annotations

from collections.abc import Callable

from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.compaction._shared import collect_message_text, estimate_text_tokens, find_safe_cutoff


def validate_min_keep_tokens(min_keep_tokens: int | None, keep_tokens: int | None) -> None:
    if min_keep_tokens is None:
        return
    if min_keep_tokens <= 0:
        raise ValueError('min_keep_tokens must be positive.')
    if keep_tokens is not None:
        raise ValueError('min_keep_tokens and keep_tokens are mutually exclusive.')


def _estimate_retained_text_tokens(messages: list[ModelMessage], tokenizer: Callable[[str], int] | None) -> int:
    """Count message-part text only, without attached request instructions."""
    segments = collect_message_text(messages)
    if tokenizer is not None:
        return sum(tokenizer(segment) for segment in segments)
    # Round once for the whole suffix, not separately for each segment.
    return estimate_text_tokens(''.join(segments))


def find_minimum_token_cutoff(
    messages: list[ModelMessage],
    min_tokens: int,
    tokenizer: Callable[[str], int] | None,
) -> int:
    """Keep the shortest whole-message suffix reaching `min_tokens`, then protect tool pairs.

    Counts include message-part text (including system prompts), not attached instructions.
    If the entire history is below the minimum, there is no prefix to compact.
    Whole messages and tool pairs can make the
    retained suffix exceed the minimum; this is not a model context-window limit.
    """
    if _estimate_retained_text_tokens(messages, tokenizer) <= min_tokens:
        return 0

    lo, hi = 0, len(messages)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _estimate_retained_text_tokens(messages[mid:], tokenizer) >= min_tokens:
            lo = mid
        else:
            hi = mid

    # A long-running tool may return more than the default search range after its call.
    return find_safe_cutoff(messages, len(messages) - lo, search_range=len(messages), include_tool_retries=True)
