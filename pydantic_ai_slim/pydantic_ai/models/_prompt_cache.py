"""Shared prompt-caching helpers used by the base `Model` and the provider model classes."""

from __future__ import annotations as _annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Literal, TypeVar

from ..exceptions import UserError
from ..settings import CacheRetention

T = TypeVar('T')

CACHE_RETENTION_ORDER: tuple[CacheRetention, ...] = ('5m', '30m', '1h')
"""All retention tiers, shortest first."""


def snap_cache_retention(
    value: Literal[True] | CacheRetention, supported: Sequence[CacheRetention]
) -> Literal[True] | CacheRetention:
    """Snap a requested cache retention to the nearest tier the provider supports.

    `True` and supported retentions pass through unchanged. An unsupported retention snaps
    down to the nearest shorter supported tier, or up to the shortest supported tier when no
    shorter one exists. On a provider with no retention tiers to request, a retention becomes
    `True`: caching with the provider's default retention.
    """
    if isinstance(value, bool):
        return value
    if value not in CACHE_RETENTION_ORDER:
        raise UserError(
            f'Unknown `cache` retention {value!r}. '
            f'Use `True`, `False`, or one of {", ".join(repr(tier) for tier in CACHE_RETENTION_ORDER)}.'
        )
    if not supported:
        return True
    if value in supported:
        return value
    rank = CACHE_RETENTION_ORDER.index(value)
    supported_ranks = sorted(CACHE_RETENTION_ORDER.index(tier) for tier in supported)
    shorter = [supported_rank for supported_rank in supported_ranks if supported_rank < rank]
    return CACHE_RETENTION_ORDER[shorter[-1] if shorter else supported_ranks[0]]


def excess_cache_points(
    blocks_newest_first: Iterable[T],
    *,
    max_points: int,
    reserved: int,
    is_cache_point: Callable[[T], bool],
    description: str,
) -> list[T]:
    """Return the cache-point blocks that exceed the provider's per-request limit.

    `reserved` counts cache points outside `blocks_newest_first` (system prompt, tool
    definitions, a server-managed automatic breakpoint) that always take priority.
    The remaining budget goes to the newest message cache points; the returned excess
    blocks are the oldest ones, for the caller to strip in its own wire format.

    Raises:
        UserError: If `reserved` alone already exceeds `max_points`.
    """
    budget = max_points - reserved
    if budget < 0:
        raise UserError(
            f'Too many cache points for {description}. '
            f'System prompt and tool definitions already use {reserved} cache points, '
            f'which exceeds the maximum of {max_points}.'
        )
    excess: list[T] = []
    for block in blocks_newest_first:
        if is_cache_point(block):
            if budget > 0:
                budget -= 1
            else:
                excess.append(block)
    return excess


LOOKBACK_SAFE_BLOCKS = 18
"""How many content blocks a moving message breakpoint can safely move past the previous request's.

To read a cached prefix, the provider looks back from each cache breakpoint for an earlier request's cache
entry, but only so far: about 20 content blocks on Amazon Bedrock and on the Claude API (where the Claude API
collapses a run of consecutive `tool_use` or `tool_result` blocks into one position, but Bedrock doesn't).
A turn that adds more blocks than that, such as one with a dozen parallel tool calls and their results, would
otherwise write the whole conversation again instead of reading it.
https://github.com/pydantic/pydantic-ai/issues/9404
"""


def previous_tail_needing_breakpoint(roles: Sequence[str], block_counts: Sequence[int]) -> int | None:
    """The index of the message that ended the previous request, if the next breakpoint is out of its lookback.

    A library-placed history breakpoint sits at the end of the last message, so the previous request put its
    breakpoint at the end of the last user-side message before the latest assistant message. When more than
    `LOOKBACK_SAFE_BLOCKS` content blocks follow it, the caller adds a breakpoint there too, so the earlier cache write is an explicit breakpoint rather than
    something the lookback has to reach.

    Args:
        roles: Each wire message's role, oldest first. Anything but `'assistant'` (tool results included) counts as
            the user side.
        block_counts: Each wire message's number of content blocks (tool calls included).
    """
    last_assistant = next((i for i in range(len(roles) - 1, -1, -1) if roles[i] == 'assistant'), None)
    # A history that ends with an assistant turn (a prefill) has no newer user-side tail to move past it.
    if last_assistant is None or last_assistant == len(roles) - 1:
        return None
    previous_tail = next((i for i in range(last_assistant - 1, -1, -1) if roles[i] != 'assistant'), None)
    if previous_tail is None or sum(block_counts[previous_tail + 1 :]) < LOOKBACK_SAFE_BLOCKS:
        return None
    return previous_tail
