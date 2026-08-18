"""Shared prompt-caching helpers used by the base `Model` and the provider model classes."""

from __future__ import annotations as _annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

from ..exceptions import UserError
from ..settings import CacheRetention, CacheSetting

T = TypeVar('T')

CACHE_RETENTION_ORDER: tuple[CacheRetention, ...] = ('5m', '30m', '1h')
"""All retention tiers, shortest first."""


def snap_cache_retention(value: CacheSetting, supported: Sequence[CacheRetention]) -> CacheSetting:
    """Snap a requested cache retention to the nearest tier the provider supports.

    Booleans and supported retentions pass through unchanged. An unsupported retention snaps
    down to the nearest shorter supported tier, or up to the shortest supported tier when no
    shorter one exists.
    """
    if isinstance(value, bool):
        return value
    if not supported or value in supported:
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


def warn_cache_point_ignored(description: str, hint: str | None = None) -> None:
    """Warn that a `CachePoint` marker was dropped from the request.

    The default warnings filter deduplicates by call site, so each dropping code path
    warns at most once per process.
    """
    message = f'`CachePoint` is not supported by {description} and was ignored.'
    if hint:
        message = f'{message} {hint}'
    warnings.warn(message, UserWarning, stacklevel=2)
