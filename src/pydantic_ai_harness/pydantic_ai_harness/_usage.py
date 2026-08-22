"""Shared helpers for nested agent usage accounting."""

from __future__ import annotations

from dataclasses import replace

from pydantic_ai.usage import UsageLimits


def reserved_usage_limits(limits: UsageLimits | None) -> UsageLimits | None:
    """Reserve the pending parent request before a nested model call made from a hook.

    The hook may run after the parent request's limit check. Reducing a finite request limit
    prevents the nested call from spending the request that was already approved for the parent.
    """
    if limits is None or limits.request_limit is None:
        return limits
    return replace(limits, request_limit=max(0, limits.request_limit - 1))
