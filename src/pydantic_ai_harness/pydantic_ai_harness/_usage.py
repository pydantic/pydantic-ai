"""Shared helpers for nested agent usage accounting."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TypeVar

from pydantic_ai.usage import RunUsage, UsageLimits

T = TypeVar('T')


def reserved_usage_limits(limits: UsageLimits | None) -> UsageLimits | None:
    """Reserve the pending parent request before a nested model call made from a hook.

    The hook may run after the parent request's limit check. Reducing a finite request limit
    prevents the nested call from spending the request that was already approved for the parent.
    """
    if limits is None or limits.request_limit is None:
        return limits
    return replace(limits, request_limit=max(0, limits.request_limit - 1))


async def run_with_count_tokens_fallback(
    run: Callable[[UsageLimits | None], Awaitable[T]],
    limits: UsageLimits | None,
    usage: RunUsage,
) -> T:
    """Run a nested agent under `limits`, dropping `count_tokens_before_request` if its model can't count tokens.

    A nested run can use a different model from the run whose limits it inherits, and
    `Model.count_tokens` raises `NotImplementedError` on models that don't implement it. There is no
    reliable way to ask a model up front: wrappers such as `FallbackModel` delegate counting, and a
    model given by name is only resolved inside `Agent.run`. So the run starts with the flag kept,
    which keeps checking each request ahead of time on models that can count, and runs once more
    with the flag cleared when it fails with `NotImplementedError` before any model request was
    counted in `usage`. The other limits still apply, checked against each response's usage.

    Once `usage.requests` has advanced, the error is raised as is: the run already got a response,
    so the error came from a later step (for example a tool), and running again would repeat it.

    Args:
        run: Starts the nested run with the given limits.
        limits: The limits the nested run inherits.
        usage: The usage the nested run counts its requests in.
    """
    if limits is None or not limits.count_tokens_before_request:
        return await run(limits)
    requests = usage.requests
    try:
        return await run(limits)
    except NotImplementedError:
        if usage.requests != requests:
            raise
    return await run(replace(limits, count_tokens_before_request=False))
