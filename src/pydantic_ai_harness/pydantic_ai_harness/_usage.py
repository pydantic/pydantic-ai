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


def forwarded_usage_limits(limits: UsageLimits | None) -> UsageLimits | None:
    """The parent's `UsageLimits` as a nested run started from a function tool should inherit them.

    Every ceiling carries over, which is what makes the budget tree-wide when the nested run
    shares the parent's `usage`. Two fields cannot pass through as-is:

    - `tool_calls_limit` is reduced by one. The tool wrapping the nested run is counted once it
      returns, not when it starts, so a nested run checking the raw limit spends a budget that does
      not yet include the call containing it, and the tree lands one over. `request_limit` never
      needs this: a function tool runs after the parent's request was made and counted.
    - `count_tokens_before_request` is dropped. It selects a request pipeline rather than setting a
      budget, and `Model.count_tokens` raises `NotImplementedError` on models that do not implement
      it. The nested run can be on a different model from the parent, so inheriting the flag would
      abort runs whose parent-side counting works. The cost is that token and cost ceilings are
      checked against a nested response rather than ahead of its request.
    """
    if limits is None:
        return None
    tool_calls_limit = limits.tool_calls_limit
    if tool_calls_limit is not None:
        tool_calls_limit = max(0, tool_calls_limit - 1)
    if tool_calls_limit == limits.tool_calls_limit and not limits.count_tokens_before_request:
        return limits
    return replace(limits, tool_calls_limit=tool_calls_limit, count_tokens_before_request=False)
