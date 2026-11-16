"""End a run with a final answer instead of `UsageLimitExceeded` when a usage limit is reached."""

from pydantic_ai_harness.wrap_up_at_limits._capability import (
    DEFAULT_WRAP_UP_PROMPT,
    NOT_EXECUTED,
    LimitName,
    WrapUpAtLimits,
    wrapped_up_by,
)

__all__ = ['DEFAULT_WRAP_UP_PROMPT', 'NOT_EXECUTED', 'LimitName', 'WrapUpAtLimits', 'wrapped_up_by']
