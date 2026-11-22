"""Deprecated import location for `pydantic_ai_harness.graceful_limits`.

`WrapUpAtLimits` was renamed to `GracefulLimits`. Importing from here still works but emits a
`HarnessDeprecationWarning`. Import from `pydantic_ai_harness.graceful_limits` instead.
"""

from pydantic_ai_harness._warn import warn_module_renamed
from pydantic_ai_harness.graceful_limits import DEFAULT_WRAP_UP_PROMPT, LimitName, WrapUpPart, wrapped_up_by
from pydantic_ai_harness.graceful_limits._deprecated import WrapUpAtLimits  # pyright: ignore[reportDeprecated]

warn_module_renamed(
    'wrap_up_at_limits',
    'graceful_limits',
    reason='`WrapUpAtLimits` is now `GracefulLimits`; `WrapUpPart`, `wrapped_up_by`, `LimitName` and '
    '`DEFAULT_WRAP_UP_PROMPT` keep their names.',
)

__all__ = ['DEFAULT_WRAP_UP_PROMPT', 'LimitName', 'WrapUpAtLimits', 'WrapUpPart', 'wrapped_up_by']
