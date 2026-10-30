from __future__ import annotations

import inspect
from collections.abc import Container


def get_warning_stacklevel(
    *, skip_packages: Container[str] = ('pydantic_ai',), skip_modules: Container[str] = ()
) -> int:
    """Return the `warnings.warn` stacklevel from the caller to the first unskipped frame."""
    frame = inspect.currentframe()
    if frame is not None:
        frame = frame.f_back
    stacklevel = 1
    try:
        while frame is not None:
            module = frame.f_globals.get('__name__', '')
            if module not in skip_modules and module.partition('.')[0] not in skip_packages:
                break
            frame = frame.f_back
            stacklevel += 1
        return stacklevel
    finally:
        del frame


class PydanticAIDeprecationWarning(UserWarning):
    """Warning emitted when a deprecated Pydantic AI API is used.

    Inherits from `UserWarning` instead of `DeprecationWarning` so that
    deprecations are visible by default at runtime, following the approach
    described in https://sethmlarson.dev/deprecations-via-warnings-dont-work-for-python-libraries.
    """


class CostCalculationFailedWarning(Warning):
    """Warning raised when cost calculation fails."""


class UsageExtractionFailedWarning(Warning):
    """Warning raised when usage extraction fails."""


class CostNotFoundWarning(Warning):
    """Warning raised when cost is not found."""


class UsageLimitUnavailableWarning(Warning):
    """Warning raised when some responses omit usage while a token or cost limit is configured."""
