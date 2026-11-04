from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

_provider_field_warnings_suppressed: ContextVar[bool] = ContextVar('_provider_field_warnings_suppressed', default=False)


@contextmanager
def suppress_provider_field_warnings() -> Generator[None]:
    """Build native tools from values the user didn't set on the tool directly, without warning them.

    Uses a context variable rather than `warnings.catch_warnings()`, which changes global state and so isn't safe when
    runs happen concurrently.
    """
    token = _provider_field_warnings_suppressed.set(True)
    try:
        yield
    finally:
        _provider_field_warnings_suppressed.reset(token)


def provider_field_warnings_suppressed() -> bool:
    return _provider_field_warnings_suppressed.get()


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
