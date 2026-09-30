from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .messages import ModelResponse
    from .usage import UsageLimits


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


class UsageNotReportedWarning(Warning):
    """Warning raised when token or cost limits are set but a model response reported no token usage.

    A provider-reported cost avoids this warning when only a cost limit is set.
    """


def warn_if_usage_not_reported(usage_limits: UsageLimits, response: ModelResponse) -> None:
    """Check a model-produced response before pricing or aggregating its cost."""
    usage = response.usage
    # Counted units such as web searches make `has_values()` true without reported token usage.
    if usage.total_tokens:
        return
    if usage_limits.has_token_limits() or (usage_limits.cost_limit is not None and usage.cost is None):
        warnings.warn(
            UsageNotReportedWarning(
                f'A token or cost limit is set, but the response from {response.model_name!r} reported no token '
                'usage, so it counts as zero tokens toward the limits. This usually means the provider or '
                'OpenAI-compatible server did not return a usage object.'
            ),
            stacklevel=2,
        )
