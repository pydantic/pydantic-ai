from __future__ import annotations

import warnings
from collections.abc import Callable


class PydanticAIDeprecationWarning(UserWarning):
    """Warning emitted when a deprecated Pydantic AI API is used.

    Inherits from `UserWarning` instead of `DeprecationWarning` so that
    deprecations are visible by default at runtime, following the approach
    described in https://sethmlarson.dev/deprecations-via-warnings-dont-work-for-python-libraries.
    """


def warn_on_prepare_callback_returned_none(prepare_func: Callable[..., object]) -> None:
    """Warn that a prepare callback returned `None`."""
    name = getattr(prepare_func, '__name__', str(prepare_func))
    warnings.warn(
        f'prepare callback {name!r} returned `None`; '
        'this hides all tool definitions passed to it for this step and will raise in v2.0. '
        'Return `[]` to hide them explicitly, or `tool_defs` to pass them through unchanged.',
        PydanticAIDeprecationWarning,
        stacklevel=3,
    )
