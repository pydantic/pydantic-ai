from __future__ import annotations

import warnings
from collections.abc import Callable

from pydantic_graph.util import get_callable_name


class PydanticAIDeprecationWarning(UserWarning):
    """Warning emitted when a deprecated Pydantic AI API is used.

    Inherits from `UserWarning` instead of `DeprecationWarning` so that
    deprecations are visible by default at runtime, following the approach
    described in https://sethmlarson.dev/deprecations-via-warnings-dont-work-for-python-libraries.
    """


def warn_on_prepare_callback_returned_none(prepare_func: Callable[..., object]) -> None:
    """Warn that a prepare callback returned `None`."""
    warnings.warn(
        f'prepare callback {get_callable_name(prepare_func)!r} returned `None`; '
        'returning `None` from a prepare callback is deprecated and hides all tool definitions passed to it for this step. '
        'Return `[]` to hide them explicitly, or `tool_defs` to pass them through unchanged.',
        PydanticAIDeprecationWarning,
        stacklevel=3,
    )
