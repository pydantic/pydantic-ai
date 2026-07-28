"""Deprecation warning machinery for renamed modules and classes.

Used by the compatibility shims left behind by the capability naming pass: a renamed
module keeps a shim package at its old path, and a renamed class keeps a module-level
`__getattr__` alias, both emitting `HarnessDeprecationWarning` through these helpers.
"""

from __future__ import annotations

import warnings


class HarnessDeprecationWarning(UserWarning):
    """Warning emitted when a deprecated pydantic-ai-harness API is used.

    Inherits from `UserWarning` instead of `DeprecationWarning` so that deprecations are
    visible by default at runtime, matching Pydantic AI's `PydanticAIDeprecationWarning`.
    Silence every harness deprecation at once with::

        import warnings
        from pydantic_ai_harness import HarnessDeprecationWarning

        warnings.filterwarnings('ignore', category=HarnessDeprecationWarning)
    """


def warn_module_renamed(old: str, new: str) -> None:
    """Emit a `HarnessDeprecationWarning` that `pydantic_ai_harness.<old>` is now `pydantic_ai_harness.<new>`.

    Called at import time from the shim package left at the old module path, so existing
    imports keep working with a clear pointer to the new location.
    """
    warnings.warn(
        f'`pydantic_ai_harness.{old}` has been renamed to `pydantic_ai_harness.{new}`. '
        f'Update your imports; this compatibility shim will be removed in a future release.',
        category=HarnessDeprecationWarning,
        stacklevel=2,
    )


def warn_class_renamed(old: str, new: str, module: str) -> None:
    """Emit a `HarnessDeprecationWarning` that class `<module>.<old>` is now `<module>.<new>`.

    Called from a module-level `__getattr__` that resolves the old class name to the new
    class, so `isinstance` checks and existing imports keep working.
    """
    warnings.warn(
        f'`{module}.{old}` has been renamed to `{module}.{new}`. '
        f'Update your imports; this deprecated alias will be removed in a future release.',
        category=HarnessDeprecationWarning,
        stacklevel=3,
    )
