"""Experimental-feature warning machinery for pydantic-ai-harness."""

from __future__ import annotations

import warnings


class HarnessExperimentalWarning(UserWarning):
    """Signals that a pydantic-ai-harness feature is experimental.

    Experimental features may change or be removed in any release, without a deprecation
    period.  Silence every experimental-harness warning at once with::

        import warnings
        from pydantic_ai_harness.experimental import HarnessExperimentalWarning

        warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)
    """


_SILENCE_HINT = (
    '    import warnings\n'
    '    from pydantic_ai_harness.experimental import HarnessExperimentalWarning\n'
    "    warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)"
)


def warn_experimental(feature: str) -> None:
    """Emit a `HarnessExperimentalWarning` for *feature*, including how to silence all of them.

    One filter silences the whole category (every experimental capability), so users never
    need a suppression line per capability.
    """
    warnings.warn(
        f'`pydantic_ai_harness.experimental.{feature}` is experimental: its API may change or be '
        f'removed in any release, without a deprecation period.\n\n'
        f'Silence all pydantic-ai-harness experimental warnings with:\n\n{_SILENCE_HINT}\n',
        category=HarnessExperimentalWarning,
        stacklevel=2,
    )


def warn_moved(old: str, new: str) -> None:
    """Emit a `DeprecationWarning` that `experimental.old` now lives at top-level `new`.

    Left behind at each old `pydantic_ai_harness.experimental.<name>` path when a capability
    graduates out of `experimental`, so existing imports keep working with a clear pointer to
    the new location.
    """
    warnings.warn(
        f'`pydantic_ai_harness.experimental.{old}` has moved to `pydantic_ai_harness.{new}`. '
        f'Update your imports; this compatibility shim will be removed in a future release.',
        category=DeprecationWarning,
        stacklevel=2,
    )
