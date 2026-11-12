"""Deprecation warning machinery for renamed APIs and changed defaults.

Used by the compatibility shims left behind by the capability naming pass: a renamed
module keeps a shim package at its old path, and a renamed class keeps a module-level
`__getattr__` alias, both emitting `HarnessDeprecationWarning` through these helpers.
`warn_default_changed` covers an option whose default moved, where existing callers keep
working but get different behavior, `warn_argument_ignored` an argument that is still
accepted but no longer does anything, `warn_argument_renamed` an argument still accepted
under its old name, `warn_callable_renamed` a method still callable under its old name, and
`renamed_duration_limit` a `resource_limits` dict still using the `max_duration_secs` key.
"""

from __future__ import annotations

import warnings
from copy import copy
from typing import Literal, TypeVar

from typing_extensions import TypedDict


class HarnessDeprecationWarning(UserWarning):
    """Warning emitted when a deprecated pydantic-ai-harness API is used.

    Inherits from `UserWarning` instead of `DeprecationWarning` so that deprecations are
    visible by default at runtime, matching Pydantic AI's `PydanticAIDeprecationWarning`.
    Silence every harness deprecation at once with:

    ```python
    import warnings
    from pydantic_ai_harness import HarnessDeprecationWarning

    warnings.filterwarnings('ignore', category=HarnessDeprecationWarning)
    ```
    """


class MCPReadOnlyNoToolsWarning(UserWarning):
    """`read_only=True` removed every tool from a hosted MCP server, so the agent gets none of its tools.

    The filter keeps only tools the server annotates with `readOnlyHint: true`, and some servers
    publish no annotations at all. Disable `read_only`, or filter this category when an empty
    toolset is intended.
    """


def warn_module_renamed(old: str, new: str, *, reason: str = '') -> None:
    """Emit a `HarnessDeprecationWarning` that `pydantic_ai_harness.<old>` is now `pydantic_ai_harness.<new>`.

    Called at import time from the shim package left at the old module path, so existing
    imports keep working with a clear pointer to the new location. `reason`, one or more
    sentences, says why, when the new name alone doesn't.
    """
    message = f'`pydantic_ai_harness.{old}` has been renamed to `pydantic_ai_harness.{new}`.'
    if reason:
        message += f' {reason}'
    warnings.warn(
        f'{message} Update your imports; this compatibility shim will be removed in a future release.',
        category=HarnessDeprecationWarning,
        stacklevel=2,
    )


def warn_default_changed(
    *, owner: str, option: str, old: object, new: object, impact: str, stacklevel: int = 4
) -> None:
    """Emit a `HarnessDeprecationWarning` that `<owner>`'s `<option>` default changed.

    For an option whose default moved to a value that changes behavior rather than breaking
    the call: existing code still runs, so nothing surfaces unless it is said out loud. Call
    this only when the caller left the option unset, so an explicit choice of either value
    stays silent, and call it once per construction rather than per use.

    `impact` states what the new default does differently; the rest of the message names the
    value that restores the old behavior and the value that keeps the new one without warning.
    `stacklevel` defaults to reporting the caller of a dataclass `__post_init__`.
    """
    warnings.warn(
        f'`{owner}` now defaults to `{option}={new!r}`; it previously defaulted to `{option}={old!r}`. '
        f'{impact} '
        f'Pass `{option}={old!r}` to restore the previous behavior, or `{option}={new!r}` to keep '
        f'the new one and silence this warning.',
        category=HarnessDeprecationWarning,
        stacklevel=stacklevel,
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


def warn_argument_renamed(owner: str, old: str, new: str, *, stacklevel: int = 3) -> None:
    """Emit a `HarnessDeprecationWarning` that `<owner>(<old>=...)` is now `<owner>(<new>=...)`.

    For an argument accepted under its old name as an alias of the new one.
    """
    warnings.warn(
        f'`{owner}({old}=...)` has been renamed to `{owner}({new}=...)`. '
        'Update the call; this deprecated alias will be removed in a future release.',
        category=HarnessDeprecationWarning,
        stacklevel=stacklevel,
    )


def warn_callable_renamed(old: str, new: str) -> None:
    """Emit a `HarnessDeprecationWarning` that `<old>()` is now `<new>()`.

    Called from the thin wrapper left under the old method name.
    """
    warnings.warn(
        f'`{old}()` has been renamed to `{new}()`. '
        'Update the call; this deprecated alias will be removed in a future release.',
        category=HarnessDeprecationWarning,
        stacklevel=3,
    )


SET_WORKING_DIR_ON_THE_WORKSPACE = (
    "commands/paths start in the workspace's working directory; set it on the workspace, e.g. "
    "`LocalWorkspace('./repo')`."
)
"""The fix for a removed working-directory argument, for `warn_argument_ignored`."""


def warn_argument_ignored(owner: str, argument: str, fix: str, *, stacklevel: int = 4) -> None:
    """Emit a `HarnessDeprecationWarning` that `<owner>(<argument>=...)` is deprecated and now has no effect.

    For an argument whose job moved elsewhere: the value is accepted so existing code still
    constructs, but it is ignored, so the warning has to say what to do instead. `fix` is that
    instruction. `stacklevel` defaults to reporting the caller of a dataclass `__post_init__`.
    """
    warnings.warn(
        f'`{owner}({argument}=...)` is deprecated and ignored: {fix}',
        category=HarnessDeprecationWarning,
        stacklevel=stacklevel,
    )


class _DurationLimit(TypedDict, total=False):
    max_duration_seconds: float
    max_duration_secs: float


_LimitsT = TypeVar('_LimitsT', bound=_DurationLimit)


def renamed_duration_limit(
    limits: _LimitsT | Literal['unlimited'] | None, owner: str, *, stacklevel: int = 4
) -> _LimitsT | Literal['unlimited'] | None:
    """Return `limits` with a deprecated `max_duration_secs` key moved to `max_duration_seconds`, warning about it.

    Called from the `__post_init__` of the dataclasses that take `resource_limits`, which store the result
    so that their per-run copies do not warn again.
    """
    if limits is None or limits == 'unlimited' or 'max_duration_secs' not in limits:
        return limits
    if 'max_duration_seconds' in limits:
        raise TypeError('Pass `max_duration_seconds` only: `max_duration_secs` is its deprecated name.')
    warnings.warn(
        f"`{owner}(resource_limits={{'max_duration_secs': ...}})` has been renamed to "
        f"`{owner}(resource_limits={{'max_duration_seconds': ...}})`. "
        'Update the call; this deprecated alias will be removed in a future release.',
        category=HarnessDeprecationWarning,
        stacklevel=stacklevel,
    )
    renamed = copy(limits)
    renamed['max_duration_seconds'] = renamed.pop('max_duration_secs')
    return renamed
