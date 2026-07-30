"""Validation for the [`ActivityConfig`][temporalio.workflow.ActivityConfig] dicts users hand us.

`ActivityConfig` is a `total=False` `TypedDict`, so an unknown key survives construction and only
fails once it's splatted into `workflow.start_activity()` — inside workflow code, where the
resulting `TypeError` isn't one of `PydanticAIPlugin`'s `workflow_failure_exception_types` and so
fails the workflow *task*, which Temporal retries forever. Validating the keys up front turns that
into a `UserError` (which does fail the workflow) at agent construction time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from difflib import get_close_matches
from typing import Any

from temporalio.workflow import ActivityConfig

from pydantic_ai.exceptions import UserError

_ACTIVITY_CONFIG_KEYS: tuple[str, ...] = tuple(sorted(ActivityConfig.__annotations__))
"""The keys `workflow.start_activity()` accepts, read off the installed `temporalio`."""


def describe_unknown(unknown: Iterable[object], valid: Sequence[str]) -> str:
    """Render unknown names as a message fragment, hinting at the closest valid name for each."""
    return ', '.join(
        f'{name!r} (did you mean {match[0]!r}?)'
        if isinstance(name, str) and (match := get_close_matches(name, valid, n=1))
        else repr(name)
        for name in unknown
    )


def validate_activity_config(config: Mapping[str, Any], source: str) -> None:
    """Raise a `UserError` if `config` holds keys that aren't `ActivityConfig` members.

    `source` names where the config came from, for example '`model_activity_config`'.
    """
    unknown = [key for key in config if key not in _ACTIVITY_CONFIG_KEYS]
    if not unknown:
        return
    raise UserError(
        f'Invalid `ActivityConfig` {"key" if len(unknown) == 1 else "keys"} in {source}: '
        f'{describe_unknown(unknown, _ACTIVITY_CONFIG_KEYS)}. '
        f'Valid keys are: {", ".join(map(repr, _ACTIVITY_CONFIG_KEYS))}.'
    )
