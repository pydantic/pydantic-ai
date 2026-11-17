"""The installed `pydantic-ai-harness` distribution, as client attribution metadata names it."""

from __future__ import annotations

from functools import cache
from importlib.metadata import PackageNotFoundError, version

HARNESS_DISTRIBUTION = 'pydantic-ai-harness'


@cache
def harness_version() -> str:
    """Return the installed `pydantic-ai-harness` version, or `'unknown'` when its package metadata is missing."""
    try:
        return version(HARNESS_DISTRIBUTION)
    except PackageNotFoundError:  # pragma: no cover
        return 'unknown'
