"""Shared collection rules for the Day AI capability tests."""

from __future__ import annotations

import importlib.util

import pytest

# `mcp` and `fastmcp` are gated on the `day-ai` extra, so slim CI runs (no extras) can't import
# these modules. Ignore them at collection; `test_packaging.py` stays collected
# because it checks package metadata, which holds on base installs too.
# A conditional expression rather than an `if` statement: branch coverage traces
# statement arcs, and no single environment can take both arms of an
# install-dependent branch.
_REQUIRED = ('mcp', 'fastmcp', 'pydantic_ai.mcp')
collect_ignore = ['test_day_ai.py'] if any(importlib.util.find_spec(name) is None for name in _REQUIRED) else []


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'
