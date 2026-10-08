"""Shared collection rules for the ComputerUse capability tests."""

from __future__ import annotations

import importlib.util

import pytest

# `LocalComputer` needs the `computer-use` extra, so slim CI runs (no extras) can't import its
# module. Ignore that file at collection; the capability itself is tested with a fake computer. A
# conditional expression rather than an `if` statement: branch coverage traces statement arcs, and
# no single environment can take both arms of an install-dependent branch.
collect_ignore = (
    ['test_local.py'] if any(importlib.util.find_spec(name) is None for name in ('mss', 'PIL', 'pynput')) else []
)


@pytest.fixture
def anyio_backend() -> str:
    """Asyncio only: `Agent.run` reaches `asyncio.create_task` for its lifecycle hooks."""
    return 'asyncio'
