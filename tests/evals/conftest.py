from __future__ import annotations

from collections.abc import AsyncIterator

import pytest


@pytest.fixture(autouse=True)
async def _cleanup_background_evaluations() -> AsyncIterator[None]:
    """Drain background evaluation tasks after each test.

    Prevents leaked tasks from a failed test from affecting subsequent tests.
    """
    yield
    try:
        from pydantic_evals.online import wait_for_evaluations
    except ImportError:
        return
    await wait_for_evaluations()
