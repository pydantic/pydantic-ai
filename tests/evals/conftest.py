from __future__ import annotations

from collections.abc import AsyncIterator

import pytest


@pytest.fixture(autouse=True)
def _no_ci_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `Dataset.evaluate` span snapshots independent of the CI the suite runs in.

    Detection is turned off, so no test depends on GitHub Actions variables.
    `test_experiment_provenance.py` turns it back on for the tests that exercise it.
    """
    monkeypatch.setenv('PYDANTIC_EVALS_CI_PROVENANCE', 'false')


@pytest.fixture(autouse=True)
async def _cleanup_background_evaluations() -> AsyncIterator[None]:
    """Drain background evaluation tasks after each test.

    Prevents leaked tasks from a failed test from affecting subsequent tests.
    """
    yield
    try:
        from pydantic_evals.online import wait_for_evaluations
    except ImportError:  # pragma: no cover
        return
    await wait_for_evaluations()
