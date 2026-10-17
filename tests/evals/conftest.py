from __future__ import annotations

from collections.abc import AsyncIterator

import pytest


@pytest.fixture(autouse=True)
def _no_ci_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `Dataset.evaluate` span snapshots independent of the CI the suite runs in.

    CI runs in GitHub Actions, where `logfire.experiment.git.*`/`logfire.experiment.ci.*` attributes would
    otherwise be added to every experiment span. Tests that exercise detection set the variables they need.
    """
    for name in (
        'GITHUB_ACTIONS',
        'PYDANTIC_EVALS_CI_METADATA',
        'PYDANTIC_EVALS_GIT_COMMIT_SHA',
        'PYDANTIC_EVALS_GIT_BRANCH',
    ):
        monkeypatch.delenv(name, raising=False)


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
