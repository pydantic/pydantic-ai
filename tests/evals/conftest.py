from __future__ import annotations

from collections.abc import AsyncIterator

import pytest


@pytest.fixture(autouse=True)
def _no_ci_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `Dataset.evaluate` span snapshots independent of the CI the suite runs in.

    Detection would otherwise add `vcs.*`/`cicd.*` attributes to every experiment span, from GitHub Actions in CI
    and from the repository's own `git` checkout everywhere. Tests that exercise detection set what they need.
    """
    for name in (
        'GITHUB_ACTIONS',
        'GITLAB_CI',
        'PYDANTIC_EVALS_CI_METADATA',
        'PYDANTIC_EVALS_GIT_COMMIT_SHA',
        'PYDANTIC_EVALS_GIT_BRANCH',
    ):
        monkeypatch.delenv(name, raising=False)
    # Points `git` at no repository, so reading the checkout finds nothing.
    monkeypatch.setenv('GIT_DIR', '/nonexistent')


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
