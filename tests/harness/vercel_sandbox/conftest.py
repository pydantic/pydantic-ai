"""Shared fixtures for VercelSandbox tests."""

from __future__ import annotations

import importlib.util
import os
import types
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

_HAS_VERCEL = importlib.util.find_spec('vercel.sandbox') is not None if importlib.util.find_spec('vercel') else False
collect_ignore = (
    [] if _HAS_VERCEL else ['test_backend.py', 'test_conformance.py', 'test_vercel_live.py', 'test_vercel_sandbox.py']
)

# Live sandboxes run for this many seconds per session, so one a crashed run leaves behind stops
# soon instead of running out the 45-minute default.
LIVE_SANDBOX_TIMEOUT = 600

if TYPE_CHECKING or _HAS_VERCEL:  # pragma: no branch - the SDK-installed and slim jobs take opposite branches
    from vercel import sandbox as vercel_sdk

    from .fake_vercel import FakeVercel


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate the live tier, before any live fixture can start a sandbox.

    It runs only with `PYDANTIC_AI_HARNESS_VERCEL_LIVE=1` and Vercel credentials: `VERCEL_OIDC_TOKEN`,
    or `VERCEL_TOKEN` with `VERCEL_TEAM_ID` and `VERCEL_PROJECT_ID`. Otherwise it skips, unless
    `VERCEL_REQUIRE_LIVE=1` (a CI live job sets it), where missing credentials fail instead: a live
    job that tested nothing must not pass as a silent skip.
    """
    if item.get_closest_marker('vercel_live') is None:
        return
    credentials = os.getenv('VERCEL_OIDC_TOKEN') or os.getenv('VERCEL_TOKEN')
    if os.getenv('PYDANTIC_AI_HARNESS_VERCEL_LIVE') == '1' and credentials:  # pragma: no cover - live job
        return
    reason = 'requires PYDANTIC_AI_HARNESS_VERCEL_LIVE=1 and VERCEL_OIDC_TOKEN or VERCEL_TOKEN'
    if os.getenv('VERCEL_REQUIRE_LIVE') == '1':  # pragma: no cover - live job
        pytest.fail(f'VERCEL_REQUIRE_LIVE=1 but the live tier cannot run: it {reason}.')
    pytest.skip(reason)


class _PoisonedVercel(types.ModuleType):
    """A `vercel.sandbox` stand-in that fails loudly on any attribute access.

    The real SDK is installed in the dev venv for the live tier, so a unit test that forgets the
    `fake_vercel` fixture would otherwise reach Vercel and, with developer credentials configured,
    create real billed sandboxes.
    """

    def __getattr__(self, name: str) -> object:  # pragma: no cover - tripwire, hit only by a misbehaving test
        raise AssertionError(
            'A vercel_sandbox unit test touched the real `vercel.sandbox` package. '
            'Use the `fake_vercel` fixture, or mark the test `vercel_live`.'
        )


@pytest.fixture(autouse=True)
def _no_real_vercel(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison `vercel.sandbox` for every test here except the opt-in live tier."""
    if 'vercel_live' in request.keywords:  # pragma: no cover - live tier runs without coverage
        yield
        return
    monkeypatch.setattr('pydantic_ai_harness.vercel_sandbox._backend.vercel', _PoisonedVercel('vercel.sandbox'))
    yield


if _HAS_VERCEL:  # pragma: no branch - the fixture cannot be defined without its SDK-backed fake

    @pytest.fixture
    async def fake_vercel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[FakeVercel]:
        """Swap the SDK's `SandboxClient` for the fake and yield its control surface.

        The real module stays in place for its error and status types, which the backend matches on.
        Processes start in `tmp_path`, resolved so `pwd -P` matches it on hosts whose temporary
        directory sits behind a symlink.
        """
        fake = FakeVercel(tmp_path.resolve())
        monkeypatch.setattr('pydantic_ai_harness.vercel_sandbox._backend.vercel', vercel_sdk)
        monkeypatch.setattr(vercel_sdk, 'SandboxClient', types.SimpleNamespace(create=fake.client))
        try:
            yield fake
        finally:
            await fake.aclose()
