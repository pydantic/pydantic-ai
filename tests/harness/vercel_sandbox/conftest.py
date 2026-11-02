"""Shared fixtures for VercelSandbox tests."""

from __future__ import annotations

import importlib.util
import os
import types
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

_HAS_VERCEL = importlib.util.find_spec('vercel.sandbox') is not None
collect_ignore = (
    [] if _HAS_VERCEL else ['test_backend.py', 'test_conformance.py', 'test_vercel_live.py', 'test_vercel_sandbox.py']
)

# Live sandboxes are destroyed by the tests that create them; CI stops a live job after 15 minutes, before a
# `finally` can run, so each one also gets a short session: one left behind that way stops within minutes.
LIVE_EXECUTION_TIME_LIMIT = 600

if TYPE_CHECKING or _HAS_VERCEL:  # pragma: no branch - the SDK-installed and slim jobs take opposite branches
    from vercel import sandbox as vercel_sandbox

    from .fake_vercel import FakeVercel


def _has_credentials() -> bool:
    return bool(os.getenv('VERCEL_OIDC_TOKEN')) or all(
        os.getenv(name) for name in ('VERCEL_TOKEN', 'VERCEL_TEAM_ID', 'VERCEL_PROJECT_ID')
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate the live tier, before any live fixture can create a sandbox.

    It runs only with `PYDANTIC_AI_HARNESS_VERCEL_LIVE=1` and Vercel credentials: `VERCEL_OIDC_TOKEN`, or
    `VERCEL_TOKEN` with `VERCEL_TEAM_ID` and `VERCEL_PROJECT_ID`. Otherwise it skips, unless
    `VERCEL_REQUIRE_LIVE=1` (CI's live job sets it), where missing credentials fail instead: a live job
    that tested nothing must not pass as a silent skip.
    """
    if item.get_closest_marker('vercel_live') is None:
        return
    if _has_credentials() and os.getenv('PYDANTIC_AI_HARNESS_VERCEL_LIVE') == '1':  # pragma: no cover - live job
        return
    reason = (
        'requires PYDANTIC_AI_HARNESS_VERCEL_LIVE=1 and VERCEL_OIDC_TOKEN, '
        'or VERCEL_TOKEN with VERCEL_TEAM_ID and VERCEL_PROJECT_ID'
    )
    if os.getenv('VERCEL_REQUIRE_LIVE') == '1':  # pragma: no cover - live job
        pytest.fail(f'VERCEL_REQUIRE_LIVE=1 but the live tier cannot run: it {reason}.')
    pytest.skip(reason)


class _PoisonedSandboxModule(types.ModuleType):
    """A `vercel.sandbox` stand-in that fails loudly on any attribute access.

    The real SDK is installed in the dev venv for the live tier, so a unit test that forgets the
    `fake_vercel` fixture would otherwise reach Vercel and, with credentials configured, create
    real billed sandboxes.
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
    monkeypatch.setattr(
        'pydantic_ai_harness.vercel_sandbox._backend.vercel_sandbox', _PoisonedSandboxModule('vercel.sandbox')
    )
    yield


if _HAS_VERCEL:  # pragma: no branch - the fixture cannot be defined without its SDK-backed fake

    @pytest.fixture
    async def fake_vercel(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeVercel]:
        """Swap the SDK's `SandboxClient.create` for the fake and return its control surface.

        The real module stays in place for its error types, which the backend matches on.
        """
        fake = FakeVercel()
        monkeypatch.setattr('pydantic_ai_harness.vercel_sandbox._backend.vercel_sandbox', vercel_sandbox)
        monkeypatch.setattr(vercel_sandbox.SandboxClient, 'create', fake.create)
        # Nothing here waits on a real control plane.
        monkeypatch.setattr('pydantic_ai_harness.vercel_sandbox._backend._POLL_INTERVAL', 0)
        yield fake
        await fake.reap()

    @pytest.fixture
    def host(fake_vercel: FakeVercel, tmp_path: Path) -> FakeVercel:
        """The fake with its sandboxes' commands running in a temporary host directory."""
        # Resolved so the working directory `pwd -P` reports matches it on hosts whose temp
        # directory sits behind a symlink.
        fake_vercel.host_root = tmp_path.resolve()
        return fake_vercel
