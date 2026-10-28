"""Shared fixtures for BoxdSandbox tests."""

from __future__ import annotations

import importlib.util
import os
import types
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING

import pytest

_HAS_BOXD = importlib.util.find_spec('boxd') is not None
collect_ignore = (
    [] if _HAS_BOXD else ['test_backend.py', 'test_boxd_live.py', 'test_boxd_sandbox.py', 'test_conformance.py']
)

# Live machines are deleted by the tests that create them; CI stops a live job after 15 minutes, before a
# `finally` can run, so each one also gets a short idle suspend: a machine left behind that way stops
# using its vCPUs instead of running until someone deletes it.
LIVE_AUTO_SUSPEND_TIMEOUT = 60

if TYPE_CHECKING or _HAS_BOXD:  # pragma: no branch - the SDK-installed and slim jobs take opposite branches
    import boxd

    from .fake_boxd import FakeBoxd


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate the live tier, before any live fixture can create a machine.

    It runs only with `PYDANTIC_AI_HARNESS_BOXD_LIVE=1` and a non-empty `BOXD_API_KEY`. Otherwise
    it skips, unless `BOXD_REQUIRE_LIVE=1` (CI's live job sets it), where a missing key fails
    instead: a live job that tested nothing must not pass as a silent skip.
    """
    if item.get_closest_marker('boxd_live') is None:
        return
    opted_in = os.getenv('PYDANTIC_AI_HARNESS_BOXD_LIVE') == '1'
    if opted_in and os.getenv('BOXD_API_KEY'):  # pragma: no cover - live job
        return
    reason = 'requires PYDANTIC_AI_HARNESS_BOXD_LIVE=1 and a non-empty BOXD_API_KEY'
    if os.getenv('BOXD_REQUIRE_LIVE') == '1':  # pragma: no cover - live job
        pytest.fail(f'BOXD_REQUIRE_LIVE=1 but the live tier cannot run: it {reason}.')
    pytest.skip(reason)


class _PoisonedBoxd(types.ModuleType):
    """A `boxd` stand-in that fails loudly on any attribute access.

    The real SDK is installed in the dev venv for the live tier, so a unit test that forgets the
    `fake_boxd` fixture would otherwise reach production and, with a developer API key configured,
    create real billed machines.
    """

    def __getattr__(self, name: str) -> object:  # pragma: no cover - tripwire, hit only by a misbehaving test
        raise AssertionError(
            'A boxd_sandbox unit test touched the real `boxd` package. '
            'Use the `fake_boxd` fixture, or mark the test `boxd_live`.'
        )


@pytest.fixture(autouse=True)
def _no_real_boxd(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison `boxd` for every test here except the opt-in live tier."""
    if 'boxd_live' in request.keywords:  # pragma: no cover - live tier runs without coverage
        yield
        return
    monkeypatch.setattr('pydantic_ai_harness.boxd_sandbox._backend.boxd', _PoisonedBoxd('boxd'))
    yield


if _HAS_BOXD:  # pragma: no branch - the fixture cannot be defined without its SDK-backed fake

    @pytest.fixture
    async def fake_boxd(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeBoxd]:
        """Swap the SDK's `AsyncBoxd` for the fake and return its control surface.

        The real module stays in place for its error types, which the backend matches on.
        """
        fake = FakeBoxd()
        monkeypatch.setattr('pydantic_ai_harness.boxd_sandbox._backend.boxd', boxd)
        monkeypatch.setattr(boxd, 'AsyncBoxd', fake.client)
        # Nothing here waits on a real control plane.
        monkeypatch.setattr('pydantic_ai_harness.boxd_sandbox._backend._POLL_INTERVAL', 0)
        yield fake
        await fake.reap()
