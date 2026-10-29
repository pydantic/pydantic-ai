"""Shared fixtures for DaytonaSandbox tests."""

from __future__ import annotations

import importlib.util
import os
import types
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

_HAS_DAYTONA = importlib.util.find_spec('daytona') is not None
collect_ignore = (
    []
    if _HAS_DAYTONA
    else ['test_backend.py', 'test_conformance.py', 'test_daytona_live.py', 'test_daytona_sandbox.py']
)

# CI stops the live job after 15 minutes, before a `finally` can delete what it started.
# Live tests create sandboxes that Daytona stops after this many idle minutes, so one left
# behind that way stops running instead of waiting for Daytona's default.
LIVE_AUTO_STOP_INTERVAL = 5

if TYPE_CHECKING or _HAS_DAYTONA:  # pragma: no branch - the SDK-installed and slim jobs take opposite branches
    import daytona

    from .fake_daytona import FakeDaytona


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate the live tier, before any live fixture can start a sandbox.

    It runs only with `PYDANTIC_AI_HARNESS_DAYTONA_LIVE=1` and a non-empty `DAYTONA_API_KEY`.
    Otherwise it skips, unless `DAYTONA_REQUIRE_LIVE=1` (CI's live job sets it), where a missing
    key fails instead: a live job that tested nothing must not pass as a silent skip.
    """
    if item.get_closest_marker('daytona_live') is None:
        return
    opted_in = os.getenv('PYDANTIC_AI_HARNESS_DAYTONA_LIVE') == '1'
    if opted_in and os.getenv('DAYTONA_API_KEY'):  # pragma: no cover - live job
        return
    reason = 'requires PYDANTIC_AI_HARNESS_DAYTONA_LIVE=1 and a non-empty DAYTONA_API_KEY'
    if os.getenv('DAYTONA_REQUIRE_LIVE') == '1':  # pragma: no cover - live job
        pytest.fail(f'DAYTONA_REQUIRE_LIVE=1 but the live tier cannot run: it {reason}.')
    pytest.skip(reason)


class _PoisonedDaytona(types.ModuleType):
    """A `daytona` stand-in that fails loudly on any attribute access.

    Real daytona is installed in the dev venv for the live tier, so a unit test that forgets the
    `fake_daytona` fixture would otherwise reach the real SDK and, with a developer API key
    configured, create real billed sandboxes.
    """

    def __getattr__(self, name: str) -> object:  # pragma: no cover - tripwire, hit only by a misbehaving test
        raise AssertionError(
            'A daytona_sandbox unit test touched the real `daytona` package. '
            'Use the `fake_daytona` fixture, or mark the test `daytona_live`.'
        )


@pytest.fixture(autouse=True)
def _no_real_daytona(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison `daytona` for every test here except the opt-in live tier."""
    if 'daytona_live' in request.keywords:  # pragma: no cover - live tier runs without coverage
        yield
        return
    monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend.daytona', _PoisonedDaytona('daytona'))
    yield


if _HAS_DAYTONA:  # pragma: no branch - the fixture cannot be defined without its SDK-backed fake

    @pytest.fixture
    def fake_daytona(monkeypatch: pytest.MonkeyPatch) -> FakeDaytona:
        """Swap the SDK's `AsyncDaytona` for the fake and return its control surface.

        The real module stays in place for its error and parameter types, which the backend matches on.
        """
        fake = FakeDaytona()
        monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend.daytona', daytona)
        monkeypatch.setattr(daytona, 'AsyncDaytona', fake.client)
        return fake


@pytest.fixture
def blockbuster_enabled() -> bool:
    """`fake_daytona` serves the SDK from local files and processes, synchronously."""
    return False
