"""Shared fixtures for BoatSandbox tests."""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest

from .fake_boat import FakeBoat

# CI stops the live job after 15 minutes, before a `finally` can delete what it started. Live tests
# create sandboxes that Boat stops this many seconds after they start, so one left behind that way
# stops running instead of waiting for Boat's default hour.
LIVE_AUTO_STOP = 900
# Boat refuses sandbox starts over the account's per-minute limit (12 on its smallest plan) with
# `429 rate_limited`; a live test starts at most this many.
LIVE_STARTS_PER_TEST = 4


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate the live tier, before any live fixture can start a sandbox.

    It runs only with `PYDANTIC_AI_HARNESS_BOAT_LIVE=1` and a non-empty `BOAT_API_KEY`. Otherwise
    it skips, unless `BOAT_REQUIRE_LIVE=1` (CI's live job sets it), where a missing key fails
    instead: a live job that tested nothing must not pass as a silent skip.
    """
    if item.get_closest_marker('boat_live') is None:
        return
    opted_in = os.getenv('PYDANTIC_AI_HARNESS_BOAT_LIVE') == '1'
    # Only the live job takes the next two branches.
    if opted_in and os.getenv('BOAT_API_KEY'):  # pragma: no cover
        return
    reason = 'requires PYDANTIC_AI_HARNESS_BOAT_LIVE=1 and a non-empty BOAT_API_KEY'
    if os.getenv('BOAT_REQUIRE_LIVE') == '1':  # pragma: no cover
        pytest.fail(f'BOAT_REQUIRE_LIVE=1 but the live tier cannot run: it {reason}.')
    pytest.skip(reason)


def _wait_for_start_budget() -> None:  # pragma: no cover - live tier
    """Wait, up to about a minute, until Boat's start limit leaves room for one live test's sandboxes."""
    headers = {'Authorization': f'Bearer {os.environ["BOAT_API_KEY"]}'}
    deadline = time.monotonic() + 65
    while time.monotonic() < deadline:
        response = httpx.get('https://boat.dev/api/v1/limits', headers=headers, timeout=30)
        # A key without `account.read` cannot read its limits: start anyway.
        if not response.is_success or response.json()['starts']['minute']['remaining'] >= LIVE_STARTS_PER_TEST:
            return
        time.sleep(5)


# A tripwire, hit only by a misbehaving test.
def _real_client(api_key: str) -> httpx.AsyncClient:  # pragma: no cover
    raise AssertionError(
        'A boat_sandbox unit test reached for the real Boat API. Use the `fake_boat` fixture, or mark the test '
        '`boat_live`.'
    )


@pytest.fixture(autouse=True)
def _no_real_boat(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Trip on any real Boat client in every test here except the opt-in live tier.

    A unit test that forgets the `fake_boat` fixture would otherwise reach the real API and, with a
    developer's `BOAT_API_KEY` exported, create real billed sandboxes.
    """
    # The live tier runs without coverage.
    if 'boat_live' in request.keywords:  # pragma: no cover
        _wait_for_start_budget()
        yield
        return
    monkeypatch.setattr('pydantic_ai_harness.boat_sandbox._api.open_client', _real_client)
    monkeypatch.delenv('BOAT_API_KEY', raising=False)
    yield


@pytest.fixture
async def fake_boat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[FakeBoat]:
    """Serve the backend's own API clients from a fake Boat, and return its control surface.

    Its sandboxes live under `tmp_path`, resolved so the working directory `pwd -P` reports
    matches it on hosts whose temp directory sits behind a symlink.
    """
    fake = FakeBoat(tmp_path.resolve())
    monkeypatch.setenv('BOAT_API_KEY', fake.api_key)
    monkeypatch.setattr('pydantic_ai_harness.boat_sandbox._api.open_client', fake.client)
    # The fake answers at once, so the backend's waits between checks would only slow the tests.
    for name, delay in (
        ('_api._RETRY_DELAY', 0.0),
        ('_backend._RETRY_DELAY', 0.0),
        ('_backend._READY_POLL', 0.0),
        ('_backend._POLL_INTERVAL', 0.01),
        ('_backend._MAX_POLL_INTERVAL', 0.01),
    ):
        monkeypatch.setattr(f'pydantic_ai_harness.boat_sandbox.{name}', delay)
    try:
        yield fake
    finally:
        await fake.close()
