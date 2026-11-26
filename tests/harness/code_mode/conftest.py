"""Fixtures shared by the CodeMode test modules."""

from __future__ import annotations

import faulthandler
import os
from collections.abc import AsyncIterator, Iterator

import pytest

from tests.harness.code_mode.grpc_relay import MontyRelay, serve_monty_relay


@pytest.fixture
# Only the relay tests use this.
async def monty_relay() -> AsyncIterator[MontyRelay]:  # pragma: lax no cover
    """Serve remote Monty workers on an ephemeral loopback port."""
    async with serve_monty_relay() as relay:
        yield relay


_RELAY_TEST_TIMEOUT_SECONDS = float(os.environ.get('MONTY_RELAY_TEST_TIMEOUT', '120'))


@pytest.fixture(autouse=True)
def _relay_hang_guard(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail a hung relay test after a hard timeout, dumping every thread's stack first (#14674).

    `dump_traceback_later(..., exit=True)` fires from a watchdog thread, so it lands even when
    the test's event loop is wedged, and it hard-exits the pytest process so CI reports a
    failure with stacks in the log instead of stalling until the job timeout.
    """
    if not {'monty_relay', 'temporal_monty_relay'} & set(request.fixturenames):
        yield
        return
    faulthandler.dump_traceback_later(_RELAY_TEST_TIMEOUT_SECONDS, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
