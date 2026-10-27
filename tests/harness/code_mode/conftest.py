"""Fixtures shared by the CodeMode test modules."""

from __future__ import annotations

import asyncio
import functools
import os
import sys
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import pytest
from pydantic_monty import AsyncMontyWebsocket
from pydantic_monty._binary import find_monty_binary  # the lookup `AsyncMonty()` uses for local workers

from tests.harness.code_mode import websocket_relay

_REMOTE_MONTY_URL = os.environ.get('MONTY_SANDBOX_TEST_URL')
"""When set, the suite's local Monty workers are dialed on this server instead (CI: staging)."""


@pytest.fixture(autouse=True)
def _remote_monty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the local worker pool for a remote one, sending `LOGFIRE_API_KEY` as a bearer token."""
    if _REMOTE_MONTY_URL:  # pragma: no cover -- remote mode only
        key = os.environ.get('LOGFIRE_API_KEY')
        headers = {'Authorization': f'Bearer {key}'} if key else {}
        remote_pool = functools.partial(AsyncMontyWebsocket, _REMOTE_MONTY_URL, connect_headers=lambda: headers)
        monkeypatch.setattr('pydantic_ai_harness._monty_exec.AsyncMonty', remote_pool)


@asynccontextmanager
# Only the relay tests use these, and they are skipped until #8824.
async def websocket_relay_server(port: int = 0) -> AsyncGenerator[str, None]:  # pragma: lax no cover
    """Run the protocol relay on a loopback port (ephemeral by default) and yield its URL."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        websocket_relay.__file__,
        '--port',
        str(port),
        '--monty-bin',
        find_monty_binary(),
        stdout=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        url_line = await asyncio.wait_for(process.stdout.readline(), timeout=30)
        assert url_line, 'WebSocket relay exited before printing its URL'
        yield url_line.decode().strip()
    finally:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except TimeoutError:  # pragma: lax no cover -- only when the relay ignores SIGTERM
            process.kill()
            await process.wait()


@pytest.fixture
async def websocket_relay_url() -> AsyncIterator[str]:  # pragma: lax no cover
    """Start the protocol relay on an ephemeral loopback port."""
    async with websocket_relay_server() as url:
        yield url
