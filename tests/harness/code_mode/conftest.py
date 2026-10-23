"""Fixtures shared by the CodeMode test modules."""

from __future__ import annotations

import asyncio
import functools
import os
import sys
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic_monty import AsyncMontyWebsocket
from pydantic_monty._binary import find_monty_binary  # the lookup `AsyncMonty()` uses for local workers

from pydantic_ai_harness.code_mode import CodeModeToolset
from tests.harness.code_mode import websocket_relay

_REMOTE_MONTY_URL = os.environ.get('MONTY_SANDBOX_TEST_URL') or None
"""When set, the whole suite runs against this Monty server instead of local workers (CI: staging)."""

_REMOTE_MONTY_KEY = os.environ.get('MONTY_EXECUTION_KEY') or None
"""Sent as a bearer token when dialing, for servers behind an authenticating proxy."""


@pytest.fixture(autouse=True)
def _remote_monty(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every `CodeMode` without its own `monty_sandbox_url` at `MONTY_SANDBOX_TEST_URL`."""
    if _REMOTE_MONTY_URL is not None:  # pragma: no cover -- remote mode only
        if 'local_monty' in request.keywords:
            pytest.skip('needs a local Monty worker')
        if _REMOTE_MONTY_KEY is not None:
            headers = {'Authorization': f'Bearer {_REMOTE_MONTY_KEY}'}
            monkeypatch.setattr(
                'pydantic_ai_harness._monty_exec.AsyncMontyWebsocket',
                functools.partial(AsyncMontyWebsocket, connect_headers=lambda: headers),
            )
        original = CodeModeToolset.__aenter__

        async def aenter(self: CodeModeToolset[Any]) -> CodeModeToolset[Any]:
            if self.monty_sandbox_url is None:
                self.monty_sandbox_url = _REMOTE_MONTY_URL
            return await original(self)

        monkeypatch.setattr(CodeModeToolset, '__aenter__', aenter)


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
