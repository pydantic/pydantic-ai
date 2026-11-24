from __future__ import annotations

import importlib.util
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

_HAS_WEBSOCKETS = importlib.util.find_spec('websockets') is not None
collect_ignore = [] if _HAS_WEBSOCKETS else ['fake_runta.py', 'test_conformance.py', 'test_runta_sandbox.py']

# Installed and slim jobs take opposite branches.
if TYPE_CHECKING or _HAS_WEBSOCKETS:  # pragma: no branch
    from .fake_runta import FakeRunta


def live_token() -> str:
    """Return `RUNTA_TOKEN` for the live tier, or skip it.

    The tier runs only with `PYDANTIC_AI_HARNESS_RUNTA_LIVE=1` and a non-empty token. Setting
    `RUNTA_REQUIRE_LIVE` turns the skip into a failure, so a job meant to run the tier cannot pass as a
    green run that tested nothing when its secret is missing.
    """
    token = os.getenv('RUNTA_TOKEN')
    if os.getenv('PYDANTIC_AI_HARNESS_RUNTA_LIVE') == '1' and token:
        return token
    reason = 'requires PYDANTIC_AI_HARNESS_RUNTA_LIVE=1 and a non-empty RUNTA_TOKEN'
    if os.getenv('RUNTA_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}:
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope='session')
def runta_token() -> str:
    """Session-scoped so it skips before any live fixture that would use the token."""
    return live_token()


# The fixture requires the fake.
if _HAS_WEBSOCKETS:  # pragma: no branch

    @pytest.fixture
    async def runta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[FakeRunta]:
        fake = FakeRunta(tmp_path)
        monkeypatch.setenv('RUNTA_TOKEN', fake.token)
        monkeypatch.delenv('RUNTA_ENDPOINT', raising=False)
        # Runtime polls wait half a second against the live service; the fake answers at once.
        monkeypatch.setattr('pydantic_ai_harness.runta_sandbox._backend._POLL_INTERVAL', 0.0)
        monkeypatch.setattr('pydantic_ai_harness.runta_sandbox._client.connect', fake.connect)

        async def handle(transport: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
            del transport
            return await fake.handle(request)

        monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', handle)
        yield fake
        for socket in fake.sockets:
            await socket.close()
