"""Defaults of the HTTP clients Pydantic AI creates, and the public factory that builds them.

These are unit tests rather than VCR tests: connection pool limits and client timeouts are local
client configuration that never appears in a recorded request. HTTPX keeps the limits on its
transport without a public accessor, so the tests record what the client hands its transport.
"""

from __future__ import annotations

from typing import Any

import httpx
import httpx2
import pytest

from pydantic_ai.models import create_async_http_client, create_async_httpx2_client, get_user_agent


@pytest.fixture
def transport_limits(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Limits | httpx2.Limits]:
    """Record the `limits` every HTTPX transport (either family) is constructed with."""
    recorded: list[httpx.Limits | httpx2.Limits] = []

    for transport_class in (httpx2.AsyncHTTPTransport, httpx.AsyncHTTPTransport):
        original_init = transport_class.__init__

        def recording_init(self: Any, *args: Any, _original_init: Any = original_init, **kwargs: Any) -> None:
            recorded.append(kwargs['limits'])
            _original_init(self, *args, **kwargs)

        monkeypatch.setattr(transport_class, '__init__', recording_init)

    return recorded


async def test_httpx2_client_defaults(transport_limits: list[httpx.Limits | httpx2.Limits]):
    async with create_async_httpx2_client() as client:
        assert client.timeout == httpx2.Timeout(600, connect=5)
        assert client.headers['User-Agent'] == get_user_agent()

    # One transport, plus one per proxy HTTPX picks up from the environment.
    assert transport_limits
    assert all(
        limits == httpx2.Limits(max_connections=1000, max_keepalive_connections=100) for limits in transport_limits
    )


async def test_httpx2_client_custom_timeout_and_limits(transport_limits: list[httpx.Limits | httpx2.Limits]):
    timeout = httpx2.Timeout(120, connect=5, pool=10)
    limits = httpx2.Limits(max_connections=200, max_keepalive_connections=50)

    async with create_async_httpx2_client(timeout=timeout, limits=limits) as client:
        assert client.timeout == timeout
        assert client.headers['User-Agent'] == get_user_agent()

    assert transport_limits
    assert all(recorded == limits for recorded in transport_limits)


async def test_httpx2_client_numeric_timeout():
    """A number of seconds sets every phase, as it does for `httpx2.AsyncClient(timeout=...)` itself."""
    async with create_async_httpx2_client(timeout=30) as client:
        assert client.timeout == httpx2.Timeout(30)


async def test_legacy_httpx_client_defaults(transport_limits: list[httpx.Limits | httpx2.Limits]):
    async with create_async_http_client() as client:
        assert client.timeout == httpx.Timeout(600, connect=5)

    assert transport_limits
    assert all(
        limits == httpx.Limits(max_connections=1000, max_keepalive_connections=100) for limits in transport_limits
    )
