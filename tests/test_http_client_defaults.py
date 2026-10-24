"""Defaults of the HTTP clients Pydantic AI creates, and the public factory that builds them.

These are unit tests rather than VCR tests: connection pool limits and client timeouts are local
client configuration that never appears in a recorded request. HTTPX keeps the limits on its
transport without a public accessor, so the tests record what the client hands its transport.
"""

from __future__ import annotations

import httpx
import httpx2

from pydantic_ai.models import create_async_http_client, create_async_httpx2_client, get_user_agent


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


async def test_httpx2_client_custom_transport(transport_limits: list[httpx.Limits | httpx2.Limits]):
    """Requests go through the given transport, and the client keeps its timeouts, user agent and connect cap."""
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200)

    async with create_async_httpx2_client(transport=httpx2.MockTransport(handler)) as client:
        assert client.timeout == httpx2.Timeout(600, connect=5)
        response = await client.get('https://example.com', timeout=30)

    assert response.status_code == 200
    assert len(requests) == 1
    assert requests[0].headers['User-Agent'] == get_user_agent()
    # `ConnectPoolTimeoutCap` keeps the 30-second request timeout from lengthening the 5-second connect timeout.
    assert requests[0].extensions['timeout'] == {'connect': 5, 'read': 30, 'write': 30, 'pool': 30}
    # The given transport owns the connection pool: the client builds no transport of its own.
    assert transport_limits == []


async def test_legacy_httpx_client_defaults(transport_limits: list[httpx.Limits | httpx2.Limits]):
    async with create_async_http_client() as client:
        assert client.timeout == httpx.Timeout(600, connect=5)

    assert transport_limits
    assert all(
        limits == httpx.Limits(max_connections=1000, max_keepalive_connections=100) for limits in transport_limits
    )
