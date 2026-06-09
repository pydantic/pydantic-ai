"""Unit tests for the gateway auth mechanics that don't depend on the backend wire contract.

The OIDC/device wire protocol still has `TODO(gateway-auth)` markers, so these cover the parts
that are contract-independent: token freshness, device-poll handling, the OIDC exchange request
shape (against a mock transport), and `GatewayAuth`'s header injection + refresh-on-401.
"""

from __future__ import annotations

import time

import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers._gateway_auth import (
    GatewayAuth,
    GatewayAuthConfig,
    GatewayCredentials,
    GatewayToken,
    _device_poll_result,  # pyright: ignore[reportPrivateUsage]
    _OidcExchangeStrategy,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.anyio


def test_token_is_fresh():
    assert GatewayToken('t').is_fresh()  # no expiry → always fresh
    assert GatewayToken('t', expires_at=time.time() + 3600).is_fresh()
    assert not GatewayToken('t', expires_at=time.time() + 5).is_fresh()  # inside the 60s margin
    assert not GatewayToken('t', expires_at=time.time() - 1).is_fresh()


def test_device_poll_result():
    pending = httpx.Response(400, json={'error': 'authorization_pending'})
    slow_down = httpx.Response(400, json={'error': 'slow_down'})
    ok = httpx.Response(200, json={'access_token': 'x'})

    assert _device_poll_result(pending, 5) == 5
    assert _device_poll_result(slow_down, 5) == 10
    assert _device_poll_result(ok, 5) is None
    with pytest.raises(UserError, match='Gateway device authorization failed: access_denied'):
        _device_poll_result(httpx.Response(400, json={'error': 'access_denied'}), 5)


async def test_oidc_exchange_request_shape(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('GATEWAY_OIDC_TOKEN', 'oidc-subject-token')
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.path] = request
        if request.url.path.endswith('/.well-known/oauth-authorization-server'):
            return httpx.Response(200, json={'token_endpoint': 'https://auth.example.com/oauth/token'})
        return httpx.Response(200, json={'access_token': 'gw-token', 'expires_in': 300})

    config = GatewayAuthConfig(
        auth_base_url='https://auth.example.com',
        resource='https://gateway.example.com/proxy',
        oidc_audience='https://gateway.example.com',
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        token = await _OidcExchangeStrategy(config).acquire_async(http)

    assert token.access_token == 'gw-token'
    assert token.expires_at is not None
    exchange = seen['/oauth/token']
    body = dict(httpx.QueryParams(exchange.content.decode()))
    assert body['grant_type'] == 'urn:ietf:params:oauth:grant-type:token-exchange'
    assert body['subject_token'] == 'oidc-subject-token'
    assert body['scope'] == 'project:gateway_proxy'
    assert body['resource'] == 'https://gateway.example.com/proxy'


class _StubCredentials(GatewayCredentials):
    """Returns a sequence of tokens so we can assert refresh-on-401 picks the next one."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._i = -1

    async def async_token(self, *, force_refresh: bool = False) -> str:
        if force_refresh or self._i < 0:
            self._i = min(self._i + 1, len(self._tokens) - 1)
        return self._tokens[self._i]


async def test_gateway_auth_injects_and_refreshes_on_401():
    auth = GatewayAuth(_StubCredentials(['first', 'second']))
    flow = auth.async_auth_flow(httpx.Request('POST', 'https://gateway.example.com/proxy/openai'))

    request = await flow.__anext__()
    assert request.headers['Authorization'] == 'Bearer first'

    # A 401 should trigger a forced refresh and a retried request with the new token.
    retried = await flow.asend(httpx.Response(401))
    assert retried.headers['Authorization'] == 'Bearer second'

    with pytest.raises(StopAsyncIteration):
        await flow.asend(httpx.Response(200))
