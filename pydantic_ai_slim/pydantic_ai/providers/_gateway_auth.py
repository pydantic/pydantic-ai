"""Credential acquisition and automatic refresh for the Pydantic AI Gateway.

This module lets the gateway provider authenticate with **short-lived OAuth tokens**
instead of a static `pylf_*` key, and refresh them transparently — mirroring the
"token provider" pattern of cloud SDKs (e.g. OpenAI's `azure_ad_token_provider`,
Google's `google.auth` credentials).

A single [`GatewayCredentials`][pydantic_ai.providers._gateway_auth.GatewayCredentials]
object knows how to obtain a token via one of three strategies and refresh it:

- **static**: a `pylf_*` key (or any pre-issued token) — never refreshed.
- **oidc**: exchange an ambient workload OIDC token (e.g. GitHub Actions) for a
  short-lived gateway token via RFC 8693 token exchange — refreshed by re-exchanging.
- **device**: an interactive OAuth 2.0 device-authorization login (RFC 8628) —
  refreshed with the `refresh_token` grant. Tokens are held **in memory only**.

It is exposed to the HTTPX-based providers (OpenAI/Groq/Anthropic/Google) as an
[`httpx.Auth`][httpx.Auth] (proactive margin refresh + reactive refresh-on-401), and
to the botocore-based Bedrock provider via a `DeferredRefreshableToken` bridge (see
[`gateway_bedrock_session`][pydantic_ai.providers._gateway_auth.gateway_bedrock_session]).

`TODO(gateway-auth)` markers flag wire-protocol values to confirm with the platform team.
"""

from __future__ import annotations as _annotations

import functools
import os
import sys
import threading
import time
import webbrowser
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import anyio
import httpx

from pydantic_ai.exceptions import UserError

if TYPE_CHECKING:
    from botocore.session import Session

# RFC 8693 / RFC 8628 grant types and the gateway OAuth scope.
TOKEN_EXCHANGE_GRANT = 'urn:ietf:params:oauth:grant-type:token-exchange'
DEVICE_CODE_GRANT = 'urn:ietf:params:oauth:grant-type:device_code'
REFRESH_TOKEN_GRANT = 'refresh_token'
# TODO(gateway-auth): confirm the subject_token_type the backend expects for a
# workload OIDC token (`...:jwt` vs `...:id_token`).
SUBJECT_TOKEN_TYPE = 'urn:ietf:params:oauth:token-type:jwt'
GATEWAY_SCOPE = 'project:gateway_proxy'

# Refresh a token this many seconds before it expires (proactive refresh for the
# HTTPX path; the Bedrock path tunes botocore's own windows — see below).
REFRESH_MARGIN_SECONDS = 60.0

# Default public OAuth client identity (a Client ID Metadata Document URL) used for
# the device flow. TODO(gateway-auth): confirm the canonical hosted CIMD URL.
DEFAULT_CIMD_PATH = '/clients/logfire-gateway.json'


@dataclass
class GatewayToken:
    """A gateway access token and its (in-memory) refresh material."""

    access_token: str
    expires_at: float | None = None
    """Unix timestamp when `access_token` expires, or `None` if it does not."""
    refresh_token: str | None = None

    def is_fresh(self, *, margin: float = REFRESH_MARGIN_SECONDS) -> bool:
        return self.expires_at is None or time.time() < self.expires_at - margin


@dataclass
class GatewayAuthConfig:
    """Wire configuration shared by the OIDC and device strategies."""

    auth_base_url: str
    """The OAuth authorization server (its RFC 8414 metadata is discovered)."""
    resource: str
    """The gateway proxy URL the token is bound to (RFC 8707 `resource`)."""
    scope: str = GATEWAY_SCOPE
    client_id: str | None = None
    """Public OAuth client id (CIMD URL) for the device flow."""
    oidc_audience: str | None = None
    """Audience to request on the workload OIDC token (matched by the trust policy)."""


def _expires_at(body: dict[str, Any]) -> float | None:
    expires_in = body.get('expires_in')
    return time.time() + float(expires_in) if expires_in is not None else None


def _to_token(body: dict[str, Any], *, fallback_refresh: str | None = None) -> GatewayToken:
    return GatewayToken(
        access_token=body['access_token'],
        expires_at=_expires_at(body),
        refresh_token=body.get('refresh_token', fallback_refresh),
    )


# --------------------------------------------------------------------------------------
# Token strategies: each knows how to acquire and refresh a token, sync and async.
# --------------------------------------------------------------------------------------


class _TokenStrategy(ABC):
    """Acquire and refresh a gateway token.

    Implemented twice (sync + async) so the botocore (sync) and HTTPX (async) call
    sites can each use their native transport.
    """

    @abstractmethod
    async def acquire_async(self, http: httpx.AsyncClient) -> GatewayToken: ...

    @abstractmethod
    def acquire_sync(self, http: httpx.Client) -> GatewayToken: ...

    async def refresh_async(self, http: httpx.AsyncClient, current: GatewayToken) -> GatewayToken:
        return await self.acquire_async(http)

    def refresh_sync(self, http: httpx.Client, current: GatewayToken) -> GatewayToken:
        return self.acquire_sync(http)


def _exchange_data(config: GatewayAuthConfig, oidc_token: str) -> dict[str, str]:
    return {
        'grant_type': TOKEN_EXCHANGE_GRANT,
        'subject_token': oidc_token,
        'subject_token_type': SUBJECT_TOKEN_TYPE,
        'scope': config.scope,
        'resource': config.resource,
    }


class _OidcExchangeStrategy(_TokenStrategy):
    """Exchange an ambient workload OIDC token for a gateway token (RFC 8693).

    Refresh re-fetches a fresh OIDC token and re-exchanges — ambient identity is
    always available in CI/cloud, so no `refresh_token` is needed.
    """

    def __init__(self, config: GatewayAuthConfig) -> None:
        self._config = config

    async def acquire_async(self, http: httpx.AsyncClient) -> GatewayToken:
        oidc_token = await _fetch_oidc_token_async(http, self._config.oidc_audience)
        token_endpoint = (await _discover_async(http, self._config.auth_base_url))['token_endpoint']
        response = await http.post(token_endpoint, data=_exchange_data(self._config, oidc_token))
        response.raise_for_status()
        return _to_token(response.json())

    def acquire_sync(self, http: httpx.Client) -> GatewayToken:
        oidc_token = _fetch_oidc_token_sync(http, self._config.oidc_audience)
        token_endpoint = _discover_sync(http, self._config.auth_base_url)['token_endpoint']
        response = http.post(token_endpoint, data=_exchange_data(self._config, oidc_token))
        response.raise_for_status()
        return _to_token(response.json())


class _DeviceFlowStrategy(_TokenStrategy):
    """Interactive OAuth 2.0 device-authorization flow (RFC 8628).

    The interactive browser approval happens once on `acquire`; refresh uses the
    `refresh_token` grant and never prompts.
    """

    def __init__(self, config: GatewayAuthConfig) -> None:
        if not config.client_id:  # pragma: no cover
            raise UserError('A `client_id` is required for the gateway device-authorization flow.')
        self._config = config
        self._client_id = config.client_id

    def _device_data(self) -> dict[str, str]:
        return {'client_id': self._client_id, 'scope': self._config.scope, 'resource': self._config.resource}

    def _prompt(self, auth: dict[str, Any]) -> None:
        verification_url = auth.get('verification_uri_complete', auth['verification_uri'])
        print(f'gateway-auth: open {auth["verification_uri"]} and enter code {auth["user_code"]}', file=sys.stderr)
        webbrowser.open(verification_url)
        print('gateway-auth: waiting for approval...', file=sys.stderr)

    async def acquire_async(self, http: httpx.AsyncClient) -> GatewayToken:
        metadata = await _discover_async(http, self._config.auth_base_url)
        started = await http.post(metadata['device_authorization_endpoint'], data=self._device_data())
        started.raise_for_status()
        auth = started.json()
        self._prompt(auth)
        interval = int(auth.get('interval', 5))
        while True:
            await anyio.sleep(interval)
            response = await http.post(
                metadata['token_endpoint'],
                data={
                    'grant_type': DEVICE_CODE_GRANT,
                    'device_code': auth['device_code'],
                    'client_id': self._client_id,
                },
            )
            done = _device_poll_result(response, interval)
            if done is not None:
                interval = done
                continue
            return _to_token(response.json())

    def acquire_sync(self, http: httpx.Client) -> GatewayToken:
        metadata = _discover_sync(http, self._config.auth_base_url)
        started = http.post(metadata['device_authorization_endpoint'], data=self._device_data())
        started.raise_for_status()
        auth = started.json()
        self._prompt(auth)
        interval = int(auth.get('interval', 5))
        while True:
            time.sleep(interval)
            response = http.post(
                metadata['token_endpoint'],
                data={
                    'grant_type': DEVICE_CODE_GRANT,
                    'device_code': auth['device_code'],
                    'client_id': self._client_id,
                },
            )
            done = _device_poll_result(response, interval)
            if done is not None:
                interval = done
                continue
            return _to_token(response.json())

    async def refresh_async(self, http: httpx.AsyncClient, current: GatewayToken) -> GatewayToken:
        if not current.refresh_token:  # pragma: no cover
            return await self.acquire_async(http)
        token_endpoint = (await _discover_async(http, self._config.auth_base_url))['token_endpoint']
        response = await http.post(token_endpoint, data=self._refresh_data(current.refresh_token))
        response.raise_for_status()
        return _to_token(response.json(), fallback_refresh=current.refresh_token)

    def refresh_sync(self, http: httpx.Client, current: GatewayToken) -> GatewayToken:
        if not current.refresh_token:  # pragma: no cover
            return self.acquire_sync(http)
        token_endpoint = _discover_sync(http, self._config.auth_base_url)['token_endpoint']
        response = http.post(token_endpoint, data=self._refresh_data(current.refresh_token))
        response.raise_for_status()
        return _to_token(response.json(), fallback_refresh=current.refresh_token)

    def _refresh_data(self, refresh_token: str) -> dict[str, str]:
        return {'grant_type': REFRESH_TOKEN_GRANT, 'refresh_token': refresh_token, 'client_id': self._client_id}


def _device_poll_result(response: httpx.Response, interval: int) -> int | None:
    """Return a new poll interval to keep waiting, or `None` once a token is issued."""
    if response.status_code == 200:
        return None
    error = response.json().get('error')
    if error == 'authorization_pending':
        return interval
    if error == 'slow_down':
        return interval + 5
    raise UserError(f'Gateway device authorization failed: {error}')


# --------------------------------------------------------------------------------------
# Discovery + ambient OIDC helpers (sync + async).
# --------------------------------------------------------------------------------------


def _well_known_url(auth_base: str) -> str:
    return f'{auth_base.rstrip("/")}/.well-known/oauth-authorization-server'


async def _discover_async(http: httpx.AsyncClient, auth_base: str) -> dict[str, Any]:
    response = await http.get(_well_known_url(auth_base))
    response.raise_for_status()
    return response.json()


def _discover_sync(http: httpx.Client, auth_base: str) -> dict[str, Any]:
    response = http.get(_well_known_url(auth_base))
    response.raise_for_status()
    return response.json()


def _github_oidc_request() -> tuple[str, str]:
    request_url = os.getenv('ACTIONS_ID_TOKEN_REQUEST_URL')
    request_token = os.getenv('ACTIONS_ID_TOKEN_REQUEST_TOKEN')
    if not request_url or not request_token:  # pragma: no cover
        raise UserError(
            'Could not fetch a workload OIDC token: not running in GitHub Actions with id-token write '
            'permission, and `GATEWAY_OIDC_TOKEN` is not set.'
        )
    return request_url, request_token


async def _fetch_oidc_token_async(http: httpx.AsyncClient, audience: str | None) -> str:
    if override := os.getenv('GATEWAY_OIDC_TOKEN'):
        return override
    request_url, request_token = _github_oidc_request()
    params = {'audience': audience} if audience else None
    response = await http.get(request_url, params=params, headers={'Authorization': f'Bearer {request_token}'})
    response.raise_for_status()
    return response.json()['value']


def _fetch_oidc_token_sync(http: httpx.Client, audience: str | None) -> str:
    if override := os.getenv('GATEWAY_OIDC_TOKEN'):
        return override
    request_url, request_token = _github_oidc_request()
    params = {'audience': audience} if audience else None
    response = http.get(request_url, params=params, headers={'Authorization': f'Bearer {request_token}'})
    response.raise_for_status()
    return response.json()['value']


# TODO(gateway-auth): add GCP/AWS metadata-server and generic OIDC-file detectors.
def ambient_oidc_available() -> bool:
    """Whether an ambient workload OIDC token can be obtained without interaction."""
    return bool(os.getenv('GATEWAY_OIDC_TOKEN')) or bool(
        os.getenv('ACTIONS_ID_TOKEN_REQUEST_URL') and os.getenv('ACTIONS_ID_TOKEN_REQUEST_TOKEN')
    )


# --------------------------------------------------------------------------------------
# GatewayCredentials: stateful token holder with proactive margin refresh + locking.
# --------------------------------------------------------------------------------------


class GatewayCredentials:
    """Holds the current gateway token and refreshes it on demand (in memory only)."""

    def __init__(self, strategy: _TokenStrategy) -> None:
        self._strategy = strategy
        self._token: GatewayToken | None = None
        self._sync_lock = threading.Lock()

    @functools.cached_property
    def _async_lock(self) -> anyio.Lock:
        # Bind the lock lazily so it attaches to the running event loop (mirrors `_VertexAIAuth`).
        return anyio.Lock()

    async def async_token(self, *, force_refresh: bool = False) -> str:
        if not force_refresh and self._token is not None and self._token.is_fresh():
            return self._token.access_token
        async with self._async_lock:
            if not force_refresh and self._token is not None and self._token.is_fresh():
                return self._token.access_token
            async with httpx.AsyncClient(timeout=30) as http:
                self._token = (
                    await self._strategy.refresh_async(http, self._token)
                    if self._token is not None
                    else await self._strategy.acquire_async(http)
                )
            return self._token.access_token

    def sync_token(self, *, force_refresh: bool = False) -> GatewayToken:
        if not force_refresh and self._token is not None and self._token.is_fresh():
            return self._token
        with self._sync_lock:
            if not force_refresh and self._token is not None and self._token.is_fresh():
                return self._token
            with httpx.Client(timeout=30) as http:
                self._token = (
                    self._strategy.refresh_sync(http, self._token)
                    if self._token is not None
                    else self._strategy.acquire_sync(http)
                )
            return self._token


def auto_credentials(config: GatewayAuthConfig) -> GatewayCredentials:
    """Auto-detect credentials: ambient workload OIDC (CI/cloud), else an interactive device flow.

    Raises a [`UserError`][pydantic_ai.exceptions.UserError] in a headless process with no ambient
    OIDC token, rather than blocking on a browser prompt.
    """
    if ambient_oidc_available():
        return GatewayCredentials(_OidcExchangeStrategy(config))
    if sys.stdin.isatty() and sys.stdout.isatty():
        return GatewayCredentials(_DeviceFlowStrategy(config))
    raise UserError(
        'Could not authenticate to the Pydantic AI Gateway: no `api_key`/`auth`, no ambient workload OIDC '
        'token, and no terminal for an interactive login. Set `PYDANTIC_AI_GATEWAY_API_KEY`, pass `auth=`, '
        'or configure a workload OIDC trust policy.'
    )


class GatewayAuth(httpx.Auth):
    """`httpx.Auth` that injects the gateway bearer token and refreshes on 401.

    Proactive margin refresh happens inside [`GatewayCredentials`][pydantic_ai.providers._gateway_auth.GatewayCredentials];
    this adds reactive refresh-on-401, following the `_VertexAIAuth` precedent.
    """

    def __init__(self, credentials: GatewayCredentials) -> None:
        self._credentials = credentials

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers['Authorization'] = f'Bearer {self._credentials.sync_token().access_token}'
        response = yield request
        if response.status_code == 401:
            token = self._credentials.sync_token(force_refresh=True)
            request.headers['Authorization'] = f'Bearer {token.access_token}'
            yield request

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        request.headers['Authorization'] = f'Bearer {await self._credentials.async_token()}'
        response = yield request
        if response.status_code == 401:
            request.headers['Authorization'] = f'Bearer {await self._credentials.async_token(force_refresh=True)}'
            yield request


# --------------------------------------------------------------------------------------
# Bedrock / Converse bridge.
#
# Bedrock uses botocore (sync), not httpx, so `GatewayAuth` does not apply. botocore
# *does* support transparent per-request refresh: `RequestSigner.get_auth_instance`
# calls `token_provider.get_frozen_token()` on every request, which refreshes via a
# synchronous callback. We return a `DeferredRefreshableToken` from a `Session`
# subclass (the same hook the existing static `_BearerTokenSession` overrides) and
# tune its refresh windows down from botocore's SSO-oriented 900s/600s defaults so
# short-lived gateway tokens are refreshed near their actual expiry.
# --------------------------------------------------------------------------------------


def gateway_bedrock_session(credentials: GatewayCredentials) -> Session:
    """Return a botocore `Session` that signs Bedrock requests with auto-refreshing gateway tokens."""
    from botocore.session import Session as BotocoreSession
    from botocore.tokens import DeferredRefreshableToken, FrozenAuthToken

    def refresh_using() -> FrozenAuthToken:
        token = credentials.sync_token(force_refresh=True)
        expiration = datetime.fromtimestamp(token.expires_at, tz=timezone.utc) if token.expires_at is not None else None
        return FrozenAuthToken(token.access_token, expiration=expiration)

    class _GatewayDeferredToken(DeferredRefreshableToken):
        # Gateway tokens are short-lived; refresh near their real expiry rather than
        # botocore's 15-min advisory / 10-min mandatory SSO windows.
        _advisory_refresh_timeout = 120
        _mandatory_refresh_timeout = 30

    deferred = _GatewayDeferredToken('gateway-oauth', refresh_using)

    class _GatewaySession(BotocoreSession):
        def get_auth_token(self, **_kwargs: Any) -> DeferredRefreshableToken:  # type: ignore[reportIncompatibleMethodOverride]
            return deferred

        def get_credentials(self) -> None:  # type: ignore[reportIncompatibleMethodOverride]
            return None

    return _GatewaySession()
