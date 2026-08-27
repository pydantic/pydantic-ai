"""OpenAI Codex subscription-auth provider: OAuth flow primitives, credential refresh, wire dialect.

Core owns the *non-interactive* protocol primitives only (PKCE context, authorization URL, code
exchange, refresh); interactive UX (browser opening, localhost callback servers) and persistent
credential storage belong to applications and harnesses.

The authorization-code + PKCE redirect flow is the only login flow the public Codex client
supports: its registration pins the redirect URI to `http://localhost:1455/auth/callback`
(exact-match, probed live 2026-08-25), and the auth service serves no device-authorization
endpoint. Hosted-web login is therefore not possible with this client - apps on the user's
machine serve (or tunnel) `localhost:1455` themselves and hand the code to `exchange_code()`.
"""

from __future__ import annotations as _annotations

import base64
import hashlib
import json
import os
import secrets
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Mapping
from dataclasses import KW_ONLY, dataclass
from datetime import datetime, timedelta, timezone
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlencode

import anyio
import httpx2
from pydantic import BaseModel, Field, SecretStr, StrictFloat, StrictInt, ValidationError
from typing_extensions import Self

from pydantic_ai._http import create_async_httpx2_client
from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.openai_codex import openai_codex_model_profile

from ._openai_compatible import (
    AsyncHTTPClient as _OpenAIHTTPClient,
    OpenAICompatibleProvider as _OpenAICompatibleProvider,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the OpenAI Codex provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = (
    'CredentialsPersistenceError',
    'CredentialsRefreshError',
    'OpenAICodexAuth',
    'OpenAICodexCredentialSnapshot',
    'OpenAICodexCredentialSource',
    'OpenAICodexCredentials',
    'OpenAICodexOAuthFlow',
    'OpenAICodexProvider',
    'refresh_credentials',
)

_CODEX_BASE_URL = 'https://chatgpt.com/backend-api/codex'
_CODEX_HOST = httpx2.URL(_CODEX_BASE_URL).host
_AUTHORIZE_URL = 'https://auth.openai.com/oauth/authorize'
_TOKEN_URL = 'https://auth.openai.com/oauth/token'
# The public Codex CLI client: OpenAI's registration pins the redirect URI to exactly this
# localhost URI (probed: alternates rejected pre-login); override `redirect_uri=` only with your own client.
_PUBLIC_CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'
_REDIRECT_URI = 'http://localhost:1455/auth/callback'
_DEFAULT_SCOPE = 'openid profile email offline_access'
_ORIGINATOR = 'pydantic-ai'
# 30s pre-expiry refresh hint (unverified JWT `exp` is a hint, not an authority).
_TOKEN_EXPIRY_BUFFER = timedelta(seconds=30)


class _CredentialsError(ModelAPIError):
    """Base for Codex credential failures.

    Subclasses [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] so the standard handling of
    provider failures (e.g. [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel]) applies;
    the auth layer runs below any specific model, so `model_name` is the provider name.
    """

    def __init__(self, message: str):
        super().__init__(model_name='openai-codex', message=message)

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        return self.__class__, (self.message,)


class CredentialsRefreshError(_CredentialsError):
    """Refreshing Codex credentials against the token endpoint failed.

    When the underlying error is `invalid_grant`, the stored grant is no longer usable and a fresh
    authorization is required (locally: rerun `codex login`; in an app: rerun your connect flow).
    """


class CredentialsPersistenceError(_CredentialsError):
    """Rotated credentials were updated in memory but the persistence callback raised.

    The in-memory credentials are current and were handed to the callback before it failed; the
    error surfaces so callers do not mistake a failed save for durability.
    """


@dataclass
class OpenAICodexCredentials:
    """Codex subscription credentials.

    Secrets use `SecretStr` so tokens never leak through reprs, logs, or accidental serialization.
    """

    _: KW_ONLY
    access_token: SecretStr
    refresh_token: SecretStr
    account_id: str

    @classmethod
    def from_codex_cli_auth(cls, data: Mapping[str, Any]) -> Self:
        """Parse the Codex CLI `~/.codex/auth.json` shape (`{'tokens': {...}}`)."""
        try:
            tokens = _CodexCliAuth.model_validate(data).tokens
        except ValidationError:
            tokens = None
        if tokens is None:
            raise UserError(
                "Malformed Codex CLI credentials: expected an object with a 'tokens' entry. "
                'Run `codex login` to regenerate them.'
            ) from None
        missing = [
            name
            for name, value in [
                ('access_token', tokens.access_token),
                ('refresh_token', tokens.refresh_token),
                ('account_id', tokens.account_id),
            ]
            if not value
        ]
        if missing:
            raise UserError(
                f'Malformed Codex CLI credentials: missing {", ".join(missing)}. Run `codex login` to regenerate them.'
            )
        return cls(
            access_token=SecretStr(tokens.access_token),
            refresh_token=SecretStr(tokens.refresh_token),
            account_id=tokens.account_id,
        )


# Dedicated models for the wire payloads consulted above and below: pydantic ignores extra
# fields by default, and a `ValidationError` (a `ValueError`) also rejects non-object payloads.


class _CodexCliTokens(BaseModel):
    """The `tokens` entry of the Codex CLI's `auth.json`."""

    access_token: str = ''
    refresh_token: str = ''
    account_id: str = ''


class _CodexCliAuth(BaseModel):
    """The subset of the Codex CLI's `auth.json` that credentials are built from."""

    tokens: _CodexCliTokens | None = None


class _JwtAuthClaim(BaseModel):
    """The nested OpenAI claim carrying the ChatGPT account id."""

    chatgpt_account_id: str | None = None


class _JwtPayload(BaseModel):
    """The unverified JWT claims consulted for expiry and account-id hints."""

    # Strict, so a numeric string is not coerced into an expiry hint.
    exp: StrictInt | StrictFloat | None = None
    # The Codex id_token nests the account id under this claim.
    auth: _JwtAuthClaim | None = Field(default=None, validation_alias='https://api.openai.com/auth')
    chatgpt_account_id: str | None = None
    account_id: str | None = None


class _TokenResponse(BaseModel):
    """The fields of an OAuth token-endpoint response that credentials are built from."""

    access_token: str | None = None
    refresh_token: str | None = None
    id_token: str | None = None
    account_id: str | None = None


class _TokenErrorResponse(BaseModel):
    """An OAuth token-endpoint error body."""

    error: str | None = None
    error_description: str | None = None


def _jwt_payload(token: str) -> _JwtPayload | None:
    """Decode a JWT payload without verifying the signature. Returns `None` for anything malformed."""
    try:
        segment = token.split('.')[1]
    except IndexError:
        return None
    padded = segment + '=' * (-len(segment) % 4)
    try:
        return _JwtPayload.model_validate(json.loads(base64.urlsafe_b64decode(padded)))
    except ValueError:
        return None


def _jwt_expires_at(token: str) -> datetime | None:
    """Best-effort unverified `exp` claim, a refresh hint, never an authority."""
    payload = _jwt_payload(token)
    if payload is None or payload.exp is None:
        return None
    try:
        return datetime.fromtimestamp(payload.exp, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _account_id_from_id_token(token: str) -> str | None:
    payload = _jwt_payload(token)
    if payload is None:
        return None
    if payload.auth and payload.auth.chatgpt_account_id:
        return payload.auth.chatgpt_account_id
    return payload.chatgpt_account_id or payload.account_id or None


def _credentials_from_token_response(
    data: _TokenResponse, fallback_account_id: str | None = None
) -> OpenAICodexCredentials:
    """Build credentials from an OAuth token-endpoint response."""
    if not data.access_token:
        raise CredentialsRefreshError('Token endpoint response is missing `access_token`.')
    if not data.refresh_token:
        raise CredentialsRefreshError(
            'Token endpoint response is missing `refresh_token`; request the `offline_access` scope.'
        )
    account_id = (
        data.account_id
        or (_account_id_from_id_token(data.id_token) if data.id_token else None)
        or fallback_account_id
    )
    if not account_id:
        raise CredentialsRefreshError('Could not determine the ChatGPT account id from the token response.')
    return OpenAICodexCredentials(
        access_token=SecretStr(data.access_token), refresh_token=SecretStr(data.refresh_token), account_id=account_id
    )


async def _post_token_request(
    url: str, form: Mapping[str, str], http_client: httpx2.AsyncClient | None = None
) -> _TokenResponse:
    """POST a form-urlencoded OAuth token request and decode the JSON response.

    When `http_client` is given the request goes through it (so custom transports and proxies apply
    to refreshes too); otherwise an ephemeral client is used.
    """
    if http_client is None:
        async with httpx2.AsyncClient(timeout=httpx2.Timeout(timeout=30, connect=5)) as client:
            response = await client.post(url, data=dict(form), headers={'Accept': 'application/json'})
    else:
        response = await http_client.post(url, data=dict(form), headers={'Accept': 'application/json'})
    if response.status_code != 200:
        try:
            body = _TokenErrorResponse.model_validate(response.json())
        except ValueError:
            body = _TokenErrorResponse()
        detail = body.error_description or body.error or response.text[:200]
        hint = '; the grant was rejected, rerun the authorization flow' if body.error == 'invalid_grant' else ''
        raise CredentialsRefreshError(
            f'Token request to {url} failed with status {response.status_code}: {detail}{hint}'
        )
    try:
        return _TokenResponse.model_validate(response.json())
    except ValueError:
        raise CredentialsRefreshError(f'Token endpoint {url} returned an unexpected response.') from None


async def refresh_credentials(
    credentials: OpenAICodexCredentials, *, http_client: httpx2.AsyncClient | None = None
) -> OpenAICodexCredentials:
    """Exchange the refresh token for a new credential set against the public Codex client.

    The protocol primitive behind the provider's automatic refresh, public so
    [`OpenAICodexCredentialSource`][pydantic_ai.providers.openai_codex.OpenAICodexCredentialSource]
    implementations can perform the upstream refresh inside their own critical section.

    Raises [`CredentialsRefreshError`][pydantic_ai.providers.openai_codex.CredentialsRefreshError]
    when the token endpoint rejects the grant (`invalid_grant` means a fresh authorization is
    required) or returns a malformed response.
    """
    data = await _post_token_request(
        _TOKEN_URL,
        {
            'grant_type': 'refresh_token',
            'refresh_token': credentials.refresh_token.get_secret_value(),
            'client_id': _PUBLIC_CLIENT_ID,
        },
        http_client=http_client,
    )
    return _credentials_from_token_response(data, fallback_account_id=credentials.account_id)


def _token_is_stale(access_token: str) -> bool:
    """Whether the unverified JWT `exp` hint says the token is within the pre-expiry buffer."""
    expires_at = _jwt_expires_at(access_token)
    return expires_at is not None and datetime.now(timezone.utc) >= expires_at - _TOKEN_EXPIRY_BUFFER


@dataclass(frozen=True)
class OpenAICodexCredentialSnapshot:
    """A credential set paired with the application's opaque persistence revision.

    The revision identifies the stored version this credential set came from (a database row
    version, an etag, a UUID), so refresh coordination can distinguish 'this grant was rejected'
    from 'another replica already rotated it'.
    """

    _: KW_ONLY
    credentials: OpenAICodexCredentials
    revision: str


class OpenAICodexCredentialSource(Protocol):
    """Application-owned, revision-aware credential resolution for multi-replica deployments.

    `credentials` + `on_credentials_refresh` is a single-process convenience: each provider
    instance refreshes independently, so two replicas sharing one stored grant can both consume
    the same rotating refresh token and invalidate each other. A credential source moves the whole
    refresh transaction (reload, decide, upstream refresh, durable save) into application code,
    where it can be wrapped in a per-user distributed lock.

    The provider calls `get_credentials()` before each request. When the returned token looks
    stale, or the backend rejects it with a 401, the provider calls again with
    `force_refresh=True` and `rejected_revision` set to the revision it used. Implementations
    should then, inside their critical section: reload the active snapshot; if its revision
    differs from `rejected_revision`, return it as-is (another replica already rotated the grant);
    otherwise perform the upstream refresh (e.g. via
    [`refresh_credentials`][pydantic_ai.providers.openai_codex.refresh_credentials]), durably
    replace the stored set with a compare-and-swap on the expected revision, and return the new
    snapshot.
    """

    async def get_credentials(
        self, *, force_refresh: bool = False, rejected_revision: str | None = None
    ) -> OpenAICodexCredentialSnapshot:
        """Return the current credential snapshot, refreshing upstream if required."""
        ...


def _read_codex_cli_credentials() -> OpenAICodexCredentials:
    """Read-only load of the Codex CLI's `auth.json` (honors `CODEX_HOME`). Never writes it."""
    code_home = Path(os.getenv('CODEX_HOME') or Path.home() / '.codex')
    path = code_home / 'auth.json'
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise UserError(
            f'No Codex CLI credentials found at `{path}`. Run `codex login` first, or pass '
            '`credentials=` / use `OpenAICodexOAuthFlow` explicitly.'
        ) from None
    except OSError as e:
        raise UserError(f'Could not read Codex CLI credentials at `{path}`: {e}') from e
    try:
        data = json.loads(text)
    except ValueError as e:
        raise UserError(f'Malformed Codex CLI credentials at `{path}`: {e}') from e
    return OpenAICodexCredentials.from_codex_cli_auth(data)


class OpenAICodexOAuthFlow:
    """Pure authorization-code + PKCE context for the OpenAI Codex public client.

    This is the only login flow the public client supports (no device flow; redirect URI pinned to
    `localhost:1455`, probed exact-match). Construction does no I/O: build the context anywhere,
    send the user to `authorization_url()`, then call `exchange_code()` from your redirect handler -
    served from port 1455 on the user's machine, or tunneled there. Core owns none of the
    interactive parts.
    """

    def __init__(self, *, redirect_uri: str = _REDIRECT_URI, state: str | None = None) -> None:
        self.redirect_uri = redirect_uri
        self.state = state or secrets.token_urlsafe(16)
        self.code_verifier = secrets.token_urlsafe(32)

    def authorization_url(self, *, scope: str = _DEFAULT_SCOPE, extra_params: Mapping[str, str] | None = None) -> str:
        """The URL to send the user to. Note the public client pins redirects to localhost.

        Args:
            scope: The OAuth scopes to request.
            extra_params: Additional query parameters, merged over the defaults (so they can also
                override them), except `client_id` and `redirect_uri`: `exchange_code()` always
                posts the public client id and the flow's `redirect_uri`, so overriding either
                here would make the authorization code unusable. The production Codex login's
                `id_token_add_organizations=true` and `codex_cli_simplified_flow=true` are sent by
                default: without the former, the `id_token` can omit the account id for multi-org
                accounts (live-verified 2026-08-25).
        """
        challenge = base64.urlsafe_b64encode(hashlib.sha256(self.code_verifier.encode()).digest()).rstrip(b'=')
        params: dict[str, str] = {
            'response_type': 'code',
            'client_id': _PUBLIC_CLIENT_ID,
            'redirect_uri': self.redirect_uri,
            'scope': scope,
            'state': self.state,
            'code_challenge': challenge.decode(),
            'code_challenge_method': 'S256',
            'id_token_add_organizations': 'true',
            'codex_cli_simplified_flow': 'true',
        }
        if extra_params:
            if overridden := sorted({'client_id', 'redirect_uri'} & extra_params.keys()):
                raise UserError(
                    f'`extra_params` cannot override {", ".join(overridden)}: `exchange_code()` always posts '
                    "the public client id and the flow's `redirect_uri`, so the authorization code would be "
                    'unusable. Pass `redirect_uri=` to the constructor instead.'
                )
            params.update(extra_params)
        return f'{_AUTHORIZE_URL}?{urlencode(params)}'

    async def exchange_code(self, code: str) -> OpenAICodexCredentials:
        """Exchange an authorization code for credentials (call this in your callback handler)."""
        data = await _post_token_request(
            _TOKEN_URL,
            {
                'grant_type': 'authorization_code',
                'code': code,
                'code_verifier': self.code_verifier,
                'redirect_uri': self.redirect_uri,
                'client_id': _PUBLIC_CLIENT_ID,
            },
        )
        return _credentials_from_token_response(data)


class OpenAICodexAuth(httpx2.Auth):
    """httpx auth injecting Codex subscription headers, with single-flight refresh-and-replay.

    Injects `Authorization: Bearer …`, `chatgpt-account-id`, and `originator`, but only on
    HTTPS requests to the Codex host, so a caller-supplied client reused for other destinations
    (or downgraded to plaintext) never leaks credentials. On a 401 it performs at most one refresh-and-replay; non-expiry 401s
    therefore cannot loop. The proactive expiry check treats the unverified JWT `exp` as a hint only.
    """

    def __init__(self, provider: OpenAICodexProvider) -> None:
        self._provider = provider

    def _apply_headers(self, request: httpx2.Request, credentials: OpenAICodexCredentials) -> None:
        request.headers['Authorization'] = f'Bearer {credentials.access_token.get_secret_value()}'
        request.headers['chatgpt-account-id'] = credentials.account_id
        request.headers['originator'] = _ORIGINATOR

    def sync_auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        raise RuntimeError('`OpenAICodexAuth` only supports async HTTP clients.')

    async def async_auth_flow(self, request: httpx2.Request) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        if request.url.scheme != 'https' or request.url.host != _CODEX_HOST:
            # Never send subscription credentials to a foreign destination or over plaintext:
            # a caller-supplied client may be reused for arbitrary requests.
            yield request
            return
        # Buffer the outgoing body so the 401 replay can resend it: a one-shot stream (an async
        # generator upload) would otherwise raise `StreamConsumed` on the second send. Codex
        # payloads are JSON already held in memory, and foreign-host requests bypass this above.
        await request.aread()
        # The two classes are deliberately coupled in one module; the provider owns the state and
        # the auth is its wire-side skin. The replay closure carries whatever context the
        # provider's mode (in-memory single-flight, or application credential source) needs.
        credentials, replay = await self._provider._prepare_request_credentials()  # pyright: ignore[reportPrivateUsage]
        self._apply_headers(request, credentials)
        response = yield request
        if response.status_code != 401:
            return
        # Release the connection before replaying.
        await response.aread()
        self._apply_headers(request, await replay())
        yield request  # replay exactly once; its response goes back to the caller


class OpenAICodexProvider(_OpenAICompatibleProvider):
    """Provider for OpenAI Codex subscription authentication.

    Wraps the standard `OpenAIProvider` machinery pointed at the Codex backend, injecting Codex
    OAuth credentials instead of API keys. One provider instance carries one tenant's credentials
    (there is no process-global cache); construct one per user/credential set. The instance binds
    its refresh lock to the first event loop that awaits a request — do not reuse across loops.

    ```python {test="skip" lint="skip"}
    provider = OpenAICodexProvider(
        credentials=credentials_from_your_db,
        on_credentials_refresh=save_to_your_db,  # rotated tokens come back here
    )
    agent = Agent('openai-codex:gpt-5.6-codex', provider=provider)
    ```
    """

    @property
    def name(self) -> str:
        return 'openai-codex'

    @property
    def base_url(self) -> str:
        return _CODEX_BASE_URL

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @property
    def credentials(self) -> OpenAICodexCredentials:
        """The current credentials (rotated in place by refreshes; persist via the callback)."""
        if self._credential_source is not None:
            raise UserError(
                '`credentials` is unavailable when a `credential_source` owns the credentials; '
                'query your source for the current snapshot instead.'
            )
        if self._credentials is None:
            raise UserError(
                '`credentials` is unavailable when the provider wraps an existing `openai_client`, '
                'which opts out of credential injection entirely.'
            )
        return self._credentials

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return openai_codex_model_profile(model_name)

    def __init__(
        self,
        credentials: OpenAICodexCredentials | None = None,
        *,
        credential_source: OpenAICodexCredentialSource | None = None,
        on_credentials_refresh: Callable[[OpenAICodexCredentials], Awaitable[None]] | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new OpenAI Codex provider.

        Args:
            credentials: The subscription credentials to inject. If omitted, they are loaded
                **read-only** from the Codex CLI's `auth.json` (honors `CODEX_HOME`) — see
                [`from_codex_cli`][pydantic_ai.providers.openai_codex.OpenAICodexProvider.from_codex_cli].
                Pydantic AI never falls back to `OPENAI_API_KEY`.
            credential_source: Application-owned, revision-aware credential resolution for
                multi-replica deployments; see
                [`OpenAICodexCredentialSource`][pydantic_ai.providers.openai_codex.OpenAICodexCredentialSource].
                Mutually exclusive with `credentials` and `on_credentials_refresh`: the source owns
                loading, refresh coordination, and persistence.
            on_credentials_refresh: Async callback invoked with the complete new credential set
                whenever tokens are rotated, so apps can persist them. If it raises, in-memory
                credentials stay current but a [`CredentialsPersistenceError`][pydantic_ai.providers.openai_codex.CredentialsPersistenceError]
                surfaces instead of pretending durability succeeded. Note this is a single-process
                convenience; multi-replica services should use `credential_source`.
            openai_client: An existing `AsyncOpenAI` client to use as-is. Opts out of credential
                injection entirely; `credentials`, `on_credentials_refresh`, and `http_client`
                must be `None`.
            http_client: An existing `httpx2.AsyncClient` to use. Must be dedicated to this
                provider (no auth of its own): the provider attaches `OpenAICodexAuth` to it, and
                sharing a client between providers would mix tenants' credentials. The auth only
                injects credentials on HTTPS requests to the Codex host, so the client can safely
                be reused for other destinations.
        """
        self._on_credentials_refresh = on_credentials_refresh
        self._credential_source = credential_source
        self._credentials: OpenAICodexCredentials | None = None
        if openai_client is not None:
            assert credentials is None, 'Cannot provide both `openai_client` and `credentials`'
            assert credential_source is None, 'Cannot provide both `openai_client` and `credential_source`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert on_credentials_refresh is None, 'Cannot provide both `openai_client` and `on_credentials_refresh`'
            self._client = openai_client
            return

        if credential_source is not None:
            assert credentials is None, 'Cannot provide both `credentials` and `credential_source`'
            assert on_credentials_refresh is None, (
                'Cannot provide both `on_credentials_refresh` and `credential_source`: the source owns persistence'
            )
        else:
            self._credentials = credentials if credentials is not None else _read_codex_cli_credentials()
        self._revision = 0
        self._last_refresh_error: tuple[int, Exception] | None = None
        self._auth = OpenAICodexAuth(self)
        if http_client is None:
            http_client = create_async_httpx2_client()
            self._own_http_client = http_client
            self._http_client_factory = self._create_http_client
        else:
            if not isinstance(http_client, httpx2.AsyncClient):
                raise UserError(
                    '`OpenAICodexProvider` requires an `httpx2` client for `http_client`: the legacy '
                    '`httpx.AsyncClient` cannot carry its credential-injecting auth.'
                )
            if http_client.auth is not None:
                raise UserError(
                    'The `http_client` already has auth configured (it may belong to another provider); '
                    'pass a dedicated client so credentials cannot mix across tenants.'
                )
        http_client.auth = self._auth
        self._http_client = http_client
        self._client = AsyncOpenAI(
            base_url=_CODEX_BASE_URL,
            # The SDK merges its own bearer header into requests; `OpenAICodexAuth` replaces it.
            api_key='codex-subscription-auth',
            http_client=http_client,
        )

    def _create_http_client(self) -> httpx2.AsyncClient:
        """Factory used when a closed provider-owned client is reopened."""
        client = create_async_httpx2_client()
        client.auth = self._auth
        self._http_client = client
        return client

    def _set_http_client(self, http_client: _OpenAIHTTPClient) -> None:
        http_client.auth = self._auth  # pyright: ignore[reportAttributeAccessIssue]
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

    @classmethod
    def from_codex_cli(
        cls,
        *,
        on_credentials_refresh: Callable[[OpenAICodexCredentials], Awaitable[None]] | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> OpenAICodexProvider:
        """Load credentials **read-only** from the Codex CLI (`~/.codex/auth.json`, honors `CODEX_HOME`).

        Convenience for local development after running `codex login`. This never writes the CLI's
        file — refreshed tokens live in memory and go to `on_credentials_refresh` if provided.
        """
        return cls(
            credentials=_read_codex_cli_credentials(),
            on_credentials_refresh=on_credentials_refresh,
            http_client=http_client,
        )

    async def _prepare_request_credentials(
        self,
    ) -> tuple[OpenAICodexCredentials, Callable[[], Awaitable[OpenAICodexCredentials]]]:
        """The credentials for an outgoing request, plus a replay resolver for a 401 on it.

        With a `credential_source`, resolution is delegated per request and stale or rejected
        tokens escalate through `get_credentials(force_refresh=True, rejected_revision=...)`, so
        the application's critical section owns the whole refresh transaction. Otherwise the
        provider's in-memory single-flight logic applies.
        """
        if (source := self._credential_source) is not None:
            snapshot = await source.get_credentials()
            if _token_is_stale(snapshot.credentials.access_token.get_secret_value()):
                # Route the proactive pre-expiry refresh through the source too: coordinating only
                # the 401 path would leave this path with the same cross-replica race.
                snapshot = await source.get_credentials(force_refresh=True, rejected_revision=snapshot.revision)
            revision = snapshot.revision

            async def replay() -> OpenAICodexCredentials:
                replayed = await source.get_credentials(force_refresh=True, rejected_revision=revision)
                return replayed.credentials

            return snapshot.credentials, replay

        await self._refresh_if_stale()
        revision_used = self._revision

        async def replay() -> OpenAICodexCredentials:
            await self._refresh_for_401(revision_used)
            return self.credentials

        return self.credentials, replay

    @cached_property
    def _refresh_lock(self) -> anyio.Lock:
        # Like the base provider's enter lock: bind lazily so we attach to whatever loop first
        # awaits a request rather than construction time.
        return anyio.Lock()

    async def _refresh_if_stale(self) -> None:
        """Proactive refresh from the unverified-JWT `exp` hint.

        Failures are swallowed here: the hint is best-effort, and the 401 path retries with real
        errors surfaced.
        """
        if not self._is_stale():
            return
        try:
            async with self._refresh_lock:
                if self._is_stale():  # recheck after acquiring: single-flight, not just serialized
                    await self._refresh_locked()
        except CredentialsPersistenceError:
            raise  # the refresh itself succeeded; a failed save must never be silent
        except Exception:
            # Transport failures and rejected grants alike fall through to the 401 path, which
            # retries with the still-current token and surfaces real errors.
            pass

    async def _refresh_for_401(self, revision_used: int) -> None:
        """Single-flight refresh after a 401 carrying `revision_used`.

        If another task already replaced the credentials since the failed request was sent, no
        network refresh happens — the caller replays with the fresh set directly.
        """
        if self._revision != revision_used:
            return
        async with self._refresh_lock:
            if self._revision != revision_used:  # recheck after acquiring
                return
            if (last := self._last_refresh_error) is not None and last[0] == revision_used:
                raise last[1]  # share the single-flight failure instead of re-running it per waiter
            try:
                await self._refresh_locked()
            except Exception as e:
                self._last_refresh_error = (revision_used, e)
                raise

    async def _refresh_locked(self) -> None:
        # The caller must hold `_refresh_lock`.
        assert self._refresh_lock.locked()
        # The provider's own client, so custom transports and proxies apply to refreshes too;
        # the auth flow ignores non-Codex hosts, so this cannot recurse or leak the bearer.
        new_credentials = await refresh_credentials(self.credentials, http_client=self._http_client)
        # Atomic replace of the complete set, then bump the revision so concurrent 401s observe it.
        self._credentials = new_credentials
        self._revision += 1
        if on_refresh := self._on_credentials_refresh:
            try:
                await on_refresh(new_credentials)
            except Exception as e:
                raise CredentialsPersistenceError(
                    'Credentials were refreshed in memory but the persistence callback raised.'
                ) from e

    def _is_stale(self) -> bool:
        return _token_is_stale(self.credentials.access_token.get_secret_value())
