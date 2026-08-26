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
from datetime import datetime, timedelta, timezone
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import anyio
import httpx2
from pydantic import BaseModel, SecretStr, TypeAdapter, ValidationError
from typing_extensions import Self

from pydantic_ai._http import create_async_httpx2_client, warn_if_legacy_httpx_client
from pydantic_ai.exceptions import UserError
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
    'OpenAICodexCredentials',
    'OpenAICodexOAuthFlow',
    'OpenAICodexProvider',
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


class CredentialsRefreshError(RuntimeError):
    """Refreshing Codex credentials against the token endpoint failed.

    When the underlying error is `invalid_grant`, the stored grant is no longer usable and a fresh
    authorization is required (locally: rerun `codex login`; in an app: rerun your connect flow).
    """


class CredentialsPersistenceError(RuntimeError):
    """Rotated credentials were updated in memory but the persistence callback raised.

    The in-memory credentials are current and were handed to the callback before it failed; the
    error surfaces so callers do not mistake a failed save for durability.
    """


class OpenAICodexCredentials(BaseModel):
    """Codex subscription credentials.

    Secrets use `SecretStr` so tokens never leak through reprs, logs, or accidental serialization.
    """

    access_token: SecretStr
    refresh_token: SecretStr
    account_id: str
    model_config = {'extra': 'ignore'}

    @classmethod
    def from_codex_cli_auth(cls, data: Mapping[str, Any]) -> Self:
        """Parse the Codex CLI `~/.codex/auth.json` shape (`{'tokens': {...}}`)."""
        tokens = data.get('tokens')
        try:
            token_data = _OBJECT_ADAPTER.validate_python(tokens)
        except ValidationError:
            raise UserError(
                "Malformed Codex CLI credentials: expected an object with a 'tokens' entry. "
                'Run `codex login` to regenerate them.'
            ) from None
        missing = [key for key in ('access_token', 'refresh_token', 'account_id') if not token_data.get(key)]
        if missing:
            raise UserError(
                f'Malformed Codex CLI credentials: missing {", ".join(missing)}. Run `codex login` to regenerate them.'
            )
        return cls(
            access_token=SecretStr(token_data['access_token']),
            refresh_token=SecretStr(token_data['refresh_token']),
            account_id=token_data['account_id'],
        )


_JWT_PAYLOAD_ADAPTER = TypeAdapter(dict[str, Any])
_OBJECT_ADAPTER = TypeAdapter(dict[str, Any])


def _jwt_payload(token: str) -> dict[str, Any] | None:
    """Decode a JWT payload without verifying the signature. Returns `None` for anything malformed."""
    try:
        segment = token.split('.')[1]
    except IndexError:
        return None
    padded = segment + '=' * (-len(segment) % 4)
    try:
        # `ValidationError` subclasses `ValueError`, so this also rejects non-object payloads.
        return _JWT_PAYLOAD_ADAPTER.validate_python(json.loads(base64.urlsafe_b64decode(padded)))
    except ValueError:
        return None


def _jwt_expires_at(token: str) -> datetime | None:
    """Best-effort unverified `exp` claim — a refresh hint, never an authority."""
    payload = _jwt_payload(token)
    exp = payload.get('exp') if payload else None
    if isinstance(exp, bool) or not isinstance(exp, int | float):
        return None
    try:
        return datetime.fromtimestamp(exp, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _account_id_from_id_token(token: str) -> str | None:
    payload = _jwt_payload(token)
    if payload is None:
        return None
    try:
        # The Codex id_token nests the account id under this claim.
        claim = _OBJECT_ADAPTER.validate_python(payload.get('https://api.openai.com/auth'))
    except ValidationError:
        claim = {}
    account_id = claim.get('chatgpt_account_id')
    if isinstance(account_id, str) and account_id:
        return account_id
    for key in ('chatgpt_account_id', 'account_id'):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _credentials_from_token_response(
    data: Mapping[str, Any], fallback_account_id: str | None = None
) -> OpenAICodexCredentials:
    """Build credentials from an OAuth token-endpoint response."""
    access_token = data.get('access_token')
    if not isinstance(access_token, str) or not access_token:
        raise CredentialsRefreshError('Token endpoint response is missing `access_token`.')
    refresh_token = data.get('refresh_token')
    if not isinstance(refresh_token, str) or not refresh_token:
        raise CredentialsRefreshError(
            'Token endpoint response is missing `refresh_token`; request the `offline_access` scope.'
        )
    id_token = data.get('id_token')
    account_id = (
        data.get('account_id')
        or (_account_id_from_id_token(id_token) if isinstance(id_token, str) else None)
        or fallback_account_id
    )
    if not account_id:
        raise CredentialsRefreshError('Could not determine the ChatGPT account id from the token response.')
    return OpenAICodexCredentials(
        access_token=SecretStr(access_token), refresh_token=SecretStr(refresh_token), account_id=str(account_id)
    )


async def _post_json(url: str, form: Mapping[str, str]) -> dict[str, Any]:
    """POST a form-urlencoded OAuth request and decode the JSON response."""
    async with httpx2.AsyncClient(timeout=httpx2.Timeout(timeout=30, connect=5)) as client:
        response = await client.post(url, data=dict(form), headers={'Accept': 'application/json'})
    if response.status_code != 200:
        try:
            body = response.json()
            detail = body.get('error_description') or body.get('error') or response.text[:200]
            error = body.get('error')
        except ValueError:
            detail, error = response.text[:200], None
        hint = '; the grant was rejected — rerun the authorization flow' if error == 'invalid_grant' else ''
        raise CredentialsRefreshError(
            f'Token request to {url} failed with status {response.status_code}: {detail}{hint}'
        )
    return response.json()


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
        data = _OBJECT_ADAPTER.validate_json(text)
    except ValidationError as e:
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
                override them). The production Codex login's `id_token_add_organizations=true` and
                `codex_cli_simplified_flow=true` are sent by default: without the former, the
                `id_token` can omit the account id for multi-org accounts (live-verified 2026-08-25).
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
            params.update(extra_params)
        return f'{_AUTHORIZE_URL}?{urlencode(params)}'

    async def exchange_code(self, code: str) -> OpenAICodexCredentials:
        """Exchange an authorization code for credentials (call this in your callback handler)."""
        data = await _post_json(
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

    def _apply_headers(self, request: httpx2.Request) -> None:
        credentials = self._provider.credentials
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
        # The two classes are deliberately coupled in one module; the provider owns the state and
        # the auth is its wire-side skin.
        await self._provider._refresh_if_stale()  # pyright: ignore[reportPrivateUsage]
        revision = self._provider._revision  # pyright: ignore[reportPrivateUsage]
        self._apply_headers(request)
        response = yield request
        if response.status_code != 401:
            return
        # Release the connection before replaying.
        await response.aread()
        await self._provider._refresh_for_401(revision)  # pyright: ignore[reportPrivateUsage]
        self._apply_headers(request)
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
        return self._credentials

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return openai_codex_model_profile(model_name)

    def __init__(
        self,
        credentials: OpenAICodexCredentials | None = None,
        *,
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
            on_credentials_refresh: Async callback invoked with the complete new credential set
                whenever tokens are rotated, so apps can persist them. If it raises, in-memory
                credentials stay current but a [`CredentialsPersistenceError`][pydantic_ai.providers.openai_codex.CredentialsPersistenceError]
                surfaces instead of pretending durability succeeded.
            openai_client: An existing `AsyncOpenAI` client to use as-is. Opts out of credential
                injection entirely; `credentials`, `on_credentials_refresh`, and `http_client`
                must be `None`.
            http_client: An existing HTTP client to use. Note the provider attaches
                `OpenAICodexAuth` to whichever client it ends up using; the auth only injects
                credentials on requests to the Codex host, so the client can safely be reused
                for other destinations.
        """
        self._on_credentials_refresh = on_credentials_refresh
        if openai_client is not None:
            assert credentials is None, 'Cannot provide both `openai_client` and `credentials`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert on_credentials_refresh is None, 'Cannot provide both `openai_client` and `on_credentials_refresh`'
            self._client = openai_client
            return

        self._credentials = credentials if credentials is not None else _read_codex_cli_credentials()
        self._revision = 0
        self._auth = OpenAICodexAuth(self)
        if http_client is None:
            http_client = create_async_httpx2_client()
            self._own_http_client = http_client
            self._http_client_factory = self._create_http_client
        else:
            warn_if_legacy_httpx_client(http_client, consumer='OpenAI-compatible providers', stacklevel=3)
        # `AuthTypes` accepts any concrete `httpx2.Auth` subclass; pyright's structural check on
        # the generator-method pair is over-strict here.
        http_client.auth = self._auth  # pyright: ignore[reportAttributeAccessIssue]
        self._client = AsyncOpenAI(
            base_url=_CODEX_BASE_URL,
            # The SDK merges its own bearer header into requests; `OpenAICodexAuth` replaces it.
            api_key='codex-subscription-auth',
            http_client=http_client,  # pyright: ignore[reportArgumentType]
        )

    def _create_http_client(self) -> _OpenAIHTTPClient:
        """Factory used when a closed provider-owned client is reopened."""
        client = create_async_httpx2_client()
        client.auth = self._auth
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
        except CredentialsRefreshError:
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
            await self._refresh_locked()

    async def _refresh_locked(self) -> None:
        # The caller must hold `_refresh_lock`.
        assert self._refresh_lock.locked()
        data = await _post_json(
            _TOKEN_URL,
            {
                'grant_type': 'refresh_token',
                'refresh_token': self._credentials.refresh_token.get_secret_value(),
                'client_id': _PUBLIC_CLIENT_ID,
            },
        )
        new_credentials = _credentials_from_token_response(data, fallback_account_id=self._credentials.account_id)
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
        expires_at = _jwt_expires_at(self._credentials.access_token.get_secret_value())
        return expires_at is not None and datetime.now(timezone.utc) >= expires_at - _TOKEN_EXPIRY_BUFFER
