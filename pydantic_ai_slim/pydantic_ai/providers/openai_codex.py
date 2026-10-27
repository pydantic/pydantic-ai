"""OpenAI Codex subscription-auth provider: OAuth flow primitives, credential refresh, wire dialect.

Core owns the protocol primitives (PKCE context, authorization URL, code exchange, refresh) and,
because the pinned redirect URI makes every login a localhost callback, the one-shot callback
catcher (`exchange_code_from_callback()`); browser opening and persistent credential storage
belong to applications and harnesses.

The authorization-code + PKCE redirect flow is the only login flow the public Codex client
supports: its registration pins the redirect URI to `http://localhost:1455/auth/callback`
(exact-match, probed live 2026-08-25), and the auth service serves no device-authorization
endpoint. `exchange_code_from_callback()` serves this exact redirect URI and exchanges the
authorization code.
"""

from __future__ import annotations as _annotations

import json
import os
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Protocol, Self
from urllib.parse import urlencode

import httpx2
from pydantic import Field, TypeAdapter, ValidationError

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.openai_codex import openai_codex_model_profile

from ._oauth import OAuthFlow
from ._oauth_provider import (
    CredentialsPersistenceError,
    CredentialsRefreshError,
    OAuthCredentialSource,
    OAuthCredentialsProvider,
    jwt_claims,
    post_token_request as _post_token_request,
)
from ._openai_compatible import AsyncHTTPClient as _OpenAIHTTPClient
from .openai import OpenAIProvider

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
    'OpenAICodexCredentialSource',
    'OpenAICodexCredentials',
    'OpenAICodexOAuthFlow',
    'OpenAICodexProvider',
)

_PROVIDER_NAME = 'openai-codex'
_CODEX_BASE_URL = 'https://chatgpt.com/backend-api/codex'
_AUTHORIZE_URL = 'https://auth.openai.com/oauth/authorize'
_TOKEN_URL = 'https://auth.openai.com/oauth/token'
# The public Codex CLI client: OpenAI's registration pins the redirect URI to exactly this
# localhost URI (probed: alternates rejected pre-login); override `redirect_uri=` only with your own client.
_PUBLIC_CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'
_REDIRECT_URI = 'http://localhost:1455/auth/callback'
_DEFAULT_SCOPE = 'openid profile email offline_access'
_ORIGINATOR = 'pydantic-ai'


@dataclass
class OpenAICodexCredentials:
    """Codex subscription credentials.

    The tokens are excluded from `repr` so a logged or traceback-embedded instance does not leak them.
    """

    _: KW_ONLY
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    account_id: str

    @classmethod
    def from_codex_cli_auth(cls, data: Mapping[str, Any]) -> Self:
        """Parse the Codex CLI `~/.codex/auth.json` shape (`{'tokens': {...}}`)."""
        try:
            tokens = _codex_cli_auth_ta.validate_python(data).tokens
        except ValidationError as e:
            raise UserError(
                f'Malformed Codex CLI credentials. Run `codex login` to regenerate them.\n\n{e.json(include_input=False)}'
            ) from None
        return cls(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            account_id=tokens.account_id,
        )


# Dataclasses for the wire payloads consulted above and below, validated through `TypeAdapter`:
# extra fields are ignored, and a `ValidationError` (a `ValueError`) also rejects non-object payloads.


@dataclass
class _CodexCliTokens:
    """The `tokens` entry of the Codex CLI's `auth.json`."""

    access_token: Annotated[str, Field(min_length=1)]
    refresh_token: Annotated[str, Field(min_length=1)]
    account_id: Annotated[str, Field(min_length=1)]


@dataclass
class _CodexCliAuth:
    """The subset of the Codex CLI's `auth.json` that credentials are built from."""

    tokens: _CodexCliTokens


@dataclass
class _JwtAuthClaim:
    """The nested OpenAI claim carrying the ChatGPT account id."""

    chatgpt_account_id: str | None = None


@dataclass
class _JwtPayload:
    """The unverified id_token claims consulted for the account id."""

    # The Codex id_token nests the account id under this claim.
    auth: Annotated[_JwtAuthClaim | None, Field(validation_alias='https://api.openai.com/auth')] = None
    chatgpt_account_id: str | None = None
    account_id: str | None = None


@dataclass
class _TokenResponse:
    """The fields of an OAuth token-endpoint response that credentials are built from.

    `refresh_token` is required because the flow always requests the `offline_access` scope, and
    credentials without it could not survive their first expiry.
    """

    access_token: Annotated[str, Field(min_length=1)]
    refresh_token: Annotated[str, Field(min_length=1)]
    id_token: str | None = None
    account_id: str | None = None


_codex_cli_auth_ta = TypeAdapter(_CodexCliAuth)
_jwt_payload_ta = TypeAdapter(_JwtPayload)
_token_response_ta = TypeAdapter(_TokenResponse)


def _account_id_from_id_token(token: str) -> str | None:
    payload = jwt_claims(token, _jwt_payload_ta)
    if payload is None:
        return None
    if payload.auth and payload.auth.chatgpt_account_id:
        return payload.auth.chatgpt_account_id
    return payload.chatgpt_account_id or payload.account_id or None


def _credentials_from_token_response(
    data: _TokenResponse, fallback_account_id: str | None = None
) -> OpenAICodexCredentials:
    """Build credentials from an OAuth token-endpoint response."""
    account_id = (
        data.account_id or (_account_id_from_id_token(data.id_token) if data.id_token else None) or fallback_account_id
    )
    if not account_id:
        raise CredentialsRefreshError(
            'Could not determine the ChatGPT account id from the token response.', _PROVIDER_NAME
        )
    return OpenAICodexCredentials(
        access_token=data.access_token, refresh_token=data.refresh_token, account_id=account_id
    )


async def _refresh_credentials(
    credentials: OpenAICodexCredentials, *, http_client: httpx2.AsyncClient | None = None
) -> OpenAICodexCredentials:
    """Exchange the refresh token for a new credential set against the public Codex client.

    Raises [`CredentialsRefreshError`][pydantic_ai.providers.openai_codex.CredentialsRefreshError]
    when the token endpoint rejects the grant (`invalid_grant` means a fresh authorization is
    required) or returns a malformed response.
    """
    data = await _post_token_request(
        _TOKEN_URL,
        {
            'grant_type': 'refresh_token',
            'refresh_token': credentials.refresh_token,
            'client_id': _PUBLIC_CLIENT_ID,
        },
        _token_response_ta,
        provider_name=_PROVIDER_NAME,
        http_client=http_client,
    )
    return _credentials_from_token_response(data, fallback_account_id=credentials.account_id)


class OpenAICodexCredentialSource(OAuthCredentialSource[OpenAICodexCredentials], Protocol):
    """Application-owned storage for the credentials, so refreshed tokens outlive the process.

    The provider owns the credential lifecycle (expiry checks, refresh, single-flight) and calls
    this only to read and write the stored set: `load()` on first use, and `save()` after tokens
    are refreshed.

    Refresh tokens are single-use, so before refreshing, the provider re-reads storage with
    `load()` and adopts newer credentials if another process using the same store already
    refreshed them. That is best-effort, not mutual exclusion: two processes can still race
    between `load()` and `save()`.

    Conformance is structural, but implementations are encouraged to subclass the protocol
    explicitly so type checkers verify the method signatures.
    """


def _read_codex_cli_credentials() -> OpenAICodexCredentials:
    """Read-only load of the Codex CLI's `auth.json` (honors `CODEX_HOME`). Never writes it."""
    code_home = Path(os.getenv('CODEX_HOME') or Path.home() / '.codex')
    path = code_home / 'auth.json'
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        raise UserError(
            f'No Codex CLI credentials found at `{path}`. Run `codex login` first, or pass '
            '`credentials=` / use `OpenAICodexOAuthFlow` explicitly.'
        ) from None
    except OSError as e:
        raise UserError(f'Could not read Codex CLI credentials at `{path}`: {e}') from e
    except UnicodeDecodeError as e:
        raise UserError(f'Codex CLI credentials at `{path}` are not valid UTF-8: {e}') from e
    try:
        data = json.loads(text)
    except ValueError as e:
        raise UserError(f'Malformed Codex CLI credentials at `{path}`: {e}') from e
    return OpenAICodexCredentials.from_codex_cli_auth(data)


class OpenAICodexOAuthFlow(OAuthFlow[OpenAICodexCredentials]):
    """Pure authorization-code + PKCE context for the OpenAI Codex public client.

    This is the only login flow the public client supports (no device flow; redirect URI pinned to
    `localhost:1455`, probed exact-match). Construction does no I/O: build the context anywhere,
    send the user to `authorization_url()`, then let `exchange_code_from_callback()` receive the
    redirect on localhost and exchange its code. The browser and credential storage stay
    caller-owned.
    """

    def __init__(self, *, redirect_uri: str = _REDIRECT_URI, state: str | None = None) -> None:
        """Create a new flow context. Construction does no I/O.

        Args:
            redirect_uri: Where the authorization code is delivered. The public client's
                registration pins this to `http://localhost:1455/auth/callback` (exact-match),
                so leave the default unchanged when using the public client.
            state: The CSRF token bound to the callback; auto-generated when `None`.
        """
        super().__init__(redirect_uri=redirect_uri, state=state)

    def authorization_url(self, *, scope: str | None = None, extra_params: Mapping[str, str] | None = None) -> str:
        """The URL to send the user to. Note the public client pins redirects to localhost.

        Args:
            scope: The OAuth scopes to request; `None` means the standard Codex login scopes.
            extra_params: Additional query parameters, merged over the defaults (so they can also
                override them), except `client_id` and `redirect_uri`: `exchange_code()` always
                posts the public client id and the flow's `redirect_uri`, so overriding either
                here would make the authorization code unusable. The production Codex login's
                `id_token_add_organizations=true` and `codex_cli_simplified_flow=true` are sent by
                default: without the former, the `id_token` can omit the account id for multi-org
                accounts (live-verified 2026-08-25).
        """
        params: dict[str, str] = {
            'response_type': 'code',
            'client_id': _PUBLIC_CLIENT_ID,
            'redirect_uri': self.redirect_uri,
            'scope': scope if scope is not None else _DEFAULT_SCOPE,
            'state': self.state,
            'code_challenge': self.code_challenge,
            'code_challenge_method': 'S256',
            'id_token_add_organizations': 'true',
            'codex_cli_simplified_flow': 'true',
        }
        return f'{_AUTHORIZE_URL}?{urlencode(self._merge_extra_params(params, extra_params))}'

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
            _token_response_ta,
            provider_name=_PROVIDER_NAME,
        )
        return _credentials_from_token_response(data)


class OpenAICodexProvider(OAuthCredentialsProvider[OpenAICodexCredentials]):
    """Provider for OpenAI Codex subscription authentication.

    Wraps the standard `OpenAIProvider` machinery pointed at the Codex backend, injecting Codex
    OAuth credentials instead of API keys. One provider instance carries one set of credentials
    (there is no process-global cache). The instance binds its refresh lock to the first event
    loop that awaits a request, so do not reuse it across loops.

    ```python {test="skip" lint="skip"}
    provider = OpenAICodexProvider(credential_source=YourCredentialStore())
    agent = Agent('openai-codex:gpt-5.6-luna', provider=provider)
    ```
    """

    @property
    def name(self) -> str:
        return _PROVIDER_NAME

    @property
    def canonical_system(self) -> str:
        # The Codex backend and the OpenAI API accept each other's encrypted reasoning and compaction items.
        return 'openai'

    @property
    def base_url(self) -> str:
        return _CODEX_BASE_URL

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        # Codex inherits selected first-party flags that compatible endpoints may not implement. Its schema
        # transformer is an exception: OpenAIProvider's tuple and string-length handling has not been verified
        # against the Codex backend, so Codex keeps the shared transformer while layering its wire dialect on top.
        return merge_profile(OpenAIProvider.model_profile(model_name), openai_codex_model_profile(model_name))

    def __init__(
        self,
        credentials: OpenAICodexCredentials | None = None,
        *,
        credential_source: OpenAICodexCredentialSource | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new OpenAI Codex provider.

        Args:
            credentials: The subscription credentials to inject. If both this and
                `credential_source` are omitted, they are loaded **read-only** from the Codex
                CLI's `auth.json` (honors `CODEX_HOME`), which never writes the file: refreshed
                tokens then live in memory only. Pydantic AI never falls back to `OPENAI_API_KEY`.
            credential_source: Application-owned storage for the credentials, so refreshed tokens
                are persisted between runs; see
                [`OpenAICodexCredentialSource`][pydantic_ai.providers.openai_codex.OpenAICodexCredentialSource].
                Mutually exclusive with `credentials`.
            openai_client: An existing `AsyncOpenAI` client to use as-is. Opts out of credential
                injection entirely; `credentials`, `credential_source`, and `http_client` must
                be `None`.
            http_client: An existing `httpx2.AsyncClient` to use. Must be dedicated to this
                provider (no auth of its own): the provider attaches its credential-injecting auth to it, and
                sharing a client between providers would mix their credentials. The auth only
                injects credentials on HTTPS requests to the Codex host, so the client can safely
                be reused for other destinations.
        """
        if openai_client is None and credentials is None and credential_source is None:
            credentials = _read_codex_cli_credentials()
        super().__init__(
            credentials, credential_source=credential_source, openai_client=openai_client, http_client=http_client
        )

    def _auth_headers(self, credentials: OpenAICodexCredentials) -> dict[str, str]:
        return {
            'Authorization': f'Bearer {credentials.access_token}',
            'chatgpt-account-id': credentials.account_id,
            'originator': _ORIGINATOR,
        }

    async def _refresh_grant(
        self, credentials: OpenAICodexCredentials, http_client: httpx2.AsyncClient
    ) -> OpenAICodexCredentials:
        return await _refresh_credentials(credentials, http_client=http_client)
