"""ChatGPT plan usage through Sign in with ChatGPT: OAuth flow, credential refresh, and provider.

OpenAI lets open-source and locally run apps use a signed-in user's ChatGPT plan for Responses API
requests, as documented in [ChatGPT plan usage](https://developers.openai.com/siwc/token-sharing-open-source).
The first sign-in registers an OAuth client for the user's ChatGPT account through dynamic registration;
later sign-ins reuse the issued `client_id`. Every sign-in redirects to an HTTP loopback callback on
`127.0.0.1`, so core owns the one-shot callback catcher (`exchange_code_from_callback()`); browser
opening and persistent credential storage belong to applications.
"""

from __future__ import annotations as _annotations

import secrets
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Annotated, Protocol
from urllib.parse import urlencode, urlsplit

import httpx2
from pydantic import Field, TypeAdapter

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.openai_chatgpt import openai_chatgpt_model_profile

from ._oauth import OAuthFlow
from ._oauth_provider import (
    CredentialsPersistenceError,
    CredentialsRefreshError,
    OAuthCredentialSource,
    OAuthCredentialsProvider,
    jwt_claims,
    jwt_expires_at,
    post_token_request as _post_token_request,
)
from ._openai_compatible import AsyncHTTPClient as _OpenAIHTTPClient
from .openai import OpenAIProvider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the OpenAI ChatGPT provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = (
    'CredentialsPersistenceError',
    'CredentialsRefreshError',
    'OpenAIChatGPTCredentialSource',
    'OpenAIChatGPTCredentials',
    'OpenAIChatGPTOAuthFlow',
    'OpenAIChatGPTProvider',
)

_PROVIDER_NAME = 'openai-chatgpt'
_BASE_URL = 'https://api.openai.com/v1'
_RESOURCE = 'https://api.openai.com/v1'
_UNUSABLE_REFRESH_TOKEN_ERRORS = frozenset(
    {
        'invalid_grant',
        'invalid_refresh_token',
        'token_expired',
        'refresh_token_expired',
        'refresh_token_invalidated',
        'refresh_token_reused',
    }
)
"""Refresh error codes that mean the user has to sign in again."""
_ISSUER = 'https://auth.openai.com'
_AUTHORIZE_URL = 'https://auth.openai.com/api/accounts/authorize'
_TOKEN_URL = 'https://auth.openai.com/api/accounts/oauth/token'
# The first-time registration entrypoint, never the client id to save or exchange codes with.
_REGISTRATION_CLIENT_ID = 'dynamic_agent_client'
_PLAN_USAGE_SCOPE = 'chatgpt.tokens.use.direct'
_DEFAULT_SCOPE = f'openid profile email offline_access resource.invoke {_PLAN_USAGE_SCOPE}'
# Only the port may vary between sign-ins; the scheme, host, and path are fixed by OpenAI.
_REDIRECT_URI = 'http://127.0.0.1:1455/auth/callback'
_REDIRECT_PATH = '/auth/callback'
_HOST_ID_PREFIXES = ('urn:uuid:', 'urn:ietf:params:oauth:jwk-thumbprint:', 'did:key:')
# Tolerated clock skew when checking the ID token's expiry.
_CLOCK_SKEW = timedelta(seconds=60)


@dataclass
class OpenAIChatGPTCredentials:
    """Credentials for using one ChatGPT account's plan, tied to the OAuth client registered for it.

    Persist every field together: `client_id` identifies the registration to reuse on the next sign-in,
    `id_token` is passed back as that sign-in's `id_token_hint`, and `scope` records whether the user
    granted plan usage. The tokens are excluded from `repr` so a logged or traceback-embedded instance
    does not leak them.
    """

    _: KW_ONLY
    client_id: str
    """The OAuth client id OpenAI issued when this ChatGPT account first signed in."""
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    id_token: str | None = field(default=None, repr=False)
    """The ID token from the latest sign-in or refresh, to pass as `id_token_hint` when signing in again."""
    scope: str = ''
    """The space-separated scopes the user granted."""

    @property
    def plan_usage_granted(self) -> bool:
        """Whether the user granted the `chatgpt.tokens.use.direct` scope, which inference requires."""
        return _PLAN_USAGE_SCOPE in self.scope.split()


@dataclass
class _TokenResponse:
    """The fields of an OAuth token-endpoint response that credentials are built from.

    `refresh_token` is required because the flow always requests the `offline_access` scope, and
    credentials without it could not survive their first expiry.
    """

    access_token: Annotated[str, Field(min_length=1)]
    refresh_token: Annotated[str, Field(min_length=1)]
    id_token: str | None = None
    scope: str | None = None


@dataclass
class _IdTokenClaims:
    """The ID token claims validated after a sign-in."""

    iss: str | None = None
    aud: str | list[str] | None = None
    sub: str | None = None
    nonce: str | None = None


_token_response_ta = TypeAdapter(_TokenResponse)
_id_token_claims_ta = TypeAdapter(_IdTokenClaims)


def _validate_id_token(id_token: str, *, client_id: str, nonce: str, subject: str | None) -> None:
    """Check the ID token's issuer, audience, expiry, nonce, and (when signing in again) subject.

    The signature is not verified: the token comes straight from the token endpoint over TLS, which
    OpenID Connect Core 1.0 section 3.1.3.7 accepts in place of the signature for this flow.
    """
    claims = jwt_claims(id_token, _id_token_claims_ta)
    if claims is None:
        raise CredentialsRefreshError('The token response carried a malformed ID token.', _PROVIDER_NAME)
    audiences = [claims.aud] if isinstance(claims.aud, str) else claims.aud or []
    if claims.iss != _ISSUER:
        raise CredentialsRefreshError(f'The ID token was issued by {claims.iss!r}, not {_ISSUER!r}.', _PROVIDER_NAME)
    if client_id not in audiences:
        raise CredentialsRefreshError(f'The ID token was not issued for client {client_id!r}.', _PROVIDER_NAME)
    expires_at = jwt_expires_at(id_token)
    if expires_at is None or expires_at + _CLOCK_SKEW < datetime.now(UTC):
        raise CredentialsRefreshError('The ID token has expired; start the sign-in again.', _PROVIDER_NAME)
    if claims.nonce != nonce:
        raise CredentialsRefreshError(
            "The ID token does not carry this sign-in attempt's `nonce`; start the sign-in again.", _PROVIDER_NAME
        )
    if not claims.sub:
        raise CredentialsRefreshError('The ID token does not identify an account.', _PROVIDER_NAME)
    if subject is not None and claims.sub != subject:
        raise UserError(
            'The sign-in completed for a different ChatGPT account than `id_token_hint` identifies; '
            'its credentials were not used.'
        )


async def _refresh_credentials(
    credentials: OpenAIChatGPTCredentials, *, http_client: httpx2.AsyncClient | None = None
) -> OpenAIChatGPTCredentials:
    """Exchange the refresh token for a new credential set against the account's issued client.

    Raises [`CredentialsRefreshError`][pydantic_ai.providers.openai_chatgpt.CredentialsRefreshError]
    when the token endpoint rejects the grant or returns a malformed response.
    """
    data = await _post_token_request(
        _TOKEN_URL,
        {
            'grant_type': 'refresh_token',
            'refresh_token': credentials.refresh_token,
            'client_id': credentials.client_id,
            'resource': _RESOURCE,
        },
        _token_response_ta,
        provider_name=_PROVIDER_NAME,
        http_client=http_client,
        rejected_grant_errors=_UNUSABLE_REFRESH_TOKEN_ERRORS,
    )
    return OpenAIChatGPTCredentials(
        client_id=credentials.client_id,
        access_token=data.access_token,
        refresh_token=data.refresh_token,
        id_token=data.id_token or credentials.id_token,
        # Omitting `scope` from a refresh keeps the original grant.
        scope=credentials.scope if data.scope is None else data.scope,
    )


class OpenAIChatGPTCredentialSource(OAuthCredentialSource[OpenAIChatGPTCredentials], Protocol):
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


class OpenAIChatGPTOAuthFlow(OAuthFlow[OpenAIChatGPTCredentials]):
    """Authorization-code + PKCE context for Sign in with ChatGPT with plan usage.

    Without a `client_id`, the sign-in registers a new OAuth client for the user's ChatGPT account; the
    resulting credentials carry the issued `client_id`, which later sign-ins to the same account pass back
    so they reuse the registration. Construction does no I/O: build the context anywhere, send the user to
    `authorization_url()`, then let `exchange_code_from_callback()` receive the redirect on `127.0.0.1` and
    exchange its code. The browser and credential storage stay caller-owned.
    """

    _bound_params = frozenset({'client_id', 'redirect_uri', 'state', 'code_challenge', 'nonce', 'resource'})

    def __init__(
        self,
        *,
        app_name: str,
        host_id: str | None = None,
        client_id: str | None = None,
        id_token_hint: str | None = None,
        redirect_uri: str = _REDIRECT_URI,
        state: str | None = None,
    ) -> None:
        """Create a new flow context. Construction does no I/O.

        Args:
            app_name: Your app's name, shown to the user when a new registration is approved. Use the
                same name across installations; it is only sent when registering.
            host_id: A stable, opaque identifier for this installation of your app, created once and
                persisted, such as `uuid.uuid4().urn`. OpenAI also accepts RFC 9278 JWK thumbprint URIs
                and `did:key` identifiers.
            client_id: The `client_id` of credentials from an earlier sign-in to the same ChatGPT account,
                to reuse its registration. `None` registers a new client.
            id_token_hint: The `id_token` of credentials from an earlier sign-in, so the user skips
                choosing the account. The new sign-in must then complete for the same account.
            redirect_uri: The loopback callback the authorization code is delivered to. OpenAI requires
                `http://127.0.0.1:PORT/auth/callback`; only the port may differ from the default.
            state: The CSRF token bound to the callback; auto-generated when `None`.
        """
        parsed = urlsplit(redirect_uri)
        try:
            port = parsed.port
        except ValueError:
            port = None
        if not port or redirect_uri != f'http://127.0.0.1:{port}{_REDIRECT_PATH}':
            raise UserError(
                f'`redirect_uri` must be `http://127.0.0.1:PORT{_REDIRECT_PATH}`: Sign in with ChatGPT only '
                f'lets the port vary, got {redirect_uri!r}.'
            )
        if host_id is not None and not host_id.startswith(_HOST_ID_PREFIXES):
            raise UserError(
                '`host_id` must be an opaque `urn:uuid`, `urn:ietf:params:oauth:jwk-thumbprint`, or `did:key` '
                f'identifier, got {host_id!r}.'
            )
        self._hint_subject: str | None = None
        if id_token_hint is not None:
            claims = jwt_claims(id_token_hint, _id_token_claims_ta)
            if claims is None or not claims.sub:
                raise UserError('`id_token_hint` is not an ID token from an earlier sign-in.')
            self._hint_subject = claims.sub
        super().__init__(redirect_uri=redirect_uri, state=state)
        self.app_name = app_name
        self.host_id = host_id
        self.client_id = client_id
        self.id_token_hint = id_token_hint
        self.nonce = secrets.token_urlsafe(16)

    def authorization_url(self, *, scope: str | None = None, extra_params: Mapping[str, str] | None = None) -> str:
        """The URL to send the user to.

        The URL carries `id_token_hint` when one was given, so keep it out of logs.

        Args:
            scope: The OAuth scopes to request; `None` means the identity scopes plus plan usage. Keep
                `offline_access` and `chatgpt.tokens.use.direct`: the credentials need a refresh token,
                and the provider refuses credentials without plan usage.
            extra_params: Additional query parameters, merged over the defaults (so they can also
                override them), except `client_id`, `redirect_uri`, `state`, `code_challenge`, `nonce`,
                and `resource`, which the callback validation and code exchange rely on. Pass
                `{'prompt': 'consent'}` to ask a user who declined plan usage again, or
                `{'login_hint': email}` to preselect an account.
        """
        params: dict[str, str] = {
            'response_type': 'code',
            'client_id': self.client_id or _REGISTRATION_CLIENT_ID,
            'redirect_uri': self.redirect_uri,
            'scope': scope if scope is not None else _DEFAULT_SCOPE,
            'resource': _RESOURCE,
            'state': self.state,
            'nonce': self.nonce,
            'code_challenge': self.code_challenge,
            'code_challenge_method': 'S256',
        }
        if self.client_id is None:
            params['agent_name_hint'] = self.app_name
        if self.host_id is not None:
            params['ext_agent_host_id'] = self.host_id
        if self.id_token_hint is not None:
            params['id_token_hint'] = self.id_token_hint
        return f'{_AUTHORIZE_URL}?{urlencode(self._merge_extra_params(params, extra_params))}'

    async def exchange_code(self, code: str, *, client_id: str | None = None) -> OpenAIChatGPTCredentials:
        """Exchange an authorization code for credentials (call this in your callback handler).

        The returned credentials may lack plan usage when the user did not grant it; check
        [`plan_usage_granted`][pydantic_ai.providers.openai_chatgpt.OpenAIChatGPTCredentials.plan_usage_granted],
        and persist them either way so the next sign-in reuses the registration.

        Args:
            code: The callback's `code` parameter.
            client_id: The callback's `client_id` parameter, which carries the issued client id after a
                new registration. A sign-in that reused `client_id` may omit it. Once accepted, it is
                kept as the flow's `client_id`: if the exchange fails, retry with a new flow created
                with `client_id=flow.client_id`, so the retry reuses the registration.
        """
        if self.client_id is not None:
            if client_id is not None and client_id != self.client_id:
                raise UserError(
                    f'The authorization callback carried client {client_id!r} instead of {self.client_id!r}; '
                    'its code was not exchanged.'
                )
            client_id = self.client_id
        elif client_id is None or client_id == _REGISTRATION_CLIENT_ID:
            raise UserError(
                'The registration is incomplete: the authorization callback did not carry the issued `client_id`.'
            )
        else:
            self.client_id = client_id
        data = await _post_token_request(
            _TOKEN_URL,
            {
                'grant_type': 'authorization_code',
                'code': code,
                'code_verifier': self.code_verifier,
                'redirect_uri': self.redirect_uri,
                'client_id': client_id,
                'resource': _RESOURCE,
            },
            _token_response_ta,
            provider_name=_PROVIDER_NAME,
        )
        if data.id_token is None:
            raise CredentialsRefreshError('The token response carried no ID token.', _PROVIDER_NAME)
        _validate_id_token(data.id_token, client_id=client_id, nonce=self.nonce, subject=self._hint_subject)
        return OpenAIChatGPTCredentials(
            client_id=client_id,
            access_token=data.access_token,
            refresh_token=data.refresh_token,
            id_token=data.id_token,
            scope=data.scope or '',
        )

    async def _exchange_callback(self, params: Mapping[str, str]) -> OpenAIChatGPTCredentials:
        return await self.exchange_code(params['code'], client_id=params.get('client_id'))


class OpenAIChatGPTProvider(OAuthCredentialsProvider[OpenAIChatGPTCredentials]):
    """Provider for ChatGPT plan usage through Sign in with ChatGPT.

    Sends Responses API requests to `https://api.openai.com/v1` with the signed-in user's OAuth access
    token instead of an API key, refreshing it as needed. One provider instance carries one set of
    credentials (there is no process-global cache). The instance binds its refresh lock to the first
    event loop that awaits a request, so do not reuse it across loops.

    ```python {test="skip" lint="skip"}
    provider = OpenAIChatGPTProvider(credential_source=YourCredentialStore())
    agent = Agent(OpenAIResponsesModel('gpt-6.1-sol', provider=provider))
    ```
    """

    @property
    def name(self) -> str:
        return _PROVIDER_NAME

    @property
    def base_url(self) -> str:
        return _BASE_URL

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        # The public OpenAI API, so first-party flags apply beneath the plan-usage dialect.
        return merge_profile(OpenAIProvider.model_profile(model_name), openai_chatgpt_model_profile(model_name))

    def __init__(
        self,
        credentials: OpenAIChatGPTCredentials | None = None,
        *,
        credential_source: OpenAIChatGPTCredentialSource | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new OpenAI ChatGPT provider.

        Args:
            credentials: The credentials from
                [`OpenAIChatGPTOAuthFlow`][pydantic_ai.providers.openai_chatgpt.OpenAIChatGPTOAuthFlow].
                Refreshed tokens then live in memory only. Pydantic AI never falls back to `OPENAI_API_KEY`.
            credential_source: Application-owned storage for the credentials, so refreshed tokens
                are persisted between runs; see
                [`OpenAIChatGPTCredentialSource`][pydantic_ai.providers.openai_chatgpt.OpenAIChatGPTCredentialSource].
                Mutually exclusive with `credentials`.
            openai_client: An existing `AsyncOpenAI` client to use as-is. Opts out of credential
                injection entirely; `credentials`, `credential_source`, and `http_client` must
                be `None`.
            http_client: An existing `httpx2.AsyncClient` to use. Must be dedicated to this
                provider (no auth of its own): the provider attaches its credential-injecting auth to it, and
                sharing a client between providers would mix their credentials. The auth only
                injects credentials on HTTPS requests to `api.openai.com`, so the client can safely
                be reused for other destinations.
        """
        if openai_client is None and credentials is None and credential_source is None:
            raise UserError(
                '`OpenAIChatGPTProvider` needs `credentials` or a `credential_source`: '
                'sign in with `OpenAIChatGPTOAuthFlow` first.'
            )
        super().__init__(
            credentials, credential_source=credential_source, openai_client=openai_client, http_client=http_client
        )

    def _auth_headers(self, credentials: OpenAIChatGPTCredentials) -> dict[str, str]:
        return {'Authorization': f'Bearer {credentials.access_token}'}

    async def _refresh_grant(
        self, credentials: OpenAIChatGPTCredentials, http_client: httpx2.AsyncClient
    ) -> OpenAIChatGPTCredentials:
        return await _refresh_credentials(credentials, http_client=http_client)

    def _validate_credentials(self, credentials: OpenAIChatGPTCredentials) -> None:
        if not credentials.plan_usage_granted:
            raise UserError(
                f'These credentials do not grant ChatGPT plan usage (`{_PLAN_USAGE_SCOPE}`). Sign in again with '
                f'`OpenAIChatGPTOAuthFlow(client_id={credentials.client_id!r}, ...)`, passing '
                "`extra_params={'prompt': 'consent'}` to `authorization_url()`, and approve using your ChatGPT plan."
            )
