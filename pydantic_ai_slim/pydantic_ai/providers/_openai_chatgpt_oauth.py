"""Registration and verified OAuth credentials for Sign in with ChatGPT."""

from __future__ import annotations as _annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx2
from pydantic import Field, TypeAdapter, ValidationError

from .._http import create_async_httpx2_client
from ..exceptions import ModelAPIError, UserError
from ._oauth import OAuthFlow

try:
    import jwt
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai-chatgpt` optional group to use Sign in with ChatGPT: '
        '`pip install "pydantic-ai-slim[openai-chatgpt]"`'
    ) from _import_error

_ISSUER = 'https://auth.openai.com'
_RESOURCE = 'https://api.openai.com/v1'
_DYNAMIC_CLIENT_ID = 'dynamic_agent_client'
_SCOPES = 'openid profile email offline_access resource.invoke chatgpt.tokens.use.direct'
_REQUIRED_SCOPES = frozenset({'resource.invoke', 'chatgpt.tokens.use.direct'})


@dataclass(kw_only=True)
class OpenAIChatGPTClient:
    """An OpenAI-provisioned OAuth client, not an issued OSS registration.

    The application's backend owns this configuration and any secret. Configuring a client
    does not grant ChatGPT plan usage: the client must separately be approved for that grant.
    """

    client_id: str
    redirect_uri: str
    token_endpoint_auth_method: Literal['none', 'client_secret_basic', 'client_secret_post'] = 'none'
    client_secret: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        redirect = urlsplit(self.redirect_uri)
        if (
            not self.client_id
            or self.client_id == _DYNAMIC_CLIENT_ID
            or redirect.scheme != 'https'
            or not redirect.hostname
            or redirect.username
            or redirect.password
            or redirect.fragment
        ):
            raise UserError(
                'A provisioned ChatGPT client needs its client ID and exact registered HTTPS `redirect_uri`.'
            )
        if (self.token_endpoint_auth_method == 'none') != (self.client_secret is None):
            raise UserError(
                'Public clients must omit `client_secret`; confidential clients must supply it server-side.'
            )


@dataclass(kw_only=True)
class OpenAIChatGPTCredentials:
    """One ChatGPT account registration and its renewable token set.

    Keep the complete record in protected application storage, including the issued client ID,
    stable host ID and callback URI. Tokens are hidden from `repr`, not from serialization.
    """

    subject: str
    client_id: str
    ext_agent_host_id: str
    redirect_uri: str
    expires_at: datetime
    scopes: tuple[str, ...]
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    id_token: str = field(repr=False)
    email: str | None = None
    earliest_refresh_at: datetime | None = None


@dataclass
class _TokenResponse:
    access_token: Annotated[str, Field(min_length=1)]
    refresh_token: Annotated[str, Field(min_length=1)]
    token_type: Literal['Bearer', 'bearer']
    expires_in: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    id_token: Annotated[str, Field(min_length=1)] | None = None
    scope: str | None = None
    earliest_refresh_at: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None


@dataclass
class _Discovery:
    issuer: Literal['https://auth.openai.com']
    jwks_uri: str
    id_token_signing_alg_values_supported: list[str]


@dataclass
class _Identity:
    sub: Annotated[str, Field(min_length=1)]
    aud: str | list[str]
    nonce: str | None = None
    azp: str | None = None
    email: str | None = None


_TOKEN_ADAPTER = TypeAdapter(_TokenResponse)
_DISCOVERY_ADAPTER = TypeAdapter(_Discovery)
_IDENTITY_ADAPTER = TypeAdapter(_Identity)


async def _request(
    method: str,
    url: str,
    *,
    http_client: httpx2.AsyncClient | None,
    data: dict[str, str] | None = None,
    auth: httpx2.BasicAuth | None = None,
) -> httpx2.Response:
    # Explicit auth disables a provider's inference auth on token and discovery requests.
    if http_client is not None:
        return await http_client.request(method, url, data=data, auth=auth)
    async with create_async_httpx2_client() as client:
        return await client.request(method, url, data=data, auth=auth)


def _error(message: str) -> ModelAPIError:
    return ModelAPIError(model_name='openai-chatgpt', message=message)


async def _exchange(
    data: dict[str, str],
    *,
    client: OpenAIChatGPTClient | None,
    http_client: httpx2.AsyncClient | None,
) -> _TokenResponse:
    auth = None
    if client is not None and client.client_secret is not None:
        if client.token_endpoint_auth_method == 'client_secret_basic':
            auth = httpx2.BasicAuth(client.client_id, client.client_secret)
        else:
            data['client_secret'] = client.client_secret
    response = await _request(
        'POST', f'{_ISSUER}/api/accounts/oauth/token', http_client=http_client, data=data, auth=auth
    )
    if response.status_code != 200:
        # OAuth error bodies may contain tokens/codes; do not include them in exceptions.
        raise _error(f'ChatGPT token exchange failed (HTTP {response.status_code}).')
    try:
        return _TOKEN_ADAPTER.validate_json(response.content)
    except ValidationError:
        raise _error('ChatGPT returned an invalid token response.') from None


async def _verify_identity(
    id_token: str,
    *,
    client_id: str,
    nonce: str | None,
    subject: str | None,
    http_client: httpx2.AsyncClient | None,
) -> _Identity:
    discovery_response = await _request('GET', f'{_ISSUER}/.well-known/openid-configuration', http_client=http_client)
    if discovery_response.status_code != 200:
        raise _error('OpenAI discovery failed.')
    try:
        discovery = _DISCOVERY_ADAPTER.validate_json(discovery_response.content)
        uri = urlsplit(discovery.jwks_uri)
        if uri.scheme != 'https' or uri.netloc != 'auth.openai.com' or uri.fragment:
            raise ValueError('Invalid JWKS endpoint')
    except (ValidationError, ValueError):
        raise _error('OpenAI returned an invalid discovery document.') from None
    keys_response = await _request('GET', discovery.jwks_uri, http_client=http_client)
    if keys_response.status_code != 200:
        raise _error('OpenAI signing keys could not be loaded.')
    try:
        keys = jwt.PyJWKSet.from_dict(keys_response.json())
        key = keys[jwt.get_unverified_header(id_token)['kid']]
        algorithms = [
            alg
            for alg in discovery.id_token_signing_alg_values_supported
            if alg in ('RS256', 'RS384', 'RS512', 'ES256', 'ES384', 'ES512')
        ]
        claims = jwt.decode(
            id_token,
            key.key,
            algorithms=algorithms,
            audience=client_id,
            issuer=_ISSUER,
            options={'require': ['sub', 'exp', 'iat', 'iss', 'aud']},
        )
        identity = _IDENTITY_ADAPTER.validate_python(claims)
        if (nonce is not None and identity.nonce != nonce) or (identity.azp is not None and identity.azp != client_id):
            raise ValueError('Identity does not match authorization')
        if isinstance(identity.aud, list) and len(identity.aud) > 1 and identity.azp != client_id:
            raise ValueError('Missing authorized party for multiple audiences')
        if subject is not None and identity.sub != subject:
            raise ValueError('Identity changed')
    except (jwt.PyJWTError, ValidationError, ValueError, KeyError, TypeError):
        raise _error('The ChatGPT ID token could not be verified against this registration.') from None
    return identity


async def _credentials(
    tokens: _TokenResponse,
    *,
    client_id: str,
    host_id: str,
    redirect_uri: str,
    nonce: str | None,
    subject: str | None,
    http_client: httpx2.AsyncClient | None,
    previous: OpenAIChatGPTCredentials | None = None,
) -> OpenAIChatGPTCredentials:
    if tokens.id_token is None:
        if previous is None:
            raise _error('ChatGPT sign-in did not return an ID token.')
        identity = _Identity(sub=previous.subject, aud=client_id, email=previous.email)
        id_token = previous.id_token
    else:
        id_token = tokens.id_token
        identity = await _verify_identity(
            id_token, client_id=client_id, nonce=nonce, subject=subject, http_client=http_client
        )
    if tokens.scope is None:
        if previous is None:
            raise _error('ChatGPT sign-in did not return granted scopes.')
        scopes = previous.scopes
    else:
        scopes = tuple(tokens.scope.split())
    if not _REQUIRED_SCOPES.issubset(scopes):
        raise _error('ChatGPT plan usage was not granted. Authorize plan usage before inference.')
    try:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=tokens.expires_in)
        earliest = (
            datetime.fromtimestamp(tokens.earliest_refresh_at, timezone.utc) if tokens.earliest_refresh_at else None
        )
    except (ValueError, OverflowError, OSError):
        raise _error('ChatGPT returned invalid token expiry information.') from None
    return OpenAIChatGPTCredentials(
        subject=identity.sub,
        client_id=client_id,
        ext_agent_host_id=host_id,
        redirect_uri=redirect_uri,
        expires_at=expires_at,
        scopes=scopes,
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        id_token=id_token,
        email=identity.email,
        earliest_refresh_at=earliest,
    )


class OpenAIChatGPTOAuthFlow(OAuthFlow[OpenAIChatGPTCredentials]):
    """One Sign in with ChatGPT authorization attempt with PKCE and verified OIDC identity.

    Construction does no I/O. Start the listener before opening the system browser. For a web
    application, keep this object server-side and bind its callback to the user's app session.
    """

    def __init__(
        self,
        *,
        ext_agent_host_id: str,
        agent_name: str,
        redirect_uri: str = 'http://127.0.0.1:1455/auth/callback',
        credentials: OpenAIChatGPTCredentials | None = None,
        client: OpenAIChatGPTClient | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        """Prepare an OSS registration, reauthorization, or approved provisioned-client login.

        Args:
            ext_agent_host_id: Stable, application-persisted identifier of this runtime host.
            agent_name: The application's actual name, used only for initial registration.
            redirect_uri: OSS HTTP loopback callback. On reauthorization only its port may change.
            credentials: The selected registration to reauthorize, retaining identity and login hints.
            client: Separately provisioned client configuration. Uses its registered callback instead
                of `redirect_uri`. Does not imply approval for ChatGPT plan usage.
            http_client: Optional client for token exchange, discovery, and JWKS requests.
        """
        if not ext_agent_host_id or not agent_name:
            raise UserError('`ext_agent_host_id` and `agent_name` must not be empty.')
        if client is not None:
            redirect_uri = client.redirect_uri
        else:
            parsed = urlsplit(redirect_uri)
            if (
                parsed.scheme != 'http'
                or parsed.hostname != '127.0.0.1'
                or not parsed.port
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise UserError('OSS ChatGPT login requires an HTTP callback on `127.0.0.1` with a port.')
        if credentials is not None:
            old, new = urlsplit(credentials.redirect_uri), urlsplit(redirect_uri)
            if credentials.ext_agent_host_id != ext_agent_host_id:
                raise UserError('The ChatGPT registration belongs to another host.')
            if client is None and (old.scheme, old.hostname, old.path) != (new.scheme, new.hostname, new.path):
                raise UserError('Only the callback port may change for an existing OSS ChatGPT registration.')
            if client is not None and credentials.client_id != client.client_id:
                raise UserError('The credentials belong to another ChatGPT client.')
        super().__init__(redirect_uri=redirect_uri)
        self.nonce = secrets.token_urlsafe(32)
        self._host_id = ext_agent_host_id
        self._agent_name = agent_name
        self._credentials = credentials
        self._client = client
        self._http_client = http_client
        self._client_id = client.client_id if client else credentials.client_id if credentials else _DYNAMIC_CLIENT_ID
        self._expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        self._consumed = False

    def authorization_url(self, *, scope: str | None = None, extra_params: Mapping[str, str] | None = None) -> str:
        """Build the authorization URL. Do not log URLs containing `id_token_hint`."""
        params = {
            'client_id': self._client_id,
            'ext_agent_host_id': self._host_id,
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'scope': scope or _SCOPES,
            'resource': _RESOURCE,
            'state': self.state,
            'nonce': self.nonce,
            'code_challenge': self.code_challenge,
            'code_challenge_method': 'S256',
        }
        if self._client_id == _DYNAMIC_CLIENT_ID:
            params['agent_name_hint'] = self._agent_name
        elif self._credentials is not None:
            params['id_token_hint'] = self._credentials.id_token
            if self._credentials.email:
                params['login_hint'] = self._credentials.email
        if extra_params and (overridden := sorted(params.keys() & extra_params.keys())):
            raise UserError(f'`extra_params` cannot override flow-bound parameters: {", ".join(overridden)}.')
        return f'{_ISSUER}/api/accounts/authorize?{urlencode(self._merge_extra_params(params, extra_params))}'

    async def exchange_code(self, code: str) -> OpenAIChatGPTCredentials:
        """Exchange a code for an existing client; new registrations must use `exchange_callback`."""
        if self._client_id == _DYNAMIC_CLIENT_ID:
            raise UserError('New ChatGPT registrations require `exchange_callback` with the issued client ID.')
        return await self.exchange_callback(f'{self.redirect_uri}?{urlencode({"state": self.state, "code": code})}')

    async def exchange_callback(self, callback_url: str) -> OpenAIChatGPTCredentials:
        """Validate the full callback and exchange once. Never fetches the supplied URL."""
        try:
            url, target = urlsplit(callback_url), urlsplit(self.redirect_uri)
            params = parse_qs(url.query, keep_blank_values=True, strict_parsing=True)
            matches = (
                (url.scheme, url.netloc, url.path) == (target.scheme, target.netloc, target.path)
                and not url.fragment
                and all(len(v) == 1 for v in params.values())
                and secrets.compare_digest(params.get('state', [''])[0], self.state)
            )
        except ValueError:
            matches, params = False, {}
        if not matches:
            raise UserError('The ChatGPT callback does not match this authorization attempt.')
        if self._consumed or datetime.now(timezone.utc) >= self._expires_at:
            raise UserError('This ChatGPT authorization was consumed or expired. Start a new sign-in.')
        self._consumed = True
        if 'error' in params:
            raise UserError('ChatGPT authorization was not granted. Start a new sign-in.')
        code, issued = params.get('code', [''])[0], params.get('client_id', [''])[0]
        if self._client_id == _DYNAMIC_CLIENT_ID:
            if not issued or issued == _DYNAMIC_CLIENT_ID:
                raise UserError('ChatGPT registration did not return an issued client ID.')
        elif issued and issued != self._client_id:
            raise UserError('The ChatGPT callback changed the selected registration.')
        else:
            issued = self._client_id
        if not code:
            raise UserError('The ChatGPT callback has no authorization code.')
        tokens = await _exchange(
            {
                'grant_type': 'authorization_code',
                'code': code,
                'client_id': issued,
                'code_verifier': self.code_verifier,
                'redirect_uri': self.redirect_uri,
                'resource': _RESOURCE,
            },
            client=self._client,
            http_client=self._http_client,
        )
        return await _credentials(
            tokens,
            client_id=issued,
            host_id=self._host_id,
            redirect_uri=self.redirect_uri,
            nonce=self.nonce,
            subject=self._credentials.subject if self._credentials else None,
            http_client=self._http_client,
        )

    async def _exchange_callback_url(self, callback_url: str) -> OpenAIChatGPTCredentials:
        return await self.exchange_callback(callback_url)

    async def exchange_code_from_callback(self) -> OpenAIChatGPTCredentials:
        """Receive one OSS loopback callback. HTTPS applications use their own redirect handler."""
        if self._client is not None:
            raise UserError('Provisioned HTTPS callbacks must be handled by the application server.')
        return await super().exchange_code_from_callback()


async def refresh_credentials(
    credentials: OpenAIChatGPTCredentials,
    *,
    client: OpenAIChatGPTClient | None,
    http_client: httpx2.AsyncClient,
) -> OpenAIChatGPTCredentials:
    if credentials.earliest_refresh_at and datetime.now(timezone.utc) < credentials.earliest_refresh_at:
        raise _error('ChatGPT token refresh is not yet allowed. Retry after `earliest_refresh_at`.')
    if client is not None and credentials.client_id != client.client_id:
        raise UserError('The credentials belong to another ChatGPT client.')
    tokens = await _exchange(
        {
            'grant_type': 'refresh_token',
            'client_id': credentials.client_id,
            'refresh_token': credentials.refresh_token,
            'resource': _RESOURCE,
        },
        client=client,
        http_client=http_client,
    )
    return await _credentials(
        tokens,
        client_id=credentials.client_id,
        host_id=credentials.ext_agent_host_id,
        redirect_uri=credentials.redirect_uri,
        nonce=None,
        subject=credentials.subject,
        http_client=http_client,
        previous=credentials,
    )
