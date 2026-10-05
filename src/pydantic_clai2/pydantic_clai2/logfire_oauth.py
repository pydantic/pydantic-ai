"""Browser sign-in to Logfire with the OAuth device flow (RFC 8628), as Code Puppy's Logfire plugin does.

CLAI shows a link and a code and opens the link. That Logfire page signs the user in, or signs a new
user up, and approves the code. No local callback server is involved, so it also works over SSH: open
the link on any device. The client registers itself (RFC 7591) and uses PKCE; the authorization server
is discovered from the MCP URL (RFC 9728), so self-hosted Logfire works too.

Tokens are kept per MCP URL in CLAI's credential store (the OS keyring, or a private file), refreshed
when they expire, and replaced by a new sign-in when refreshing fails.
"""

import base64
import hashlib
import secrets
import threading
import time
import webbrowser
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import Annotated, Literal
from urllib.parse import urlsplit

import anyio
import httpx
from anyio import to_thread
from keyring.errors import KeyringError
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from pydantic_ai.exceptions import UserError
from pydantic_clai2.config.credential_store import delete_credentials, load_codex_credentials, save_codex_credentials
from pydantic_clai2.mcp import http_client
from pydantic_clai2.ui.rendering.tool_output import terminal_text

ACCOUNT = 'logfire-oauth'
"""The credential account holding Logfire sign-ins, one per MCP URL."""
READ_SCOPE = 'project:read'
"""The scope Logfire's MCP server requires; read-and-write sign-ins also ask for every scope it lists."""
SIGN_IN_TIMEOUT = 660.0
"""Seconds a first connection may wait: Logfire's device codes last 600 seconds."""
DEVICE_GRANT = 'urn:ietf:params:oauth:grant-type:device_code'
_REFRESH_MARGIN = 60.0

Announce = Callable[[str], None]
Sleep = Callable[[float], Awaitable[None]]


class SignInError(Exception):
    """Browser sign-in did not complete; the message says why and how to retry."""

    def __init__(self, message: str) -> None:
        # Messages can quote the server's own error text, so terminal controls in it are made inert.
        super().__init__(terminal_text(message))


class Tokens(BaseModel):
    """One completed sign-in and what refreshing it needs."""

    model_config = ConfigDict(extra='forbid', frozen=True)
    client_id: str
    token_endpoint: str
    access_token: str = Field(repr=False)
    refresh_token: str | None = Field(default=None, repr=False)
    expires_at: float
    writable: bool = False
    """Whether the sign-in asked for write access; asking again would get the same grant, so it serves write tools."""

    def serves(self, *, read_only: bool) -> bool:
        """Whether its scopes cover the tools in use."""
        return read_only or self.writable

    def fresh(self) -> bool:
        """Whether the access token has more than a minute left."""
        return self.expires_at - _REFRESH_MARGIN > time.time()


_STORE: TypeAdapter[dict[str, Tokens]] = TypeAdapter(dict[str, Tokens])
_ENTRIES: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
_WRITES = threading.Lock()
_UNSAVED: dict[str, Tokens] = {}
_logouts: int = 0
_MIN_INTERVAL = 1.0
"""Bumped by `forget()`, so a sign-in or refresh that started earlier knows not to save."""
"""Sign-ins the keyring or file refused, kept in memory so they last this session; guarded by `_WRITES`."""


def _load_all() -> dict[str, Tokens]:
    try:
        raw = load_codex_credentials(account=ACCOUNT)
        entries = _ENTRIES.validate_json(raw) if raw else {}
    except (UserError, ValidationError, UnicodeDecodeError, KeyringError, OSError):
        return {}  # An unreadable (or concurrently deleted) store means signing in again, not a failed connection.
    store: dict[str, Tokens] = {}
    for resource, entry in entries.items():
        try:
            store[resource] = Tokens.model_validate(entry)
        except ValidationError:
            continue  # One unreadable sign-in means signing in to that URL again, not losing the others.
    return store


def load(resource: str) -> Tokens | None:
    """The sign-in for one MCP URL, if any, even when it has expired."""
    return _UNSAVED.get(resource) or _load_all().get(resource)


def _save(resource: str, tokens: Tokens, logouts: int) -> str | None:
    """Store `tokens` unless a logout came after `logouts`; the reason when the keyring or file refused them."""
    with _WRITES:
        if logouts != _logouts:
            raise SignInError('Signed out of Logfire while signing in, so this sign-in was discarded.')
        try:
            store = _load_all()
            store[resource] = tokens
            save_codex_credentials(value=_STORE.dump_json(store).decode(), account=ACCOUNT)
        except (UserError, KeyringError, OSError) as exc:
            _UNSAVED[resource] = tokens
            return type(exc).__name__
        _UNSAVED.pop(resource, None)
        return None


def forget() -> bool:
    """Sign out everywhere, including sign-ins in progress; `False` when there was nothing to forget."""
    global _logouts
    with _WRITES:
        _logouts += 1
        try:
            stored = bool(load_codex_credentials(account=ACCOUNT))
        except (UserError, UnicodeDecodeError, KeyringError, OSError):
            stored = True  # Unreadable now is not gone: delete it so it cannot come back.
        unsaved = bool(_UNSAVED)
        _UNSAVED.clear()
        delete_credentials(account=ACCOUNT)
        return stored or unsaved


def _https(url: str) -> str:
    # Device codes and refresh tokens are sent to these URLs.
    if urlsplit(url).scheme != 'https':
        raise ValueError(f'Logfire OAuth URLs must use https, not {url}')
    return url


_Https = Annotated[str, AfterValidator(_https)]


class _Resource(BaseModel):
    resource: str
    authorization_servers: list[_Https] = Field(min_length=1)
    scopes_supported: list[str] = []


class _Server(BaseModel):
    issuer: str
    device_authorization_endpoint: _Https
    token_endpoint: _Https
    registration_endpoint: _Https | None = None


class _Registered(BaseModel):
    client_id: str = Field(min_length=1)


class _Device(BaseModel):
    device_code: str
    user_code: str
    verification_uri: _Https
    verification_uri_complete: _Https | None = None
    expires_in: float = 600
    interval: float = 5


class _Granted(BaseModel):
    access_token: str = Field(min_length=1)
    token_type: str
    refresh_token: str | None = None
    expires_in: float = 3600
    scope: str | None = None
    """Absent when the server granted exactly the scopes asked for (RFC 6749 section 5.1)."""

    @field_validator('token_type')
    @classmethod
    def _bearer(cls, token_type: str) -> str:
        # MCP sends tokens as `Authorization: Bearer`; another scheme would be sent wrongly.
        if token_type.lower() != 'bearer':
            raise ValueError(f'CLAI needs a Bearer token, not {token_type!r}')
        return token_type

    def covers(self, scope: str) -> bool:
        return self.scope is None or set(scope.split()) <= set(self.scope.split())


class _Problem(BaseModel):
    error: str = 'server_error'
    error_description: str | None = None


def _problem(response: httpx.Response) -> _Problem:
    """Logfire nests OAuth errors under `detail`; standard servers do not."""
    try:
        body: object = response.json()
    except ValueError:
        return _Problem(error_description=f'HTTP {response.status_code}')
    if isinstance(body, dict) and isinstance(detail := body.get('detail'), dict):  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        body = detail  # pyright: ignore[reportUnknownVariableType]
    try:
        return _Problem.model_validate(body)
    except ValidationError:
        return _Problem(error_description=f'HTTP {response.status_code}')


async def _discover(http: httpx.AsyncClient, resource: str) -> tuple[_Server, list[str]]:
    parts = urlsplit(resource)
    origin = f'{parts.scheme}://{parts.netloc}'
    path = '' if parts.path == '/' else parts.path
    response = await http.get(f'{origin}/.well-known/oauth-protected-resource{path}')
    if response.is_success:
        described = _Resource.model_validate_json(response.content)
        if described.resource != resource:
            # RFC 9728 section 3.3: metadata for another resource must not choose where the user signs in.
            raise SignInError(f'{resource} described itself as {described.resource}; not signing in.')
        issuer, scopes = described.authorization_servers[0], described.scopes_supported
    else:
        issuer, scopes = origin, []
    # RFC 8414 section 3.1: the well-known segment goes between the host and any issuer path.
    issuer_parts = urlsplit(issuer)
    well_known = '/.well-known/oauth-authorization-server' + issuer_parts.path.rstrip('/')
    response = await http.get(f'{issuer_parts.scheme}://{issuer_parts.netloc}{well_known}')
    response.raise_for_status()
    server = _Server.model_validate_json(response.content)
    if server.issuer.rstrip('/') != issuer.rstrip('/'):
        # RFC 8414 section 3.3: metadata served for one issuer must not be used for another.
        raise SignInError(f'{issuer} described itself as {server.issuer}; not signing in.')
    return server, scopes


async def _register(http: httpx.AsyncClient, server: _Server, scope: str) -> str:
    if server.registration_endpoint is None:
        raise SignInError('This Logfire server does not let CLAI register for browser sign-in. Use an API key.')
    response = await http.post(
        server.registration_endpoint,
        json={
            'client_name': 'CLAI',
            'client_uri': 'https://github.com/pydantic/pydantic-ai',
            'grant_types': [DEVICE_GRANT, 'refresh_token'],
            'token_endpoint_auth_method': 'none',
            'application_type': 'native',
            'scope': scope,
        },
    )
    if response.is_error:
        raise SignInError(f'Logfire refused to register CLAI: {_problem(response).error_description}')
    return _Registered.model_validate_json(response.content).client_id


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
    return verifier, challenge


def _open(url: str) -> bool:
    try:
        return webbrowser.open(url)
    except webbrowser.Error:
        return False


async def sign_in(
    *, resource: str, read_only: bool, announce: Announce, http: httpx.AsyncClient, sleep: Sleep = anyio.sleep
) -> Tokens:
    """Run the device flow now and save the result; raises `SignInError` when it is denied or expires."""
    logouts = _logouts
    try:
        server, offered = await _discover(http, resource)
        writable = not read_only
        # Without a published list, only the scope the MCP server needs is known.
        scope = ' '.join(dict.fromkeys([READ_SCOPE, *offered])) if writable else READ_SCOPE
        previous = await to_thread.run_sync(load, resource)
        # A client registered for read-only sign-in may not be allowed the write scopes.
        reusable = previous is not None and previous.token_endpoint == server.token_endpoint and not writable
        client_id = previous.client_id if previous is not None and reusable else await _register(http, server, scope)
        verifier, challenge = _pkce()
        # RFC 8707: the token is issued for this MCP URL only, so no other server can reuse it.
        form = {'scope': scope, 'resource': resource, 'code_challenge': challenge, 'code_challenge_method': 'S256'}
        response = await http.post(server.device_authorization_endpoint, data={**form, 'client_id': client_id})
        if reusable and response.is_error and _problem(response).error == 'invalid_client':
            # The server forgot the client CLAI registered earlier; register again once.
            client_id = await _register(http, server, scope)
            response = await http.post(server.device_authorization_endpoint, data={**form, 'client_id': client_id})
        if response.is_error:
            raise SignInError(f'Logfire refused browser sign-in: {_problem(response).error_description}')
        device = _Device.model_validate_json(response.content)
        link = device.verification_uri_complete or device.verification_uri
        announce(f'Sign in to Logfire (new users can sign up there): open {link}')
        announce(f'Enter code: {device.user_code}')
        announce('Approve only the code shown here. You can open the link on another device.')
        if not await to_thread.run_sync(_open, link):
            announce('No browser opened; open the link above yourself.')
        granted = await _poll(
            http, server=server, resource=resource, client_id=client_id, device=device, verifier=verifier, sleep=sleep
        )
        if not granted.covers(READ_SCOPE):
            raise SignInError(f'Logfire did not grant {READ_SCOPE}, which the MCP server needs.')
        if not granted.covers(scope):
            # Asking again would get the same grant, so say so once rather than on every request.
            announce('Logfire granted fewer scopes than asked; some write tools may be refused.')
    except (httpx.HTTPError, ValidationError) as exc:
        raise SignInError(f'Logfire sign-in failed: {type(exc).__name__}. Run /logfire_mcp login to retry.') from exc
    tokens = Tokens(
        client_id=client_id,
        token_endpoint=server.token_endpoint,
        access_token=granted.access_token,
        refresh_token=granted.refresh_token,
        expires_at=time.time() + granted.expires_in,
        writable=writable,
    )
    problem = await to_thread.run_sync(_save, resource, tokens, logouts)
    # The approval is spent either way, so an unsaved sign-in still serves this session.
    announce(
        'Signed in to Logfire.'
        if problem is None
        else f'Signed in to Logfire for this session only: saving the sign-in failed ({problem}).'
    )
    return tokens


async def _poll(
    http: httpx.AsyncClient,
    *,
    server: _Server,
    resource: str,
    client_id: str,
    device: _Device,
    verifier: str,
    sleep: Sleep,
) -> _Granted:
    interval = max(device.interval, _MIN_INTERVAL)  # A zero or negative interval would poll without pause.
    deadline = time.monotonic() + device.expires_in
    while time.monotonic() < deadline:
        # A server-chosen interval cannot outlast the code.
        await sleep(max(min(interval, deadline - time.monotonic()), 0))
        # Logfire requires PKCE on the device flow too, so the verifier goes with the device code.
        try:
            response = await http.post(
                server.token_endpoint,
                data={
                    'grant_type': DEVICE_GRANT,
                    'device_code': device.device_code,
                    'client_id': client_id,
                    'code_verifier': verifier,
                    'resource': resource,
                },
            )
        except httpx.TransportError:
            interval += 5  # Back off as for `slow_down`, even when the server asked for no wait.
            continue  # A slow or dropped poll is retried; the approval on the Logfire page still stands.
        if response.is_success:
            return _Granted.model_validate_json(response.content)
        problem = _problem(response)
        if problem.error == 'slow_down':
            interval += 5
        elif problem.error != 'authorization_pending':
            reason = 'was denied' if problem.error == 'access_denied' else f'failed: {problem.error_description}'
            raise SignInError(f'Logfire sign-in {reason}. Run /logfire_mcp login to retry.')
    raise SignInError('The Logfire sign-in code expired before it was approved. Run /logfire_mcp login to retry.')


async def _refresh(http: httpx.AsyncClient, *, resource: str, tokens: Tokens) -> Tokens | None:
    if tokens.refresh_token is None:
        return None
    logouts = _logouts
    try:
        response = await http.post(
            tokens.token_endpoint,
            data={
                'grant_type': 'refresh_token',
                'refresh_token': tokens.refresh_token,
                'client_id': tokens.client_id,
                'resource': resource,
            },
        )
        if response.is_error:
            return None
        granted = _Granted.model_validate_json(response.content)
    except (httpx.HTTPError, ValidationError):
        return None
    if not granted.covers(READ_SCOPE):
        return None  # Unusable for MCP; signing in again beats replacing a working token with it.
    refreshed = tokens.model_copy(
        update={
            'access_token': granted.access_token,
            'refresh_token': granted.refresh_token or tokens.refresh_token,
            'expires_at': time.time() + granted.expires_in,
        }
    )
    # Unsaved, it lasts this session; a logout since the refresh began discards it instead.
    await to_thread.run_sync(_save, resource, refreshed, logouts)
    return refreshed


class DeviceAuth(httpx.Auth):
    """Bearer tokens from a stored sign-in; refreshes them, or starts a browser sign-in when there is none."""

    def __init__(
        self,
        *,
        resource: str,
        read_only: bool,
        announce: Announce,
        http: Callable[[], httpx.AsyncClient] = http_client,
        sleep: Sleep = anyio.sleep,
    ) -> None:
        """Tokens for `resource`, the MCP URL; `announce` shows the sign-in link and code."""
        self._resource = resource
        self._read_only = read_only
        self._announce = announce
        self._http = http
        self._sleep = sleep
        self._lock = anyio.Lock()

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        """Unsupported: signing in waits on the network, and MCP clients are async."""
        raise RuntimeError('Logfire sign-in needs an async client.')

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Send the token; on a 401, refresh or sign in again and retry once."""
        tokens = await self._tokens(rejected=None)
        request.headers['Authorization'] = f'Bearer {tokens.access_token}'
        response = yield request
        if response.status_code == 401:
            tokens = await self._tokens(rejected=tokens)
            request.headers['Authorization'] = f'Bearer {tokens.access_token}'
            yield request

    async def _tokens(self, *, rejected: Tokens | None) -> Tokens:
        # One sign-in at a time: an MCP connection sends several requests at once.
        async with self._lock:
            tokens = await to_thread.run_sync(load, self._resource)
            if tokens is None or not tokens.serves(read_only=self._read_only):
                return await self.sign_in()
            if tokens != rejected and tokens.fresh():
                return tokens  # Another request, or another CLAI process, may have refreshed it already.
            async with self._http() as http:
                refreshed = await _refresh(http, resource=self._resource, tokens=tokens)
            return refreshed or await self.sign_in()

    async def sign_in(self) -> Tokens:
        """Run the device flow now, announcing the link and code."""
        async with self._http() as http:
            return await sign_in(
                resource=self._resource,
                read_only=self._read_only,
                announce=self._announce,
                http=http,
                sleep=self._sleep,
            )


Status = Literal['signed in', 'expired', 'signed out']


def status(*, resource: str, read_only: bool) -> Status:
    """Whether runs can use a stored sign-in; `expired` ones refresh or sign in again on the next run."""
    tokens = load(resource)
    if tokens is None or not tokens.serves(read_only=read_only):
        return 'signed out'
    return 'signed in' if tokens.fresh() or tokens.refresh_token else 'expired'
