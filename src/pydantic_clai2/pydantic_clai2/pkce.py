"""Browser sign-in for a registered public OAuth client: authorization code, S256 PKCE, rotating refresh.

For services that need a registered app but accept PKCE in place of a client secret, such as Slack once an
app enables PKCE, or any MCP server without Dynamic Client Registration. Core's `OAuthFlow` supplies `state`, the
PKCE pair, and the one-shot localhost callback. This module adds the RFC 6749 token exchange and refresh, and
keeps the tokens in CLAI's credential store under one account, never in settings.

A plugin describes its client once and asks for a token per run:

```python
sign_in = PKCESignIn(client=CLIENT, account='slack-oauth', service='Slack', setup='/plugins configure slack')
await sign_in.sign_in()  # from a menu or command: opens the browser, waits for the callback
token = await sign_in.token()  # per run: refreshes when close to expiry, raises `UserError` naming `setup`
```
"""

import asyncio
import time
import webbrowser
from collections.abc import Callable, Coroutine, Mapping
from contextlib import AbstractContextManager
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import anyio
import httpx
from anyio import CancelScope, fail_after
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers._oauth import OAuthFlow
from pydantic_clai2.config.credential_store import (
    credential_lock,
    delete_credentials,
    load_codex_credentials,
    save_codex_credentials,
)

REFRESH_MARGIN = 300
"""Seconds before expiry at which a token is refreshed, so a run does not start with one about to lapse."""


class PublicClient(BaseModel):
    """A registered OAuth app that signs in without a client secret. Nothing here is secret."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    authorize_url: str
    token_url: str
    client_id: str = Field(min_length=1)
    redirect_uri: str
    """A `http://localhost:PORT/...` or `http://127.0.0.1:PORT/...` URL registered with the app; the port is fixed."""
    scopes: tuple[str, ...] = ()
    scope_separator: str = ' '


class Tokens(BaseModel):
    """What a sign-in produced, tied to the client that can refresh it."""

    client_id: str
    access_token: SecretStr
    refresh_token: SecretStr | None = None
    expires_at: float | None = None
    scopes: tuple[str, ...] = ()
    """What the service granted; a sign-in saved without them grants nothing a client asks for."""

    def grants(self, client: PublicClient) -> bool:
        """Whether these tokens are for `client` and carry every scope it asks for."""
        return self.client_id == client.client_id and set(client.scopes) <= set(self.scopes)

    def stale(self, now: float) -> bool:
        """Whether the access token expires within `REFRESH_MARGIN` seconds."""
        return self.expires_at is not None and self.expires_at - REFRESH_MARGIN <= now

    def stored(self) -> str:
        """The credential-store JSON: the one serialization that reveals the tokens (`model_dump` masks them)."""
        return _Stored(
            client_id=self.client_id,
            access_token=self.access_token.get_secret_value(),
            refresh_token=None if self.refresh_token is None else self.refresh_token.get_secret_value(),
            expires_at=self.expires_at,
            scopes=self.scopes,
        ).model_dump_json(exclude_none=True)


class _Stored(BaseModel):
    client_id: str
    access_token: str
    refresh_token: str | None
    expires_at: float | None
    scopes: tuple[str, ...]


class _TokenResponse(BaseModel):
    """RFC 6749 section 5, read leniently: some services (Slack) report errors with HTTP 200 and `ok: false`."""

    access_token: str | None = None
    refresh_token: str | None = None
    expires_in: float | None = None
    scope: str | None = None
    error: str | None = None


async def _request_tokens(
    client: PublicClient,
    form: Mapping[str, str],
    *,
    requested: tuple[str, ...],
    service: str,
    transport: httpx.AsyncBaseTransport | None,
) -> Tokens:
    """A token response as `Tokens`; one without `scope` granted what was `requested` (RFC 6749 section 5.1)."""
    async with httpx.AsyncClient(transport=transport, timeout=30) as http:
        try:
            response = await http.post(client.token_url, data=dict(form), headers={'Accept': 'application/json'})
            body = _TokenResponse.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError):
            raise UserError(f'{service} did not answer the sign-in request. Try again.') from None
    if not body.access_token:
        raise UserError(f'{service} refused the sign-in: {body.error or f"HTTP {response.status_code}"}.')
    return Tokens(
        client_id=client.client_id,
        access_token=SecretStr(body.access_token),
        refresh_token=SecretStr(body.refresh_token) if body.refresh_token else None,
        expires_at=None if body.expires_in is None else time.time() + body.expires_in,
        scopes=requested if body.scope is None else _split_scope(body.scope, client.scope_separator),
    )


def _split_scope(scope: str, separator: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in scope.split(separator) if part.strip())


class PKCEFlow(OAuthFlow[Tokens]):
    """One sign-in attempt: a fresh `state` and PKCE pair for `client`."""

    def __init__(self, client: PublicClient, *, service: str, transport: httpx.AsyncBaseTransport | None = None):
        """No I/O until `exchange_code_from_callback` or `exchange_code`."""
        super().__init__(redirect_uri=client.redirect_uri)
        self.client = client
        self.service = service
        self.transport = transport

    def authorization_url(self, *, scope: str | None = None, extra_params: Mapping[str, str] | None = None) -> str:
        """The URL to open; `scope=None` requests the client's `scopes`."""
        if scope is None:
            scope = self.client.scope_separator.join(self.client.scopes)
        params = {
            'response_type': 'code',
            'client_id': self.client.client_id,
            'redirect_uri': self.redirect_uri,
            'state': self.state,
            'code_challenge': self.code_challenge,
            'code_challenge_method': 'S256',
        }
        if scope:
            params['scope'] = scope
        base = urlsplit(self.client.authorize_url)
        query = [
            *parse_qsl(base.query, keep_blank_values=True),
            *self._merge_extra_params(params, extra_params).items(),
        ]
        return urlunsplit(base._replace(query=urlencode(query)))

    async def exchange_code(self, code: str) -> Tokens:
        """Trade the callback's code for tokens, proving possession with the PKCE verifier instead of a secret."""
        form = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': self.redirect_uri,
            'client_id': self.client.client_id,
            'code_verifier': self.code_verifier,
        }
        return await _request_tokens(
            self.client, form, requested=self.client.scopes, service=self.service, transport=self.transport
        )


async def refresh(
    client: PublicClient, tokens: Tokens, *, service: str, transport: httpx.AsyncBaseTransport | None = None
) -> Tokens:
    """Exchange the refresh token; keep the old one when the service does not rotate it."""
    assert tokens.refresh_token is not None
    form = {
        'grant_type': 'refresh_token',
        'refresh_token': tokens.refresh_token.get_secret_value(),
        'client_id': client.client_id,
    }
    renewed = await _request_tokens(client, form, requested=tokens.scopes, service=service, transport=transport)
    return renewed if renewed.refresh_token else renewed.model_copy(update={'refresh_token': tokens.refresh_token})


async def finish_write(write: Coroutine[object, object, object]) -> None:
    """Let a started credential write complete even when cancelled, then re-raise the cancellation.

    A thread cannot be interrupted and the write is atomic, so waiting is the only way for a cancelled caller to
    know nothing changes after it returns. The wait is shielded because an anyio cancel scope keeps cancelling
    every await inside it, not just the first.
    """
    task = asyncio.ensure_future(write)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        with CancelScope(shield=True):
            await task
        raise


async def _until_listening(redirect_uri: str, callback: asyncio.Future[Tokens]) -> None:
    """Return once the callback server answers, or when it has already failed (to bind, usually).

    Core's server answers a request without this flow's `state` and keeps waiting, so the probe is harmless.
    """
    async with httpx.AsyncClient(timeout=1) as http:
        while not callback.done():
            try:
                await http.get(redirect_uri)
                return
            except httpx.TransportError:
                await anyio.sleep(0.02)


async def _until_released(redirect_uri: str, *, limit: float = 2) -> None:
    """Wait, up to `limit` seconds, until a cancelled attempt's callback server stops listening.

    Core abandons that server's thread on cancel, and it only notices between its 0.5s accept polls, so without this
    signing in again at once finds the fixed redirect port still taken.
    """
    split = urlsplit(redirect_uri)
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        try:
            _, writer = await asyncio.open_connection(split.hostname, split.port)
        except OSError:
            return
        writer.close()
        await writer.wait_closed()
        await anyio.sleep(0.05)


class PKCESignIn:
    """A signed-in session for one client, kept in the credential store under `account`."""

    def __init__(
        self,
        *,
        client: PublicClient,
        account: str,
        service: str,
        setup: str,
        open_browser: Callable[[str], bool] = webbrowser.open,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 300,
    ) -> None:
        """`service` names it in messages and `setup` is the command that signs in again."""
        self.client = client
        self.account = account
        self.service = service
        self.setup = setup
        self.open_browser = open_browser
        self.transport = transport
        self.timeout = timeout

    def _load(self) -> Tokens | None:
        try:
            raw = load_codex_credentials(account=self.account)
            return None if raw is None else Tokens.model_validate_json(raw)
        except (UserError, ValidationError):
            return None  # Unreadable or incomplete tokens mean signing in again.

    def _save(self, tokens: Tokens) -> None:
        save_codex_credentials(account=self.account, value=tokens.stored())

    def _lock(self) -> AbstractContextManager[None]:
        return credential_lock(account=self.account, busy=f'Another CLAI is refreshing the {self.service} sign-in.')

    def signed_in(self) -> bool:
        """Whether this client has tokens that are live or refreshable. Reads the keyring; call it off the loop."""
        tokens = self._load()
        return (
            tokens is not None
            and tokens.grants(self.client)
            and (tokens.refresh_token is not None or not tokens.stale(time.time()))
        )

    def sign_out(self) -> None:
        """Forget the tokens. They stay valid at the service until they expire or are revoked there."""
        with self._lock():
            delete_credentials(account=self.account)

    def start(self) -> PKCEFlow:
        """A new attempt, so a caller can show `authorization_url()` before `sign_in(flow)` waits on it."""
        return PKCEFlow(self.client, service=self.service, transport=self.transport)

    async def sign_in(self, flow: PKCEFlow | None = None, *, show: Callable[[str], object] = print) -> Tokens:
        """Open the browser and wait up to `timeout` for the callback; save and return the tokens.

        `show` receives the URL to open by hand when no browser starts. Cancelling before the callback leaves any
        earlier sign-in; once the tokens are being saved, cancelling waits for the save, so it is never half done.
        """
        if flow is not None and flow.client != self.client:
            raise ValueError(f'That sign-in attempt is for another {self.service} app.')
        flow = flow or self.start()
        url = flow.authorization_url()
        # Listen before the browser opens: an app already approved redirects back at once, and a callback that
        # arrives before the one-shot server binds is lost.
        callback = asyncio.ensure_future(flow.exchange_code_from_callback())
        try:
            with fail_after(self.timeout):
                await _until_listening(flow.redirect_uri, callback)
                try:
                    opened = await anyio.to_thread.run_sync(self.open_browser, url, abandon_on_cancel=True)
                except webbrowser.Error:
                    opened = False
                if not opened:
                    show(f'Open this URL to sign in to {self.service}: {url}')
                tokens = await callback
        except TimeoutError:
            raise UserError(f'{self.service} sign-in timed out. Run {self.setup} to try again.') from None
        except OSError:
            raise UserError(
                f'Could not listen on {self.client.redirect_uri} for the {self.service} sign-in. '
                'Close whatever uses that port, or another sign-in, and try again.'
            ) from None
        finally:
            callback.cancel()
            await asyncio.gather(callback, return_exceptions=True)
            if callback.cancelled():
                with CancelScope(shield=True):
                    await _until_released(flow.redirect_uri)
        await finish_write(asyncio.to_thread(self._locked_save, tokens))
        return tokens

    def _locked_save(self, tokens: Tokens) -> None:
        with self._lock():
            self._save(tokens)

    async def token(self) -> str:
        """A live access token, refreshed and saved when close to expiry. Raises `UserError` naming `setup`.

        Refresh runs under a cross-process lock and re-reads first, because a rotating service invalidates the
        old refresh token: two sessions refreshing at once would sign each other out.
        """
        tokens = self._usable(await anyio.to_thread.run_sync(self._load, abandon_on_cancel=True))
        if not tokens.stale(time.time()):
            return tokens.access_token.get_secret_value()
        if tokens.refresh_token is None:
            raise UserError(f'The {self.service} sign-in expired. Run {self.setup} to sign in again.')
        return await anyio.to_thread.run_sync(self._refresh_locked, abandon_on_cancel=True)

    def _usable(self, tokens: Tokens | None) -> Tokens:
        """`tokens`, when they are this client's and grant everything it asks for now; otherwise raise."""
        if tokens is None or tokens.client_id != self.client.client_id:
            raise UserError(f'Not signed in to {self.service}. Run {self.setup} to sign in.')
        if not tokens.grants(self.client):
            raise UserError(
                f'The {self.service} sign-in does not grant what this setup needs. Run {self.setup} to sign in again.'
            )
        return tokens

    def _refresh_locked(self) -> str:
        with self._lock():
            tokens = self._usable(self._load())
            if tokens.stale(time.time()):
                if tokens.refresh_token is None:
                    raise UserError(f'Not signed in to {self.service}. Run {self.setup} to sign in.')
                try:
                    tokens = asyncio.run(refresh(self.client, tokens, service=self.service, transport=self.transport))
                except UserError:
                    raise UserError(
                        f'The {self.service} sign-in could not be renewed. Run {self.setup} to sign in again.'
                    ) from None
                self._save(tokens)  # Saved even when narrowed: the old refresh token may be spent.
                tokens = self._usable(tokens)
            return tokens.access_token.get_secret_value()
