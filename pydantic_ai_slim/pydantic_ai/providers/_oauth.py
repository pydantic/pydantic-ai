"""Shared authorization-code + PKCE primitives for provider login flows.

Provider OAuth flows (e.g. OpenAI Codex) subclass [`OAuthFlow`][pydantic_ai.providers._oauth.OAuthFlow]
so they share one shape: construction does no I/O, `authorization_url()` builds the redirect, and
`exchange_code()` turns the callback's authorization code into provider credentials.
"""

from __future__ import annotations as _annotations

import base64
import hashlib
import secrets
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, ClassVar, Generic, TypeVar
from urllib.parse import parse_qs, urlparse

import anyio.to_thread

from pydantic_ai.exceptions import UserError

CredentialsT = TypeVar('CredentialsT')

_CALLBACK_READ_TIMEOUT = 5.0
"""Seconds a callback connection may stall before it is dropped, so it cannot block the real redirect."""


class OAuthFlow(ABC, Generic[CredentialsT]):
    """Authorization-code + PKCE context for a provider's public OAuth client.

    Subclasses pin their provider's endpoints, client id, and credential type; the base carries
    the pieces every flow needs: the `state` guarding the callback, the PKCE verifier/challenge
    pair, and the redirect URI the authorization code is bound to.
    """

    _bound_params: ClassVar[frozenset[str]] = frozenset({'client_id', 'redirect_uri', 'state', 'code_challenge'})
    """Authorization parameters the flow validates or exchanges against, so a caller override breaks the login."""

    def __init__(self, *, redirect_uri: str, state: str | None = None) -> None:
        self.redirect_uri = redirect_uri
        self.state = state or secrets.token_urlsafe(16)
        self.code_verifier = secrets.token_urlsafe(32)

    @property
    def code_challenge(self) -> str:
        """The S256 PKCE challenge derived from `code_verifier`."""
        digest = hashlib.sha256(self.code_verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b'=').decode()

    def _merge_extra_params(self, params: dict[str, str], extra_params: Mapping[str, str] | None) -> dict[str, str]:
        """Merge caller-supplied query parameters over the defaults, refusing identity overrides."""
        if extra_params:
            if overridden := sorted(self._bound_params & extra_params.keys()):
                raise UserError(
                    f'`extra_params` cannot override {", ".join(overridden)}: the flow validates the callback and '
                    'exchanges the code with its own values, so the authorization code would be unusable. Pass '
                    '`redirect_uri=` to the constructor instead.'
                )
            params.update(extra_params)
        return params

    @abstractmethod
    def authorization_url(self, *, scope: str | None = None, extra_params: Mapping[str, str] | None = None) -> str:
        """The URL to send the user to; `scope=None` means the provider's default scopes."""
        ...

    @abstractmethod
    async def exchange_code(self, code: str) -> CredentialsT:
        """Exchange an authorization code for provider credentials (call this in your redirect handler)."""
        ...

    async def exchange_code_from_callback(self) -> CredentialsT:
        """Serve `redirect_uri` for one authorization callback, then exchange the received code.

        Binds the host and port from `redirect_uri` with a one-shot local HTTP server, ignores
        requests that don't carry this flow's `state`, and raises
        [`UserError`][pydantic_ai.exceptions.UserError] when the provider reports an authorization
        error instead of a code (e.g. the user clicked Deny). Callers wanting a time limit can wrap
        the call in [`anyio.fail_after`](https://anyio.readthedocs.io/en/stable/cancellation.html).
        """
        parsed = urlparse(self.redirect_uri)
        address = (parsed.hostname or 'localhost', parsed.port or 80)
        callback_path = parsed.path
        expected_state = self.state
        callback: dict[str, str] = {}

        class CallbackHandler(BaseHTTPRequestHandler):
            def setup(self) -> None:
                # `server.timeout` only bounds the accept; a connection that never sends its request
                # line would otherwise block the server (and the real callback) indefinitely.
                self.request.settimeout(_CALLBACK_READ_TIMEOUT)
                super().setup()

            def do_GET(self) -> None:
                try:
                    url = urlparse(self.path)
                except ValueError:
                    # A stray request with a malformed request line (port scanner, browser
                    # prefetch) must not crash the server thread mid-login.
                    self.send_error(400, 'Malformed request')
                    return
                params = {name: values[0] for name, values in parse_qs(url.query).items()}
                if url.path == callback_path and params.get('state') == expected_state:
                    if params.get('code'):
                        callback.update(params)
                    else:
                        callback['error'] = params.get('error', 'unknown')
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'You can close this tab.')

            def log_message(self, format: str, *args: Any) -> None:
                pass  # keep the caller's console clean

        cancelled = threading.Event()

        def serve_one_callback() -> None:
            with HTTPServer(address, CallbackHandler) as server:
                server.timeout = 0.5  # recheck for a result or cancellation between requests
                while not callback and not cancelled.is_set():
                    server.handle_request()

        try:
            await anyio.to_thread.run_sync(serve_one_callback, abandon_on_cancel=True)
        finally:
            cancelled.set()
        if 'code' not in callback:
            raise UserError(f'Authorization failed: {callback["error"]}')
        return await self._exchange_callback(callback)

    async def _exchange_callback(self, params: Mapping[str, str]) -> CredentialsT:
        """Exchange a callback that carried this flow's `state` and a `code`.

        Flows whose callback carries more than the code (such as a client id issued during
        registration) override this to read it.
        """
        return await self.exchange_code(params['code'])
