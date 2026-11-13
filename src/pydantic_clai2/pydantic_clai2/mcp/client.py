"""The FastMCP side of `/mcp`, imported on the first connection or sign-in rather than at startup.

FastMCP loads the MCP SDK, which takes most of a second, and the `mcp` plugin loads at every startup.
"""

import logging
import os
from collections.abc import Callable

import fastmcp
import httpx
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import SSETransport, StdioTransport, StreamableHttpTransport
from mcp.shared._httpx_utils import create_mcp_http_client

from pydantic_clai2.mcp.tokens import TokenStore
from pydantic_clai2.plugins.sign_in import SignInRequiredError

__all__ = ['Client', 'SSETransport', 'SignIn', 'StdioTransport', 'StreamableHttpTransport', 'http_client']

# FastMCP logs the OAuth URL and callback server at INFO through its own handler, over the prompt.
# The browser opening is the signal; an explicit `FASTMCP_LOG_LEVEL` still wins for debugging. (FastMCP sets
# its logger's level as it's imported, which is why it's imported above: imported later, it would undo this.)
if 'FASTMCP_LOG_LEVEL' not in os.environ:  # pragma: no branch
    logging.getLogger(fastmcp.__name__).setLevel(logging.WARNING)


def http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
    follow_redirects: bool = False,
) -> httpx.AsyncClient:
    """Do not let a configured endpoint redirect MCP requests to another server.

    The MCP SDK builds the client, so it is the HTTPX the installed FastMCP drives: legacy `httpx` under
    FastMCP 3, `httpx2` under FastMCP 4, where a browser sign-in (`SignIn`) is an `httpx2.Auth` that a legacy
    client rejects. The annotations name legacy `httpx`, the generation this workspace locks.
    FastMCP passes `follow_redirects=True` to every client factory; it is accepted and overridden.
    """
    del follow_redirects
    client = create_mcp_http_client(headers=headers, timeout=timeout, auth=auth)
    client.follow_redirects = False
    return client


class SignIn(OAuth):
    """A browser sign-in whose tokens persist under `name`; FastMCP refreshes them or opens the browser on connect."""

    def __init__(self, name: str, *, callback_host: str = '127.0.0.1') -> None:
        """Tokens go to the `mcp-NAME` credential; see `TokenStore`."""
        self.tokens = TokenStore(name)
        super().__init__(client_name='CLAI', callback_host=callback_host, token_storage=self.tokens)


class StoredOnly(SignIn):
    """For runs: stored tokens, refreshed when they expire, and a `SignInRequiredError` where FastMCP would open a browser."""

    def __init__(self, name: str, *, callback_host: str, refusal: str) -> None:
        super().__init__(name, callback_host=callback_host)
        self._refusal = refusal

    async def redirect_handler(self, authorization_url: str) -> None:
        raise SignInRequiredError(self._refusal)


class Shown(SignIn):
    """For `OAuthSignIn.sign_in`: show the link too, for a browser that does not open (over SSH, say)."""

    def __init__(self, name: str, *, callback_host: str, show: Callable[[str], object]) -> None:
        super().__init__(name, callback_host=callback_host)
        self._show = show

    async def redirect_handler(self, authorization_url: str) -> None:
        self._show(f'If no browser opened, open this link:\n{authorization_url}')
        await super().redirect_handler(authorization_url)
