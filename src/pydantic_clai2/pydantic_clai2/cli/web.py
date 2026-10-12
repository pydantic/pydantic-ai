"""`clai2 --web`: CLAI's stock agent in Pydantic AI's browser chat UI, served on this machine only."""

import secrets
import socket
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from rich.console import Console
from starlette.requests import HTTPConnection
from starlette.responses import PlainTextResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

from pydantic_clai2._app import open_stock_agent
from pydantic_clai2.config import Settings

HOST = '127.0.0.1'
"""IPv4 loopback only: the coding tools run with the user's permissions, so the UI never listens beyond this machine."""

DEFAULT_PORT = 7932
"""The port `clai web` uses too."""

_TOKEN = 'token'


@dataclass(frozen=True, kw_only=True)
class TokenGate:
    """Answer only a browser that opened the link `serve_web` printed, as Jupyter does.

    Core's `to_web()` app already refuses other websites (its `Host` check and JSON-only chat
    endpoint), but anything else on this machine, such as another user's process, can still reach
    the loopback port and would run tools as this user. The link carries a random `token`; opening
    it sets the token as an `HttpOnly`, `SameSite=Strict` cookie and redirects to the same page
    without it. Every other request must carry the cookie.
    """

    app: ASGIApp
    token: str
    cookie: str
    """Named per port: browsers share cookies across a host's ports, so two servers must not overwrite each other's."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] == 'lifespan':
            await self.app(scope, receive, send)
            return
        connection = HTTPConnection(scope)
        if self._matches(connection.cookies.get(self.cookie)):
            await self.app(scope, receive, send)
        elif scope['type'] == 'websocket':
            await WebSocketClose(code=1008)(scope, receive, send)
        elif scope['method'] == 'GET' and self._matches(connection.query_params.get(_TOKEN)):
            # A relative location, so the redirect never names a host the request supplied.
            path, query = connection.url.path, connection.url.remove_query_params(_TOKEN).query
            response = RedirectResponse(f'{path}?{query}' if query else path, status_code=303)
            response.set_cookie(self.cookie, self.token, httponly=True, samesite='strict')
            await response(scope, receive, send)
        else:
            refusal = 'Open the link `clai2 --web` printed in its terminal; that signs this browser in.'
            await PlainTextResponse(refusal, status_code=403)(scope, receive, send)

    def _matches(self, candidate: str | None) -> bool:
        # As bytes: `compare_digest` raises on a `str` with non-ASCII characters, which a request can send.
        return candidate is not None and secrets.compare_digest(candidate.encode(), self.token.encode())


async def serve_web(*, settings: Settings, port: int, workspace: Path, console: Console | None = None) -> None:
    """Serve the stock agent at `http://127.0.0.1:PORT` until Ctrl-C.

    The agent is `open_stock_agent`'s, working in `workspace` with `settings.model`; port `0` picks a free one.
    Raises `OSError` when the port cannot be bound, before the agent opens.
    """
    if settings.model is None:
        raise ValueError('Choose a model with -m PROVIDER:NAME')
    console = console or Console()
    try:
        # Bound here rather than by uvicorn, which exits the process when it cannot bind.
        listener = socket.create_server((HOST, port))
    except OSError as exc:
        raise OSError(f'Cannot serve the browser UI: {exc.strerror}') from exc
    with listener:
        port = listener.getsockname()[1]
        token = secrets.token_urlsafe(32)
        async with open_stock_agent(workspace=workspace, model=settings.model) as agent:
            app = TokenGate(app=agent.to_web(), token=token, cookie=f'clai2-web-{port}')
            for line in (
                f'Serving {workspace} with {settings.model}. Press Ctrl-C to stop.',
                f'Open http://{HOST}:{port}/?{_TOKEN}={token}',
            ):
                console.print(line, markup=False, highlight=False, soft_wrap=True)
            server = uvicorn.Server(uvicorn.Config(app, log_level='warning'))
            await server.serve(sockets=[listener])
