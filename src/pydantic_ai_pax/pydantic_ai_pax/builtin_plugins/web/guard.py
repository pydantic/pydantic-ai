"""The access check in front of Pax's loopback web servers: the web UI and the desktop viewer.

Each server binds to loopback with a per-launch token. `Guard` refuses a request without that token, one naming
another host (which DNS rebinding would send), and one from another site's page.
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable, MutableMapping
from urllib.parse import parse_qs

from starlette.datastructures import Headers
from starlette.responses import Response

Scope = MutableMapping[str, object]
Message = MutableMapping[str, object]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

HEADER = 'x-clai-token'
"""The header the web UI's `/api/` requests carry the token in."""


class Guard:
    """Refuse any request without the token, for another host name, or, for the WebSocket and `/api/`, from another origin.

    `header` names the header an `/api/` request carries the token in.
    """

    def __init__(self, app: ASGIApp, *, token: str, hosts: frozenset[str], header: str = HEADER) -> None:
        self.app = app
        self.token = token
        self.hosts = hosts
        self.header = header
        self.origins = frozenset(f'http://{host}' for host in hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope['type']
        if kind not in ('http', 'websocket'):
            await self.app(scope, receive, send)
            return
        if self._allowed(kind, str(scope['path']), Headers(scope=scope), _query(scope)):
            await self.app(scope, receive, send)
        elif kind == 'websocket':
            await send({'type': 'websocket.close', 'code': 1008})
        else:
            refused = Response(
                "Open the address Pax printed: it carries this launch's access token.",
                status_code=403,
                media_type='text/plain',
            )
            await refused(scope, receive, send)

    def _allowed(self, kind: object, path: str, headers: Headers, query: dict[str, list[str]]) -> bool:
        if headers.get('host') not in self.hosts:
            return False
        if kind == 'websocket':
            return headers.get('origin') in self.origins and self._valid(query.get('token', [''])[0])
        if path.startswith('/api/'):
            # A browser sends `Origin` with a cross-site request, so one from another page never reaches the API.
            origin = headers.get('origin')
            return (origin is None or origin in self.origins) and self._valid(headers.get(self.header, ''))
        if path.startswith('/assets/'):
            return True
        return self._valid(query.get('token', [''])[0])

    def _valid(self, token: str) -> bool:
        return secrets.compare_digest(token.encode(), self.token.encode())


def _query(scope: Scope) -> dict[str, list[str]]:
    raw = scope.get('query_string', b'')
    return parse_qs(raw.decode() if isinstance(raw, bytes) else '')
