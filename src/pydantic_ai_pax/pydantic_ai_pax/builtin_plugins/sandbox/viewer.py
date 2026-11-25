"""On-demand loopback live desktop viewer for `/computer-use` sandboxes.

Serves a minimal self-contained page on `127.0.0.1` with a per-launch token:
- Updates immediately via Server-Sent Events whenever the model completes a `computer` call, showing the
  screenshot and subtle markers for clicks, moves, and drags.
- Supports on-demand live screenshots (`POST /api/capture`) via `Display.screenshot()` and optional client-side
  auto-refresh while the page is open.
- Sets `frame-ancestors 'self' http://127.0.0.1:* http://localhost:*` so Pax's web UI embeds it directly as a
  side panel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import tempfile
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, quote

import anyio
from anyio import to_thread

from pydantic_ai.computer_use import ComputerResult, ComputerToolAction
from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import WorkspaceError, WorkspaceRef
from pydantic_ai_pax.builtin_plugins.sandbox.computer_use import ActionMarker, action_markers, format_actions

if TYPE_CHECKING:
    from starlette.datastructures import Headers

    from pydantic_ai_pax.builtin_plugins.web.server import WebServer

Scope = MutableMapping[str, object]
Message = MutableMapping[str, object]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

HEADER = 'x-pax-desktop-token'
MAX_FRAMES = 30
"""Maximum screenshots kept in memory per desktop sandbox."""

_PAGE = Path(__file__).with_name('viewer.html')
"""The viewer page, with `__TITLE__`, `__NONCE__`, and `__CONFIG__` for `_viewer_html` to fill in."""
_SUBSCRIBER_QUEUE_SIZE = 16
_logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class DesktopFrame:
    """One recorded screenshot of a desktop sandbox and the actions that produced it."""

    seq: int
    timestamp: datetime
    png: bytes = field(repr=False)
    summary: str
    markers: tuple[ActionMarker, ...] = ()
    error: str | None = None

    def meta(self) -> dict[str, object]:
        """JSON-serializable metadata sent over SSE and the initial state endpoint."""
        return {
            'seq': self.seq,
            'timestamp': self.timestamp.isoformat(),
            'summary': self.summary,
            'error': self.error,
            'markers': [
                {'kind': marker.kind, 'points': [list(point) for point in marker.points], 'label': marker.label}
                for marker in self.markers
            ],
        }


def sandbox_key(ref: WorkspaceRef) -> str:
    """Stable URL key for `ref`."""
    return f'{ref.provider}:{ref.id}'


class DesktopSession:
    """Bounded frame history and SSE subscribers for one desktop sandbox."""

    def __init__(
        self,
        *,
        ref: WorkspaceRef,
        title: str,
        capture: Callable[[], Awaitable[bytes]],
    ) -> None:
        self.ref = ref
        self.key = sandbox_key(ref)
        self.title = title
        self._capture = capture
        self._frames: deque[DesktopFrame] = deque(maxlen=MAX_FRAMES)
        self._seq = 0
        self._subscribers: set[asyncio.Queue[DesktopFrame | None]] = set()

    @property
    def frames(self) -> list[DesktopFrame]:
        return list(self._frames)

    def frame(self, seq: int) -> DesktopFrame | None:
        return next((item for item in self._frames if item.seq == seq), None)

    def record(
        self,
        png: bytes,
        *,
        actions: Sequence[ComputerToolAction] = (),
        result: ComputerResult | None = None,
        summary: str | None = None,
    ) -> DesktopFrame:
        """Append a frame and push it to all live SSE subscribers."""
        self._seq += 1
        text = summary if summary is not None else (format_actions(actions) or 'Screenshot')
        error = result.get('error') if result is not None else None
        item = DesktopFrame(
            seq=self._seq,
            timestamp=datetime.now(UTC),
            png=png,
            summary=text,
            markers=action_markers(actions),
            error=error,
        )
        self._frames.append(item)
        for queue in list(self._subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(item)
        return item

    async def capture_live(self) -> DesktopFrame:
        """Take a fresh screenshot directly from the sandbox display and publish it."""
        png = await self._capture()
        return self.record(png, summary='Live snapshot')

    def subscribe(self) -> asyncio.Queue[DesktopFrame | None]:
        queue: asyncio.Queue[DesktopFrame | None] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[DesktopFrame | None]) -> None:
        self._subscribers.discard(queue)

    def close(self) -> None:
        """Signal all SSE subscribers to close."""
        for queue in list(self._subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(None)


class _ViewerGuard:
    """Require the per-launch token and loopback `Host` header on every viewer request."""

    def __init__(self, app: ASGIApp, *, token: str, hosts: frozenset[str]) -> None:
        self.app = app
        self.token = token
        self.hosts = hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return
        from starlette.datastructures import Headers
        from starlette.responses import Response

        headers = Headers(scope=scope)
        path = str(scope['path'])
        raw_query = scope.get('query_string', b'')
        query = parse_qs(raw_query.decode() if isinstance(raw_query, bytes) else '')
        if self._allowed(path, headers, query):
            await self.app(scope, receive, send)
            return
        refused = Response(
            'Open the desktop link Pax printed: it carries this session access token.',
            status_code=403,
            media_type='text/plain',
        )
        await refused(scope, receive, send)

    def _allowed(self, path: str, headers: Headers, query: dict[str, list[str]]) -> bool:
        if headers.get('host') not in self.hosts:
            return False
        if path.startswith('/api/'):
            origin = headers.get('origin')
            if origin is not None and origin not in {f'http://{host}' for host in self.hosts}:
                return False
            return self._valid(headers.get(HEADER, ''))
        return self._valid(query.get('token', [''])[0])

    def _valid(self, token: str) -> bool:
        return secrets.compare_digest(token.encode(), self.token.encode())


def _viewer_html(page: str, *, session: DesktopSession, token: str, nonce: str) -> str:
    """`viewer.html` (`page`) for `session`, with its title, script nonce, and starting state filled in."""
    config = json.dumps(
        {
            'key': session.key,
            'token': token,
            'header': HEADER,
            'maxFrames': MAX_FRAMES,
            'frames': [frame.meta() for frame in session.frames],
        }
    ).replace('</', '<\\/')
    title = escape(f'{session.title} desktop · {session.ref.id}')
    return page.replace('__TITLE__', title).replace('__NONCE__', nonce).replace('__CONFIG__', config)


async def _event_stream(session: DesktopSession) -> AsyncIterator[str]:
    """Yield SSE `data:` payloads for existing and new frames on `session` until it closes."""
    queue = session.subscribe()
    try:
        for item in session.frames:
            yield f'data: {json.dumps(item.meta())}\n\n'
        while (item := await queue.get()) is not None:
            yield f'data: {json.dumps(item.meta())}\n\n'
    finally:
        session.unsubscribe(queue)


def create_viewer_app(viewer: DesktopViewer, *, token: str, hosts: frozenset[str]) -> ASGIApp:
    """Build the FastAPI app for `viewer`, protected by `_ViewerGuard`."""
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def _get_session(sandbox: str) -> DesktopSession:
        session = viewer.get(sandbox)
        if session is None:
            raise HTTPException(status_code=404, detail='Desktop sandbox not found.')
        return session

    @app.get('/')
    async def index(sandbox: str = Query(...)) -> HTMLResponse:
        session = _get_session(sandbox)
        nonce = secrets.token_urlsafe(16)
        page = await to_thread.run_sync(_PAGE.read_text, 'utf-8')
        html = _viewer_html(page, session=session, token=token, nonce=nonce)
        return HTMLResponse(
            html,
            headers={
                'Cache-Control': 'no-store',
                'Referrer-Policy': 'no-referrer',
                'X-Content-Type-Options': 'nosniff',
                'Content-Security-Policy': (
                    f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'; "
                    "img-src 'self' data:; connect-src 'self'; "
                    "frame-ancestors 'self' http://127.0.0.1:* http://localhost:*; base-uri 'none'; form-action 'none'"
                ),
            },
        )

    @app.get('/frames/{seq}.png')
    async def frame_image(seq: int, sandbox: str = Query(...)) -> Response:
        session = _get_session(sandbox)
        item = session.frame(seq)
        if item is None:
            raise HTTPException(status_code=404, detail='Frame not found.')
        return Response(
            item.png,
            media_type='image/png',
            headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'},
        )

    @app.get('/events')
    async def events(sandbox: str = Query(...)) -> StreamingResponse:
        session = _get_session(sandbox)
        return StreamingResponse(
            _event_stream(session),
            media_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no', 'X-Content-Type-Options': 'nosniff'},
        )

    @app.post('/api/capture')
    async def capture(sandbox: str = Query(...)) -> JSONResponse:
        session = _get_session(sandbox)
        try:
            item = await session.capture_live()
        except (UserError, WorkspaceError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return JSONResponse(item.meta())

    return _ViewerGuard(app, token=token, hosts=hosts)


def _write_launch_page(url: str) -> Path:
    """Write a `0600` redirect page to `url`, keeping any access token out of browser command-line arguments."""
    descriptor, name = tempfile.mkstemp(prefix='pax-desktop-', suffix='.html')
    with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
        escaped = escape(url)
        file.write(
            f'<!doctype html><meta http-equiv="refresh" content="0;url={escaped}"><a href="{escaped}">Open desktop</a>'
        )
    return Path(name)


def _unlink_pages(pages: list[Path]) -> None:
    for page in pages:
        page.unlink(missing_ok=True)


class DesktopViewer:
    """Manages one on-demand loopback viewer server for a Pax session's desktop sandboxes."""

    def __init__(self) -> None:
        self._sessions: dict[str, DesktopSession] = {}
        self._token = secrets.token_urlsafe(24)
        self._server: WebServer | None = None
        self._launch_pages: list[Path] = []

    def register(self, session: DesktopSession) -> DesktopSession:
        self._sessions[session.key] = session
        return session

    def get(self, key: str) -> DesktopSession | None:
        return self._sessions.get(key)

    def remove(self, ref: WorkspaceRef) -> None:
        if (session := self._sessions.pop(sandbox_key(ref), None)) is not None:
            session.close()

    def ensure_started(self) -> int:
        """Start the loopback HTTP server on an ephemeral port if not already running, and return its port."""
        if self._server is None:
            from pydantic_ai_pax.builtin_plugins.web.server import WebServer

            def build(port: int) -> ASGIApp:
                hosts = frozenset({f'127.0.0.1:{port}', f'localhost:{port}'})
                return create_viewer_app(self, token=self._token, hosts=hosts)

            self._server = WebServer.serve(build, host='127.0.0.1', port=0)
        return self._server.port

    def url(self, ref: WorkspaceRef) -> str:
        """The viewer URL for `ref`, starting the loopback server on demand."""
        port = self.ensure_started()
        return (
            f'http://127.0.0.1:{port}/?token={quote(self._token, safe="")}&sandbox={quote(sandbox_key(ref), safe="")}'
        )

    def started_url(self, ref: WorkspaceRef) -> str | None:
        """The viewer URL for `ref` once the loopback server runs and has its session, never starting the server."""
        return self.url(ref) if self._server is not None and sandbox_key(ref) in self._sessions else None

    def try_url(self, ref: WorkspaceRef) -> str | None:
        """Like `url(ref)`, returning `None` if the loopback server cannot bind a port."""
        try:
            return self.url(ref)
        except OSError:
            _logger.warning('Could not start loopback desktop viewer for %s', ref, exc_info=True)
            return None

    async def launch_url(self, url: str) -> str:
        """Return a private `file://` redirect URL for `url`, tracked for cleanup in `close`."""
        page = await to_thread.run_sync(_write_launch_page, url)
        self._launch_pages.append(page)
        return page.as_uri()

    async def close(self) -> None:
        """Close all SSE streams, remove private redirect pages, and stop the loopback server if started."""
        for session in list(self._sessions.values()):
            session.close()
        self._sessions.clear()
        if self._launch_pages:
            pages, self._launch_pages = self._launch_pages, []
            with anyio.CancelScope(shield=True):
                await to_thread.run_sync(_unlink_pages, pages)
        if (server := self._server) is not None:
            self._server = None
            with anyio.CancelScope(shield=True):
                await server.stop()
