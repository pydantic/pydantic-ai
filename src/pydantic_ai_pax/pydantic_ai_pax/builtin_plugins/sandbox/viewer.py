"""On-demand loopback live desktop viewer for `/computer-use` sandboxes.

Serves a minimal self-contained page on `127.0.0.1` with a per-launch token:
- Updates immediately via Server-Sent Events whenever the model completes a `computer` call, showing the
  screenshot and subtle markers for clicks, moves, and drags.
- Supports on-demand live screenshots (`POST /api/capture`) via `Display.screenshot()` and optional client-side
  auto-refresh while the page is open.
- Exposes an on-demand "Open live desktop" button (`POST /api/live-url`) when the sandbox provider offers a
  native interactive desktop URL (such as boxd's console Desktop view or Daytona's noVNC preview).
- Sets `frame-ancestors 'self' http://127.0.0.1:* http://localhost:*` so Pax's web UI embeds it directly as a
  side panel.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, quote

import anyio

from pydantic_ai.computer_use import ComputerResult, ComputerToolAction
from pydantic_ai.workspaces import WorkspaceRef
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
_SUBSCRIBER_QUEUE_SIZE = 16


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
        live_url: Callable[[], Awaitable[str | None]],
        has_live_url: bool = False,
    ) -> None:
        self.ref = ref
        self.key = sandbox_key(ref)
        self.title = title
        self._capture = capture
        self._live_url = live_url
        self.has_live_url = has_live_url
        self._frames: deque[DesktopFrame] = deque(maxlen=MAX_FRAMES)
        self._seq = 0
        self._subscribers: set[asyncio.Queue[DesktopFrame | None]] = set()
        self._seeded = False

    @property
    def seeded(self) -> bool:
        return self._seeded

    def mark_seeded(self) -> None:
        self._seeded = True

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

    async def live_url(self) -> str | None:
        """Fetch a provider-native live desktop URL on demand."""
        return await self._live_url()

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


def _viewer_html(*, session: DesktopSession, token: str, nonce: str) -> str:
    """Self-contained live desktop viewer page for `session`."""
    title = escape(f'{session.title} desktop · {session.ref.id}')
    config = json.dumps(
        {
            'key': session.key,
            'token': token,
            'hasLiveUrl': session.has_live_url,
            'frames': [frame.meta() for frame in session.frames],
        }
    ).replace('</', '<\\/')
    live_button = (
        '<button id="live-btn" class="btn primary" type="button" title="Open the provider interactive desktop">'
        'Interactive desktop ↗</button>'
        if session.has_live_url
        else ''
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style nonce="{nonce}">
:root {{
  --bg: #0f1117; --panel: #171a23; --border: #262b38; --text: #e6e9f0;
  --muted: #8b93a7; --accent: #e92063; --error: #f87171; --ok: #34d399;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; height: 100%; background: var(--bg); color: var(--text); font: 13px/1.4 system-ui, -apple-system, sans-serif; }}
body {{ display: flex; flex-direction: column; overflow: hidden; }}
header {{
  display: flex; align-items: center; gap: 8px; padding: 8px 12px;
  background: var(--panel); border-bottom: 1px solid var(--border); flex-wrap: wrap;
}}
.dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--ok); flex: none; }}
.title {{ font-weight: 600; margin-right: auto; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.btn {{
  background: #232837; color: var(--text); border: 1px solid var(--border); border-radius: 6px;
  padding: 4px 10px; font: inherit; cursor: pointer;
}}
.btn:hover {{ background: #2c3347; }}
.btn.active {{ border-color: var(--accent); color: #fff; }}
.btn.primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
.btn:disabled {{ opacity: 0.5; cursor: wait; }}
.stage {{
  flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center;
  padding: 12px; position: relative; overflow: hidden;
}}
.canvas {{ position: relative; max-width: 100%; max-height: 100%; display: none; line-height: 0; }}
.canvas.visible {{ display: inline-block; }}
.canvas img {{ max-width: 100%; max-height: calc(100vh - 148px); display: block; border-radius: 6px; border: 1px solid var(--border); }}
.overlay {{ position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }}
.empty {{ color: var(--muted); text-align: center; max-width: 360px; }}
.bar {{
  display: flex; align-items: center; gap: 10px; padding: 6px 12px;
  background: var(--panel); border-top: 1px solid var(--border); font-family: ui-monospace, SFMono-Regular, monospace;
  font-size: 12px; color: var(--muted);
}}
.bar .summary {{ color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; }}
.bar .error {{ color: var(--error); }}
.strip {{
  display: flex; gap: 6px; padding: 6px 12px; background: var(--bg);
  border-top: 1px solid var(--border); overflow-x: auto; min-height: 54px; align-items: center;
}}
.thumb {{
  flex: none; padding: 2px; border: 2px solid transparent; border-radius: 4px; background: var(--panel); cursor: pointer;
}}
.thumb.selected {{ border-color: var(--accent); }}
.thumb img {{ height: 38px; display: block; border-radius: 2px; }}
</style>
</head>
<body>
<header>
  <span class="dot" id="status-dot" title="Connected"></span>
  <span class="title">{title}</span>
  <button id="refresh-btn" class="btn" type="button">Refresh</button>
  <button id="auto-btn" class="btn" type="button" aria-pressed="false">Auto: off</button>
  <button id="follow-btn" class="btn active" type="button" aria-pressed="true">Live</button>
  {live_button}
</header>
<main class="stage">
  <div id="empty" class="empty">Waiting for the first screenshot. Click <strong>Refresh</strong> to capture the current screen now.</div>
  <div id="canvas" class="canvas">
    <img id="screen" alt="Desktop screen">
    <svg id="overlay" class="overlay"></svg>
  </div>
</main>
<div class="bar">
  <span id="step-label">No frames</span>
  <span id="summary" class="summary"></span>
  <span id="error" class="error"></span>
</div>
<div id="strip" class="strip"></div>
<script nonce="{nonce}">
const CFG = {config};
const frames = [...CFG.frames];
let selectedSeq = frames.length ? frames[frames.length - 1].seq : null;
let followLive = true;
let autoTimer = null;

const emptyEl = document.getElementById('empty');
const canvasEl = document.getElementById('canvas');
const imgEl = document.getElementById('screen');
const svgEl = document.getElementById('overlay');
const stepEl = document.getElementById('step-label');
const summaryEl = document.getElementById('summary');
const errorEl = document.getElementById('error');
const stripEl = document.getElementById('strip');
const refreshBtn = document.getElementById('refresh-btn');
const autoBtn = document.getElementById('auto-btn');
const followBtn = document.getElementById('follow-btn');
const liveBtn = document.getElementById('live-btn');

const qs = (extra = '') => '?token=' + encodeURIComponent(CFG.token) + '&sandbox=' + encodeURIComponent(CFG.key) + extra;
const frameUrl = (seq) => '/frames/' + seq + '.png' + qs();

function drawMarkers(frame) {{
  svgEl.replaceChildren();
  const w = imgEl.naturalWidth || 0;
  const h = imgEl.naturalHeight || 0;
  if (!w || !h || !frame || !frame.markers || !frame.markers.length) return;
  svgEl.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
  const NS = 'http://www.w3.org/2000/svg';
  for (const m of frame.markers) {{
    if (m.kind === 'drag' && m.points.length >= 2) {{
      const poly = document.createElementNS(NS, 'polyline');
      poly.setAttribute('points', m.points.map((p) => p[0] + ',' + p[1]).join(' '));
      poly.setAttribute('fill', 'none');
      poly.setAttribute('stroke', '#e92063');
      poly.setAttribute('stroke-width', '3');
      poly.setAttribute('stroke-dasharray', '6 4');
      svgEl.appendChild(poly);
    }}
    for (const [x, y] of m.points) {{
      const c = document.createElementNS(NS, 'circle');
      c.setAttribute('cx', String(x));
      c.setAttribute('cy', String(y));
      c.setAttribute('r', m.kind === 'move' ? '7' : '11');
      c.setAttribute('fill', m.kind === 'move' ? 'rgba(52,211,153,0.25)' : 'rgba(233,32,99,0.25)');
      c.setAttribute('stroke', m.kind === 'move' ? '#34d399' : '#e92063');
      c.setAttribute('stroke-width', '2.5');
      svgEl.appendChild(c);
    }}
  }}
}}

function render() {{
  followBtn.classList.toggle('active', followLive);
  followBtn.setAttribute('aria-pressed', String(followLive));
  const current = frames.find((f) => f.seq === selectedSeq) || frames[frames.length - 1] || null;
  if (!current) {{
    emptyEl.style.display = 'block';
    canvasEl.classList.remove('visible');
    stepEl.textContent = 'No frames';
    summaryEl.textContent = '';
    errorEl.textContent = '';
    stripEl.replaceChildren();
    return;
  }}
  selectedSeq = current.seq;
  emptyEl.style.display = 'none';
  canvasEl.classList.add('visible');
  const nextSrc = frameUrl(current.seq);
  if (imgEl.getAttribute('src') !== nextSrc) {{
    imgEl.onload = () => drawMarkers(current);
    imgEl.src = nextSrc;
  }} else {{
    drawMarkers(current);
  }}
  stepEl.textContent = '#' + current.seq;
  summaryEl.textContent = current.summary || '';
  errorEl.textContent = current.error || '';

  stripEl.replaceChildren();
  for (const f of frames) {{
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'thumb' + (f.seq === current.seq ? ' selected' : '');
    btn.title = '#' + f.seq + ' ' + (f.summary || '');
    btn.onclick = () => {{ followLive = f.seq === frames[frames.length - 1].seq; selectedSeq = f.seq; render(); }};
    const ti = document.createElement('img');
    ti.src = frameUrl(f.seq);
    ti.alt = '#' + f.seq;
    btn.appendChild(ti);
    stripEl.appendChild(btn);
  }}
}}

function addFrame(meta) {{
  if (frames.some((f) => f.seq === meta.seq)) return;
  frames.push(meta);
  if (frames.length > {MAX_FRAMES}) frames.shift();
  if (followLive || selectedSeq === null) selectedSeq = meta.seq;
  render();
}}

async function capture() {{
  refreshBtn.disabled = true;
  try {{
    const res = await fetch('/api/capture?sandbox=' + encodeURIComponent(CFG.key), {{
      method: 'POST',
      headers: {{ '{HEADER}': CFG.token }},
    }});
    if (res.ok) {{
      const meta = await res.json();
      addFrame(meta);
    }} else {{
      errorEl.textContent = await res.text();
    }}
  }} catch (err) {{
    errorEl.textContent = String(err);
  }} finally {{
    refreshBtn.disabled = false;
  }}
}}

refreshBtn.onclick = capture;
followBtn.onclick = () => {{
  followLive = true;
  if (frames.length) selectedSeq = frames[frames.length - 1].seq;
  render();
}};
autoBtn.onclick = () => {{
  if (autoTimer) {{
    clearInterval(autoTimer);
    autoTimer = null;
    autoBtn.textContent = 'Auto: off';
    autoBtn.classList.remove('active');
    autoBtn.setAttribute('aria-pressed', 'false');
  }} else {{
    autoBtn.textContent = 'Auto: 2s';
    autoBtn.classList.add('active');
    autoBtn.setAttribute('aria-pressed', 'true');
    capture();
    autoTimer = setInterval(capture, 2000);
  }}
}};
if (liveBtn) {{
  liveBtn.onclick = async () => {{
    liveBtn.disabled = true;
    try {{
      const res = await fetch('/api/live-url?sandbox=' + encodeURIComponent(CFG.key), {{
        method: 'POST',
        headers: {{ '{HEADER}': CFG.token }},
      }});
      if (res.ok) {{
        const data = await res.json();
        if (data.url) window.open(data.url, '_blank', 'noopener,noreferrer');
      }} else {{
        errorEl.textContent = await res.text();
      }}
    }} catch (err) {{
      errorEl.textContent = String(err);
    }} finally {{
      liveBtn.disabled = false;
    }}
  }};
}}

const source = new EventSource('/events' + qs());
source.onmessage = (event) => {{
  try {{ addFrame(JSON.parse(event.data)); }} catch {{}}
}};
render();
</script>
</body>
</html>
"""


async def _event_stream(session: DesktopSession) -> AsyncIterator[str]:
    """Yield SSE `data:` payloads for every new frame on `session` until it closes."""
    queue = session.subscribe()
    try:
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
        html = _viewer_html(session=session, token=token, nonce=nonce)
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
            headers={'Cache-Control': 'private, max-age=3600', 'X-Content-Type-Options': 'nosniff'},
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
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return JSONResponse(item.meta())

    @app.post('/api/live-url')
    async def live_url(sandbox: str = Query(...)) -> JSONResponse:
        session = _get_session(sandbox)
        try:
            url = await session.live_url()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if url is None:
            raise HTTPException(status_code=404, detail='This sandbox provider has no native live desktop URL.')
        return JSONResponse({'url': url})

    return _ViewerGuard(app, token=token, hosts=hosts)


class DesktopViewer:
    """Manages one on-demand loopback viewer server for a Pax session's desktop sandboxes."""

    def __init__(self) -> None:
        self._sessions: dict[str, DesktopSession] = {}
        self._token = secrets.token_urlsafe(24)
        self._server: WebServer | None = None

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

    async def close(self) -> None:
        """Close all SSE streams and stop the loopback server if it was started."""
        for session in list(self._sessions.values()):
            session.close()
        self._sessions.clear()
        if (server := self._server) is not None:
            self._server = None
            with anyio.CancelScope(shield=True):
                await server.stop()
