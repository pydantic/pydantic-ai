from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeVar

import anyio
import anyio.to_thread
import httpx2

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.settings import ModelSettings

from ._hosts import HostValidationMiddleware, normalized_pattern
from .api import BUNDLED_UI_SDK_VERSION, ModelsParam, create_api_app

try:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, Response
    from starlette.routing import Mount
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `starlette` package to use `Agent.web()` method, '
        'you can use the `web` optional group — `pip install "pydantic-ai-slim[web]"`'
    ) from _import_error

CHAT_UI_VERSION = '2.1.0'
DEFAULT_HTML_URL = f'https://cdn.jsdelivr.net/npm/@pydantic/ai-chat-ui@{CHAT_UI_VERSION}/dist/index.html'
# `dist/index.html` references its stylesheet and its lazily-imported chunks on the CDN, so a copy
# downloaded for self-hosting still reaches the public internet at runtime. `offline/index.html` is
# a single file with all of it inlined, for air-gapped deployments.
OFFLINE_HTML_URL = f'https://cdn.jsdelivr.net/npm/@pydantic/ai-chat-ui@{CHAT_UI_VERSION}/offline/index.html'

AgentDepsT = TypeVar('AgentDepsT')
OutputDataT = TypeVar('OutputDataT')


# Cache operations run in worker threads. Serialize reads and atomic replacement because Windows
# may reject replacing a destination file while another thread has it open for reading.
_CACHE_FILE_LOCK = threading.Lock()
_OPENING_HEAD_RE = re.compile(rb'<head(?:\s[^>]*)?>', re.IGNORECASE)
_OPENING_SCRIPT_RE = re.compile(rb'<script(?:\s|>)', re.IGNORECASE)


def _normalize_public_path(path: str, parameter: str) -> str:
    if not path.startswith('/') or path.startswith('//') or any(character in path for character in '\\?#\t\n\r'):
        raise UserError(
            f"Invalid `{parameter}` {path!r}. Use an absolute same-origin path such as '/chat/' "
            'without a query or fragment.'
        )
    return f'{path.rstrip("/")}/'


def _first_uncommented_match(content: bytes, pattern: re.Pattern[bytes]) -> re.Match[bytes] | None:
    lower_content = content.lower()
    search_from = 0
    while match := pattern.search(content, search_from):
        comment_start = lower_content.find(b'<!--', search_from)
        if comment_start == -1 or match.start() < comment_start:
            return match
        comment_end = lower_content.find(b'-->', comment_start + 4)
        if comment_end == -1:
            return None
        search_from = comment_end + 3
    return None


def _inject_chat_config(content: bytes, *, base_path: str, api_path: str) -> bytes:
    config = json.dumps(
        {'basePath': base_path, 'apiPath': api_path},
        ensure_ascii=True,
        separators=(',', ':'),
    )
    config = config.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    bootstrap = f'<script>window.PYDANTIC_AI_CHAT_CONFIG={config};</script>'.encode()

    if head_match := _first_uncommented_match(content, _OPENING_HEAD_RE):
        insertion_point = head_match.end()
    elif script_match := _first_uncommented_match(content, _OPENING_SCRIPT_RE):
        insertion_point = script_match.start()
    else:
        insertion_point = len(content)

    return content[:insertion_point] + bootstrap + content[insertion_point:]


async def _get_cache_dir() -> Path:
    """Get the cache directory for storing UI HTML files.

    Uses XDG_CACHE_HOME on Unix, LOCALAPPDATA on Windows, or falls back to ~/.cache.
    """
    if os.name == 'nt':  # pragma: no cover
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))

    cache_dir = base / 'pydantic-ai' / 'web-ui'
    await anyio.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def _read_cached_file(cache_file: Path) -> bytes | None:
    """Return cached file contents, or `None` if it is missing or empty.

    An empty file is treated as a miss (a truncated/partial write left by a prior crash)
    so the caller refetches instead of serving an incomplete payload.
    """
    with _CACHE_FILE_LOCK:
        try:
            content = cache_file.read_bytes()
        except FileNotFoundError:
            return None
        return content or None


def _write_cached_file(cache_file: Path, content: bytes) -> None:
    """Write `content` to `cache_file` atomically via a same-directory temp file + `os.replace`.

    The temp file lives in `cache_file.parent` (same filesystem, so the rename is atomic) and is
    unlinked on any failure — including a write failure or interruption — so a crashed write can
    never leave the destination existing-but-incomplete nor leak a temp file.

    Kept sync and offloaded via `to_thread` as a whole: `anyio.NamedTemporaryFile` needs anyio 4.9,
    above our floor.
    """
    with _CACHE_FILE_LOCK:
        tmp_file = tempfile.NamedTemporaryFile(dir=cache_file.parent, prefix=f'.{cache_file.name}.', delete=False)
        tmp_path = Path(tmp_file.name)
        try:
            # Close the handle before the rename: Windows refuses to replace a file that still has an
            # open handle, which would break the atomic write there.
            with tmp_file:
                tmp_file.write(content)
                tmp_file.flush()
            os.replace(tmp_path, cache_file)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise


async def _get_ui_html(html_source: str | Path | None = None) -> bytes:
    """Get UI HTML content from the specified source or default CDN.

    When html_source is provided, it is used directly.
    When html_source is None, fetches from the default CDN.

    Args:
        html_source: Path or URL for the chat UI HTML. Can be:
            - None: Uses the default CDN (cached locally)
            - A Path instance: Reads from the local file
            - A URL (http:// or https://): Fetches from the URL
            - A file path string: Reads from the local file
    """
    # Use default CDN with caching
    if html_source is None:
        return await _get_cached_or_fetch(f'{CHAT_UI_VERSION}.html', DEFAULT_HTML_URL)

    # Handle Path instances
    if isinstance(html_source, Path):
        return await _read_local_file(html_source)

    # Handle URLs with filesystem caching
    if html_source.startswith(('http://', 'https://')):
        url_hash = hashlib.sha256(html_source.encode()).hexdigest()[:16]
        return await _get_cached_or_fetch(f'url_{url_hash}.html', html_source)

    # Handle local file paths (strings)
    return await _read_local_file(Path(html_source))


async def _read_local_file(path: Path) -> bytes:
    return await anyio.to_thread.run_sync(_read_local_file_sync, path)


def _read_local_file_sync(path: Path) -> bytes:
    expanded = path.expanduser()
    if expanded.is_file():
        return expanded.read_bytes()
    raise FileNotFoundError(f'Local UI file not found: {path}')


async def _get_cached_or_fetch(cache_name: str, url: str) -> bytes:
    """Return `cache_name` from the filesystem cache, fetching it from `url` on a miss.

    Filesystem reads and writes run in worker threads, so serving the UI never blocks the event
    loop.
    """
    cache_file = await _get_cache_dir() / cache_name

    if content := await anyio.to_thread.run_sync(_read_cached_file, cache_file):
        return content

    async with httpx2.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.content

    await anyio.to_thread.run_sync(_write_cached_file, cache_file, content)
    return content


def create_web_app(
    agent: Agent[AgentDepsT, OutputDataT],
    models: ModelsParam = None,
    native_tools: Sequence[AbstractNativeTool] | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    instructions: str | None = None,
    html_source: str | Path | None = None,
    sdk_version: Literal[5, 6, 7] = BUNDLED_UI_SDK_VERSION,
    allowed_hosts: Sequence[str] | None = None,
    *,
    base_path: str | None = None,
    api_path: str | None = None,
) -> Starlette:
    """Create a Starlette app that serves a web chat UI for the given agent.

    By default, the UI is fetched from a CDN and cached locally. The html_source
    parameter allows overriding this for enterprise environments, offline usage,
    or custom UI builds.

    Args:
        agent: The Pydantic AI agent to serve
        models: Models to make available in the UI. Can be:
            - A sequence of model names/instances (e.g., `['openai:gpt-5', 'anthropic:claude-sonnet-4-6']`)
            - A dict mapping display labels to model names/instances
                (e.g., `{'GPT 5': 'openai:gpt-5', 'Claude': 'anthropic:claude-sonnet-4-6'}`)
            If not provided, the UI will have no model options.
        native_tools: Optional list of additional native tools to make available in the UI.
            Tools already configured on the agent are always included but won't appear as options.
        deps: Optional dependencies to use for all requests.
        model_settings: Optional settings to use for all model requests.
        instructions: Optional extra instructions to pass to each agent run.
        html_source: Path or URL for the chat UI HTML. Can be:
            - None (default): Fetches from CDN and caches locally
            - A Path instance: Reads from the local file
            - A URL string (http:// or https://): Fetches from the URL
            - A file path string: Reads from the local file
        sdk_version: Vercel AI SDK version to target on the chat endpoint: 5, 6, or 7. Defaults to
            `7` to match the bundled v7 UI, which needs it for tool-approval streaming (7 emits the
            same wire as 6, since v7's data-stream protocol equals v6's). Only lower it to `5` when
            pairing an older UI via `html_source`.
        allowed_hosts: Additional hostnames to answer to, e.g. `['ui.example.com']` or
            `['*.example.com']` (subdomains only, so list the apex separately if you serve it).
            IP addresses and `localhost` are always allowed; any other `Host` header is refused
            with a `421`, so that a website cannot reach the UI on your machine by pointing a
            hostname it controls at you (DNS rebinding). Pass `['*']` to answer to any host, only
            if something in front of the app already authenticates requests.
        base_path: Public same-origin directory for conversation URLs and browser navigation.
            Defaults to the request's ASGI `root_path`. This only configures browser URLs; it does
            not mount the returned app at this path.
        api_path: Public same-origin directory containing the `configure` and `chat` endpoints.
            Defaults to `<root_path>/api/`. This only configures browser requests; it does not
            change the returned app's internal `/api` routes.

    Returns:
        A configured Starlette application ready to be served

    Raises:
        UserError: If an `allowed_hosts` entry is not a hostname, `*.example.com`, or `*`, or if a
            public path override is not an absolute same-origin path.
    """
    # Normalized here rather than left to the middleware so a bad pattern is reported from this call
    # instead of from the first request: Starlette builds its middleware stack lazily. Normalizing is
    # idempotent, so the middleware doing it again over the same list is a no-op.
    allowed_hosts = [normalized_pattern(pattern) for pattern in allowed_hosts or ()]
    normalized_base_path = _normalize_public_path(base_path, 'base_path') if base_path is not None else None
    normalized_api_path = _normalize_public_path(api_path, 'api_path') if api_path is not None else None

    api_app = create_api_app(
        agent=agent,
        models=models,
        native_tools=native_tools,
        deps=deps,
        model_settings=model_settings,
        instructions=instructions,
        sdk_version=sdk_version,
    )

    routes = [Mount('/api', app=api_app)]
    # Applied to the whole app rather than just `/api/chat`: a request that reaches us under a
    # hostname we don't answer to is misdirected whatever it asks for, and guarding every route
    # means a route added later is covered without anyone having to remember to opt it in.
    middleware = [Middleware(HostValidationMiddleware, allowed_hosts=allowed_hosts)]
    app = Starlette(routes=routes, middleware=middleware)

    async def index(request: Request) -> Response:
        """Serve the chat UI from filesystem cache or CDN."""
        content = await _get_ui_html(html_source)
        root_path = _normalize_public_path(request.scope.get('root_path') or '/', 'root_path')
        content = _inject_chat_config(
            content,
            base_path=normalized_base_path or root_path,
            api_path=normalized_api_path or f'{root_path}api/',
        )

        return HTMLResponse(
            content=content,
            headers={
                'Cache-Control': 'public, max-age=3600',
            },
        )

    app.router.add_route('/', index, methods=['GET'])
    app.router.add_route('/{id}', index, methods=['GET'])

    return app
