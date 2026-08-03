from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeVar

import httpx

from pydantic_ai import Agent
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.settings import ModelSettings

from .api import BUNDLED_UI_SDK_VERSION, ModelsParam, create_api_app

try:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, Response
    from starlette.routing import Mount
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `starlette` package to use `Agent.web()` method, '
        'you can use the `web` optional group — `pip install "pydantic-ai-slim[web]"`'
    ) from _import_error

CHAT_UI_VERSION = '2.0.0'
DEFAULT_HTML_URL = f'https://cdn.jsdelivr.net/npm/@pydantic/ai-chat-ui@{CHAT_UI_VERSION}/dist/index.html'

AgentDepsT = TypeVar('AgentDepsT')
OutputDataT = TypeVar('OutputDataT')


def _get_cache_dir() -> Path:
    """Get the cache directory for storing UI HTML files.

    Uses XDG_CACHE_HOME on Unix, LOCALAPPDATA on Windows, or falls back to ~/.cache.
    """
    if os.name == 'nt':  # pragma: no cover
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))

    cache_dir = base / 'pydantic-ai' / 'web-ui'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _read_cached_file(cache_file: Path) -> bytes | None:
    """Return cached file contents, or `None` if it is missing or empty.

    An empty file is treated as a miss (a truncated/partial write left by a prior crash)
    so the caller refetches instead of serving an incomplete payload.
    """
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
    """
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
        cache_dir = _get_cache_dir()
        cache_file = cache_dir / f'{CHAT_UI_VERSION}.html'

        if content := _read_cached_file(cache_file):
            return content

        async with httpx.AsyncClient() as client:
            response = await client.get(DEFAULT_HTML_URL)
            response.raise_for_status()
            content = response.content

        _write_cached_file(cache_file, content)
        return content

    # Handle Path instances
    if isinstance(html_source, Path):
        html_source = html_source.expanduser()
        if html_source.is_file():
            return html_source.read_bytes()
        raise FileNotFoundError(f'Local UI file not found: {html_source}')

    # Handle URLs with filesystem caching
    if html_source.startswith(('http://', 'https://')):
        cache_dir = _get_cache_dir()
        url_hash = hashlib.sha256(html_source.encode()).hexdigest()[:16]
        cache_file = cache_dir / f'url_{url_hash}.html'

        if content := _read_cached_file(cache_file):
            return content

        async with httpx.AsyncClient() as client:
            response = await client.get(html_source)
            response.raise_for_status()
            content = response.content

        _write_cached_file(cache_file, content)
        return content

    # Handle local file paths (strings)
    local_path = Path(html_source).expanduser()
    if local_path.is_file():
        return local_path.read_bytes()
    raise FileNotFoundError(f'Local UI file not found: {html_source}')


def create_web_app(
    agent: Agent[AgentDepsT, OutputDataT],
    models: ModelsParam = None,
    native_tools: Sequence[AbstractNativeTool] | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    instructions: str | None = None,
    html_source: str | Path | None = None,
    sdk_version: Literal[5, 6, 7] = BUNDLED_UI_SDK_VERSION,
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

    Returns:
        A configured Starlette application ready to be served
    """
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
    app = Starlette(routes=routes)

    async def index(request: Request) -> Response:
        """Serve the chat UI from filesystem cache or CDN."""
        content = await _get_ui_html(html_source)

        return HTMLResponse(
            content=content,
            headers={
                'Cache-Control': 'public, max-age=3600',
            },
        )

    app.router.add_route('/', index, methods=['GET'])
    app.router.add_route('/{id}', index, methods=['GET'])

    return app
