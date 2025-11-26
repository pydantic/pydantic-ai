"""Factory function for creating a web chat app for a Pydantic AI agent."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TypeVar

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import AbstractBuiltinTool

from .api import AIModel, add_api_routes

DEFAULT_UI_VERSION = 'latest'
CDN_URL_TEMPLATE = 'https://cdn.jsdelivr.net/npm/@pydantic/ai-chat-ui@{version}/dist/index.html'

AgentDepsT = TypeVar('AgentDepsT')
OutputDataT = TypeVar('OutputDataT')

# In-memory cache for performance within a single session
_memory_cache: dict[str, bytes] = {}


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


def _sanitize_version(version: str) -> str:
    """Sanitize version string for use as filename."""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', version)


async def _get_ui_html(version: str) -> bytes:
    """Get UI HTML content, checking memory cache, then FS cache, then fetching from CDN."""
    if version in _memory_cache:
        return _memory_cache[version]

    cache_dir = _get_cache_dir()
    cache_file = cache_dir / f'{_sanitize_version(version)}.html'

    if cache_file.exists():
        content = cache_file.read_bytes()
        _memory_cache[version] = content
        return content

    cdn_url = CDN_URL_TEMPLATE.format(version=version)
    async with httpx.AsyncClient() as client:
        response = await client.get(cdn_url)
        response.raise_for_status()
        content = response.content

    cache_file.write_bytes(content)
    _memory_cache[version] = content

    return content


def create_web_app(
    agent: Agent[AgentDepsT, OutputDataT],
    models: list[AIModel] | None = None,
    builtin_tools: list[AbstractBuiltinTool] | None = None,
) -> Starlette:
    """Create a Starlette app that serves a web chat UI for the given agent.

    Args:
        agent: The Pydantic AI agent to serve
        models: Optional list of AI models. If not provided, the UI will have no model options.
        builtin_tools: Optional list of builtin tools. If not provided, no tools will be available.

    Returns:
        A configured Starlette application ready to be served
    """
    app = Starlette()

    app.state.agent = agent

    add_api_routes(app, models=models, builtin_tools=builtin_tools)

    async def index(request: Request) -> Response:
        """Serve the chat UI, cached on filesystem and in memory."""
        version = request.query_params.get('version')
        ui_version = version or DEFAULT_UI_VERSION

        content = await _get_ui_html(ui_version)

        return HTMLResponse(
            content=content,
            headers={
                'Cache-Control': 'public, max-age=31536000, immutable',
            },
        )

    app.router.add_route('/', index, methods=['GET'])
    app.router.add_route('/{id}', index, methods=['GET'])

    return app
