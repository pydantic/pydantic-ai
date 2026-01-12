from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import httpx

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.settings import ModelSettings

from .api import ModelsParam, create_api_app

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

DEFAULT_UI_VERSION = '1.0.0'
CDN_URL_TEMPLATE = 'https://cdn.jsdelivr.net/npm/@pydantic/ai-chat-ui@{version}/dist/index.html'

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


def _sanitize_version(version: str) -> str:
    """Sanitize version string for use as filename."""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', version)


async def _get_ui_html(version: str) -> bytes:
    """Get UI HTML content from filesystem cache or fetch from CDN."""
    cache_dir = _get_cache_dir()
    cache_file = cache_dir / f'{_sanitize_version(version)}.html'

    if cache_file.exists():
        return cache_file.read_bytes()

    cdn_url = CDN_URL_TEMPLATE.format(version=version)
    async with httpx.AsyncClient() as client:
        response = await client.get(cdn_url)
        response.raise_for_status()
        content = response.content

    cache_file.write_bytes(content)
    return content


def create_web_app(
    agent: Agent[AgentDepsT, OutputDataT],
    models: ModelsParam = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    instructions: str | None = None,
) -> Starlette:
    """Create a Starlette app that serves a web chat UI for the given agent.

    Args:
        agent: The Pydantic AI agent to serve
        models: Models to make available in the UI. Can be:
            - A sequence of model names/instances (e.g., `['openai:gpt-5', 'anthropic:claude-sonnet-4-5']`)
            - A dict mapping display labels to model names/instances
                (e.g., `{'GPT 5': 'openai:gpt-5', 'Claude': 'anthropic:claude-sonnet-4-5'}`)
            If not provided, the UI will have no model options.
        builtin_tools: Optional list of additional builtin tools to make available in the UI.
            Tools already configured on the agent are always included but won't appear as options.
        deps: Optional dependencies to use for all requests.
        model_settings: Optional settings to use for all model requests.
        instructions: Optional extra instructions to pass to each agent run.

    Returns:
        A configured Starlette application ready to be served
    """
    api_app = create_api_app(
        agent=agent,
        models=models,
        builtin_tools=builtin_tools,
        deps=deps,
        model_settings=model_settings,
        instructions=instructions,
    )

    routes = [Mount('/api', app=api_app)]
    app = Starlette(routes=routes)

    async def index(request: Request) -> Response:
        """Serve the chat UI from filesystem cache or CDN."""
        version = request.query_params.get('version')
        ui_version = version or DEFAULT_UI_VERSION

        try:
            content = await _get_ui_html(ui_version)
        except httpx.HTTPStatusError as e:
            return HTMLResponse(
                content=f'Failed to fetch UI version "{ui_version}": {e.response.status_code}',
                status_code=502,
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
