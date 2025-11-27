"""Factory function for creating a web chat app for a Pydantic AI agent."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeVar

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.models import KnownModelName, Model, infer_model

from .api import AIModel, add_api_routes


def format_model_display_name(model_name: str) -> str:
    """Format model name for display in UI.

    Handles common patterns:
    - gpt-5 -> GPT 5
    - claude-sonnet-4-5 -> Claude Sonnet 4.5
    - gemini-2.5-pro -> Gemini 2.5 Pro
    """
    parts = model_name.split('-')
    result: list[str] = []

    for i, part in enumerate(parts):
        if i == 0 and part.lower() == 'gpt':
            result.append(part.upper())
        elif part.replace('.', '').isdigit():
            if result and result[-1].replace('.', '').isdigit():
                result[-1] = f'{result[-1]}.{part}'
            else:
                result.append(part)
        else:
            result.append(part.capitalize())

    return ' '.join(result)


DEFAULT_UI_VERSION = 'latest'
CDN_URL_TEMPLATE = 'https://cdn.jsdelivr.net/npm/@pydantic/ai-chat-ui@{version}/dist/index.html'

AgentDepsT = TypeVar('AgentDepsT')
OutputDataT = TypeVar('OutputDataT')

# Type alias for models parameter - accepts model names/instances or a dict mapping labels to models
ModelsParam = Sequence[Model | KnownModelName | str] | Mapping[str, Model | KnownModelName | str] | None

# In-memory cache for performance within a single session
_memory_cache: dict[str, bytes] = {}


def _resolve_models(
    models: ModelsParam,
    builtin_tools: list[AbstractBuiltinTool] | None,
) -> list[AIModel]:
    """Convert models parameter to list of AIModel objects.

    Args:
        models: Model names/instances or dict mapping labels to models
        builtin_tools: Available builtin tools to check model support

    Returns:
        List of AIModel objects with resolved model IDs, display names, and supported tools
    """
    if models is None:
        return []

    builtin_tool_ids = {tool.kind for tool in (builtin_tools or [])}
    result: list[AIModel] = []

    if isinstance(models, Mapping):
        items = list(models.items())
    else:
        items = [(None, m) for m in models]

    for label, model_ref in items:
        model = infer_model(model_ref)
        model_id = f'{model.system}:{model.model_name}'
        display_name = label or format_model_display_name(model.model_name)
        model_supported_tools = model.supported_builtin_tools()
        supported_tool_ids = list(model_supported_tools & builtin_tool_ids)
        result.append(AIModel(id=model_id, name=display_name, builtin_tools=supported_tool_ids))  # pyright: ignore[reportArgumentType]

    return result


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
    models: ModelsParam = None,
    builtin_tools: list[AbstractBuiltinTool] | None = None,
) -> Starlette:
    """Create a Starlette app that serves a web chat UI for the given agent.

    Args:
        agent: The Pydantic AI agent to serve
        models: Models to make available in the UI. Can be:
            - A sequence of model names/instances (e.g., `['openai:gpt-5', 'anthropic:claude-sonnet-4-5']`)
            - A dict mapping display labels to model names/instances
              (e.g., `{'GPT 5': 'openai:gpt-5', 'Claude': 'anthropic:claude-sonnet-4-5'}`)
            If not provided, the UI will have no model options.
        builtin_tools: Optional list of builtin tools. If not provided, no tools will be available.

    Returns:
        A configured Starlette application ready to be served
    """
    resolved_models = _resolve_models(models, builtin_tools)

    app = Starlette()

    app.state.agent = agent

    add_api_routes(app, models=resolved_models, builtin_tools=builtin_tools)

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
