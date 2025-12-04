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
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset

from .api import ModelInfo, add_api_routes

DEFAULT_UI_VERSION = '0.0.4'
CDN_URL_TEMPLATE = 'https://cdn.jsdelivr.net/npm/@pydantic/ai-chat-ui@{version}/dist/index.html'

AgentDepsT = TypeVar('AgentDepsT')
OutputDataT = TypeVar('OutputDataT')

# Type alias for models parameter - accepts model names/instances or a dict mapping labels to models
ModelsParam = Sequence[Model | KnownModelName | str] | Mapping[str, Model | KnownModelName | str] | None


def _resolve_models(
    models: ModelsParam,
    builtin_tools: list[AbstractBuiltinTool] | None,
) -> list[ModelInfo]:
    """Convert models parameter to list of ModelInfo objects.

    Args:
        models: Model names/instances or dict mapping labels to models
        builtin_tools: Available builtin tools to check model support

    Returns:
        List of ModelInfo objects with resolved model IDs, display names, and supported tools
    """
    if models is None:
        return []

    builtin_tool_types = {type(tool) for tool in (builtin_tools or [])}
    result: list[ModelInfo] = []

    if isinstance(models, Mapping):
        items = list(models.items())
    else:
        items = [(None, m) for m in models]

    for label, model_ref in items:
        model = infer_model(model_ref)
        # Use original string if provided to preserve openai-chat: vs openai-responses: distinction
        if isinstance(model_ref, str):
            model_id = model_ref
        else:
            model_id = f'{model.system}:{model.model_name}'
        display_name = label or model.label
        model_supported_tools = model.profile.supported_builtin_tools
        supported_tool_ids = [t.kind for t in (model_supported_tools & builtin_tool_types)]
        result.append(ModelInfo(id=model_id, name=display_name, builtin_tools=supported_tool_ids))

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
    builtin_tools: list[AbstractBuiltinTool] | None = None,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
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
        builtin_tools: Optional list of builtin tools. If not provided, no tools will be available.
        toolsets: Optional list of toolsets (e.g., MCP servers). These provide additional tools
            that work with any model.
        deps: Optional dependencies to use for all requests.
        model_settings: Optional settings to use for all model requests.
        instructions: Optional extra instructions to pass to each agent run.

    Returns:
        A configured Starlette application ready to be served
    """
    resolved_models = _resolve_models(models, builtin_tools)

    app = Starlette()

    add_api_routes(
        app,
        agent=agent,
        models=resolved_models,
        builtin_tools=builtin_tools,
        toolsets=toolsets,
        deps=deps,
        model_settings=model_settings,
        instructions=instructions,
    )

    async def index(request: Request) -> Response:
        """Serve the chat UI from filesystem cache or CDN."""
        version = request.query_params.get('version')
        ui_version = version or DEFAULT_UI_VERSION

        content = await _get_ui_html(ui_version)

        return HTMLResponse(
            content=content,
            headers={
                'Cache-Control': 'public, max-age=3600',
            },
        )

    app.router.add_route('/', index, methods=['GET'])
    app.router.add_route('/{id}', index, methods=['GET'])

    return app
