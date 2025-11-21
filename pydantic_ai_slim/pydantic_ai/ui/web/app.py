"""Factory function for creating a web chat app for a Pydantic AI agent."""

from __future__ import annotations

from typing import TypeVar

import fastapi
import httpx
from fastapi import Query, Request
from fastapi.responses import HTMLResponse

from pydantic_ai import Agent

from .agent_options import AIModel, BuiltinToolDef
from .api import create_api_router

DEFAULT_UI_VERSION = 'latest'
CDN_URL_TEMPLATE = 'https://cdn.jsdelivr.net/npm/@pydantic/ai-chat-ui@{version}/dist/index.html'

AgentDepsT = TypeVar('AgentDepsT')
OutputDataT = TypeVar('OutputDataT')

_cached_ui_html: dict[str, bytes] = {}


def create_web_app(
    agent: Agent[AgentDepsT, OutputDataT],
    models: list[AIModel] | None = None,
    builtin_tool_defs: list[BuiltinToolDef] | None = None,
) -> fastapi.FastAPI:
    """Create a FastAPI app that serves a web chat UI for the given agent.

    Args:
        agent: The Pydantic AI agent to serve
        models: Optional list of AI models (defaults to AI_MODELS)
        builtin_tool_defs: Optional list of builtin tool definitions. Each definition includes
            the tool ID, display name, and tool instance (defaults to DEFAULT_BUILTIN_TOOL_DEFS)

    Returns:
        A configured FastAPI application ready to be served
    """
    app = fastapi.FastAPI()

    app.state.agent = agent

    app.include_router(create_api_router(models=models, builtin_tool_defs=builtin_tool_defs))

    @app.get('/')
    @app.get('/{id}')
    async def index(request: Request, version: str | None = Query(None)):  # pyright: ignore[reportUnusedFunction]
        """Serve the chat UI from CDN, cached on the client on first use.

        Accepts an optional query param for the version to load (e.g. '1.0.0'). Defaults to pinned version.
        """
        ui_version = version or DEFAULT_UI_VERSION
        cdn_url = CDN_URL_TEMPLATE.format(version=ui_version)

        if ui_version not in _cached_ui_html:
            async with httpx.AsyncClient() as client:
                response = await client.get(cdn_url)
                response.raise_for_status()
                _cached_ui_html[ui_version] = response.content

        return HTMLResponse(
            content=_cached_ui_html[ui_version],
            headers={
                'Cache-Control': 'public, max-age=31536000, immutable',
            },
        )

    return app
