"""Factory function for creating a web chat app for a Pydantic AI agent."""

from __future__ import annotations

from typing import TypeVar

import fastapi
import httpx
from fastapi import Request
from fastapi.responses import HTMLResponse

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import AbstractBuiltinTool

from .agent_options import AIModel, BuiltinTool
from .api import create_api_router

CDN_URL = 'https://cdn.jsdelivr.net/npm/@pydantic/ai-chat-ui/dist/index.html'

AgentDepsT = TypeVar('AgentDepsT')
OutputDataT = TypeVar('OutputDataT')


def create_chat_app(
    agent: Agent[AgentDepsT, OutputDataT],
    models: list[AIModel] | None = None,
    builtin_tools: dict[str, AbstractBuiltinTool] | None = None,
    builtin_tool_defs: list[BuiltinTool] | None = None,
) -> fastapi.FastAPI:
    """Create a FastAPI app that serves a web chat UI for the given agent.

    Args:
        agent: The Pydantic AI agent to serve
        models: Optional list of AI models (defaults to AI_MODELS)
        builtin_tools: Optional dict of builtin tool instances (defaults to BUILTIN_TOOLS)
        builtin_tool_defs: Optional list of builtin tool definitions (defaults to BUILTIN_TOOL_DEFS)

    Returns:
        A configured FastAPI application ready to be served
    """
    app = fastapi.FastAPI()

    app.state.agent = agent

    app.include_router(
        create_api_router(models=models, builtin_tools=builtin_tools, builtin_tool_defs=builtin_tool_defs)
    )

    @app.get('/')
    @app.get('/{id}')
    async def index(request: Request):  # pyright: ignore[reportUnusedFunction]
        """Serve the chat UI from CDN."""
        async with httpx.AsyncClient() as client:
            response = await client.get(CDN_URL)
            return HTMLResponse(content=response.content, status_code=response.status_code)

    return app
