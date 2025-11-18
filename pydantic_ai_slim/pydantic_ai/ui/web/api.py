"""API router for the web chat UI."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel
from pydantic.alias_generators import to_camel

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.ui.vercel_ai._adapter import VercelAIAdapter

from .agent_options import AI_MODELS, BUILTIN_TOOL_DEFS, BUILTIN_TOOLS, AIModel, BuiltinTool


def get_agent(request: Request) -> Agent:
    """Get the agent from app state."""
    agent = getattr(request.app.state, 'agent', None)
    if agent is None:
        raise RuntimeError('No agent configured. Server must be started with a valid agent.')
    return agent


def create_api_router(
    models: list[AIModel] | None = None,
    builtin_tools: dict[str, AbstractBuiltinTool] | None = None,
    builtin_tool_defs: list[BuiltinTool] | None = None,
) -> APIRouter:
    """Create the API router for chat endpoints.

    Args:
        models: Optional list of AI models (defaults to AI_MODELS)
        builtin_tools: Optional dict of builtin tool instances (defaults to BUILTIN_TOOLS)
        builtin_tool_defs: Optional list of builtin tool definitions (defaults to BUILTIN_TOOL_DEFS)
    """
    _models = models or AI_MODELS
    _builtin_tools: dict[str, AbstractBuiltinTool] = builtin_tools or BUILTIN_TOOLS
    _builtin_tool_defs = builtin_tool_defs or BUILTIN_TOOL_DEFS

    router = APIRouter()

    @router.options('/api/chat')
    def options_chat():  # pyright: ignore[reportUnusedFunction]
        """Handle CORS preflight requests."""
        pass

    class ConfigureFrontend(BaseModel, alias_generator=to_camel, populate_by_name=True):
        """Response model for frontend configuration."""

        models: list[AIModel]
        builtin_tools: list[BuiltinTool]

    @router.get('/api/configure')
    async def configure_frontend() -> ConfigureFrontend:  # pyright: ignore[reportUnusedFunction]
        """Endpoint to configure the frontend with available models and tools."""
        return ConfigureFrontend(
            models=_models,
            builtin_tools=_builtin_tool_defs,
        )

    @router.get('/api/health')
    async def health() -> dict[str, bool]:  # pyright: ignore[reportUnusedFunction]
        """Health check endpoint."""
        return {'ok': True}

    class ChatRequestExtra(BaseModel, extra='ignore', alias_generator=to_camel):
        """Extra data extracted from chat request."""

        model: str | None = None
        builtin_tools: list[str] = []

    @router.post('/api/chat')
    async def post_chat(  # pyright: ignore[reportUnusedFunction]
        request: Request, agent: Annotated[Agent, Depends(get_agent)]
    ) -> Response:
        """Handle chat requests via Vercel AI Adapter."""
        adapter = await VercelAIAdapter.from_request(request, agent=agent)
        extra_data = ChatRequestExtra.model_validate(adapter.run_input.__pydantic_extra__)
        streaming_response = await VercelAIAdapter.dispatch_request(
            request,
            agent=agent,
            model=extra_data.model,
            builtin_tools=[_builtin_tools[tool_id] for tool_id in extra_data.builtin_tools],
        )
        return streaming_response

    return router
