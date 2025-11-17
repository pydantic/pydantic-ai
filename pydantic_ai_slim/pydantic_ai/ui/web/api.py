"""API router for the web chat UI."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel
from pydantic.alias_generators import to_camel

from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai._adapter import VercelAIAdapter

from .agent_options import (
    AI_MODELS,
    BUILTIN_TOOL_DEFS,
    BUILTIN_TOOLS,
    AIModel,
    AIModelID,
    BuiltinTool,
    BuiltinToolID,
)


def get_agent(request: Request) -> Agent:
    """Get the agent from app state."""
    agent = getattr(request.app.state, 'agent', None)
    if agent is None:
        raise RuntimeError('No agent configured. Server must be started with a valid agent.')
    return agent


def create_api_router() -> APIRouter:
    """Create the API router for chat endpoints."""
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
            models=AI_MODELS,
            builtin_tools=BUILTIN_TOOL_DEFS,
        )

    @router.get('/api/health')
    async def health() -> dict[str, bool]:  # pyright: ignore[reportUnusedFunction]
        """Health check endpoint."""
        return {'ok': True}

    class ChatRequestExtra(BaseModel, extra='ignore', alias_generator=to_camel):
        """Extra data extracted from chat request."""

        model: AIModelID | None = None
        builtin_tools: list[BuiltinToolID] = []

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
            builtin_tools=[BUILTIN_TOOLS[tool_id] for tool_id in extra_data.builtin_tools],
        )
        return streaming_response

    return router
