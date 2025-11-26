"""API routes for the web chat UI."""

from pydantic import BaseModel
from pydantic.alias_generators import to_camel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import BUILTIN_TOOL_ID, AbstractBuiltinTool
from pydantic_ai.ui.vercel_ai._adapter import VercelAIAdapter


class AIModel(BaseModel, alias_generator=to_camel, populate_by_name=True):
    """Defines an AI model with its associated built-in tools."""

    id: str
    name: str
    builtin_tools: list[BUILTIN_TOOL_ID]


def _get_agent(request: Request) -> Agent:
    """Get the agent from app state."""
    agent = getattr(request.app.state, 'agent', None)
    if agent is None:
        raise RuntimeError('No agent configured. Server must be started with a valid agent.')
    return agent


class BuiltinToolInfo(BaseModel, alias_generator=to_camel, populate_by_name=True):
    """Serializable info about a builtin tool for frontend config."""

    id: str
    name: str


class _ConfigureFrontend(BaseModel, alias_generator=to_camel, populate_by_name=True):
    """Response model for frontend configuration."""

    models: list[AIModel]
    builtin_tools: list[BuiltinToolInfo]


class _ChatRequestExtra(BaseModel, extra='ignore', alias_generator=to_camel):
    """Extra data extracted from chat request."""

    model: str | None = None
    builtin_tools: list[BUILTIN_TOOL_ID] = []


def add_api_routes(
    app: Starlette,
    models: list[AIModel] | None = None,
    builtin_tools: list[AbstractBuiltinTool] | None = None,
) -> None:
    """Add API routes to a Starlette app.

    Args:
        app: The Starlette app to add routes to.
        models: Optional list of AI models. If not provided, the UI will have no model options.
        builtin_tools: Optional list of builtin tools. If not provided, no tools will be available.
    """
    _models = models or []
    _builtin_tools = builtin_tools or []
    _tools_by_kind: dict[str, AbstractBuiltinTool] = {tool.kind: tool for tool in _builtin_tools}

    async def options_chat(request: Request) -> Response:
        """Handle CORS preflight requests."""
        return Response()

    async def configure_frontend(request: Request) -> Response:
        """Endpoint to configure the frontend with available models and tools."""
        config = _ConfigureFrontend(
            models=_models,
            builtin_tools=[BuiltinToolInfo(id=tool.kind, name=tool.label) for tool in _builtin_tools],
        )
        return JSONResponse(config.model_dump(by_alias=True))

    async def health(request: Request) -> Response:
        """Health check endpoint."""
        return JSONResponse({'ok': True})

    async def post_chat(request: Request) -> Response:
        """Handle chat requests via Vercel AI Adapter."""
        agent = _get_agent(request)
        adapter = await VercelAIAdapter.from_request(request, agent=agent)
        extra_data = _ChatRequestExtra.model_validate(adapter.run_input.__pydantic_extra__)
        request_builtin_tools = [
            _tools_by_kind[tool_id] for tool_id in extra_data.builtin_tools if tool_id in _tools_by_kind
        ]
        streaming_response = await VercelAIAdapter.dispatch_request(
            request,
            agent=agent,
            model=extra_data.model,
            builtin_tools=request_builtin_tools,
        )
        return streaming_response

    app.router.add_route('/api/chat', options_chat, methods=['OPTIONS'])
    app.router.add_route('/api/chat', post_chat, methods=['POST'])
    app.router.add_route('/api/configure', configure_frontend, methods=['GET'])
    app.router.add_route('/api/health', health, methods=['GET'])
