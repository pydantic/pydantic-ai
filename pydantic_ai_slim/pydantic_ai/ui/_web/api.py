"""API routes for the web chat UI."""

from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel
from pydantic.alias_generators import to_camel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

AgentDepsT = TypeVar('AgentDepsT')
OutputDataT = TypeVar('OutputDataT')


class ModelInfo(BaseModel, alias_generator=to_camel, populate_by_name=True):
    """Defines an AI model with its associated built-in tools."""

    id: str
    name: str
    builtin_tools: list[str]


class BuiltinToolInfo(BaseModel, alias_generator=to_camel, populate_by_name=True):
    """Serializable info about a builtin tool for frontend config."""

    id: str
    name: str


class ConfigureFrontend(BaseModel, alias_generator=to_camel, populate_by_name=True):
    """Response model for frontend configuration."""

    models: list[ModelInfo]
    builtin_tools: list[BuiltinToolInfo]


class ChatRequestExtra(BaseModel, extra='ignore', alias_generator=to_camel):
    """Extra data extracted from chat request."""

    model: str | None = None
    """Model ID selected by the user, e.g. 'openai:gpt-5'. Maps to JSON field 'model'."""
    builtin_tools: list[str] = []
    """Tool IDs selected by the user, e.g. ['web_search', 'code_execution']. Maps to JSON field 'builtinTools'."""


def validate_request_options(
    extra_data: ChatRequestExtra,
    model_ids: set[str],
    allowed_tool_ids: set[str],
) -> str | None:
    """Validate that requested model and tools are in the allowed lists.

    Returns an error message if validation fails, or None if valid.
    """
    if extra_data.model and model_ids and extra_data.model not in model_ids:
        return f'Model "{extra_data.model}" is not in the allowed models list'

    # base model also valdiates this but makes sesne to have an api check, since one could be a UI bug/misbehavior
    # the other would be a pydantic-ai bug
    # also as future proofing since we don't know how users will use this feature in the future
    invalid_tools = [t for t in extra_data.builtin_tools if t not in allowed_tool_ids]
    if invalid_tools:
        return f'Builtin tool(s) {invalid_tools} not in the allowed tools list'

    return None


# TODO remove the app arg and return a router instead (refactor the upstream logic to mount the router)
# https://github.com/pydantic/pydantic-ai/pull/3456/files#r2582659204
def add_api_routes(
    app: Starlette,
    agent: Agent[AgentDepsT, OutputDataT],
    models: list[ModelInfo] | None = None,
    builtin_tools: list[AbstractBuiltinTool] | None = None,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    instructions: str | None = None,
) -> None:
    """Add API routes to a Starlette app.

    Args:
        app: The Starlette app to add routes to.
        agent: Agent instance.
        models: Optional list of AI models. If not provided, the UI will have no model options.
        builtin_tools: Optional list of builtin tools. If not provided, no tools will be available.
        toolsets: Optional list of toolsets (e.g., MCP servers). These provide additional tools.
        deps: Optional dependencies to use for all requests.
        model_settings: Optional settings to use for all model requests.
        instructions: Optional extra instructions to pass to each agent run.
    """
    models = models or []
    model_ids = {m.id for m in models}
    builtin_tools = builtin_tools or []
    allowed_tool_ids = {tool.unique_id for tool in builtin_tools}
    toolsets = list(toolsets) if toolsets else None

    async def options_chat(request: Request) -> Response:
        """Handle CORS preflight requests."""
        return Response()

    async def configure_frontend(request: Request) -> Response:
        """Endpoint to configure the frontend with available models and tools."""
        config = ConfigureFrontend(
            models=models,
            builtin_tools=[BuiltinToolInfo(id=tool.unique_id, name=tool.label) for tool in builtin_tools],
        )
        return JSONResponse(config.model_dump(by_alias=True))

    async def health(request: Request) -> Response:
        """Health check endpoint."""
        return JSONResponse({'ok': True})

    async def post_chat(request: Request) -> Response:
        """Handle chat requests via Vercel AI Adapter."""
        adapter = await VercelAIAdapter[AgentDepsT, OutputDataT].from_request(request, agent=agent)
        extra_data = ChatRequestExtra.model_validate(adapter.run_input.__pydantic_extra__)

        if error := validate_request_options(extra_data, model_ids, allowed_tool_ids):
            return JSONResponse({'error': error}, status_code=400)

        request_builtin_tools = [tool for tool in builtin_tools if tool.unique_id in extra_data.builtin_tools]
        streaming_response = await VercelAIAdapter[AgentDepsT, OutputDataT].dispatch_request(
            request,
            agent=agent,
            model=extra_data.model,
            builtin_tools=request_builtin_tools,
            toolsets=toolsets,
            deps=deps,
            model_settings=model_settings,
            instructions=instructions,
        )
        return streaming_response

    app.router.add_route('/api/chat', options_chat, methods=['OPTIONS'])
    app.router.add_route('/api/chat', post_chat, methods=['POST'])
    app.router.add_route('/api/configure', configure_frontend, methods=['GET'])
    app.router.add_route('/api/health', health, methods=['GET'])
