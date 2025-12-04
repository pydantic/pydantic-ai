"""API routes for the web chat UI."""

from collections.abc import Mapping, Sequence
from typing import TypeVar

from pydantic import BaseModel
from pydantic.alias_generators import to_camel
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.models import KnownModelName, Model, infer_model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

AgentDepsT = TypeVar('AgentDepsT')
OutputDataT = TypeVar('OutputDataT')

# Type alias for models parameter - accepts model names/instances or a dict mapping labels to models
ModelsParam = Sequence[Model | KnownModelName | str] | Mapping[str, Model | KnownModelName | str] | None


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


def create_api_routes(
    agent: Agent[AgentDepsT, OutputDataT],
    models: ModelsParam = None,
    builtin_tools: list[AbstractBuiltinTool] | None = None,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    instructions: str | None = None,
) -> list[Route]:
    """Create API routes for the web chat UI.

    Args:
        agent: Agent instance.
        models: Models to make available in the UI. Can be:
            - A sequence of model names/instances (e.g., `['openai:gpt-5', Model(...)]`)
            - A dict mapping display labels to model names/instances
            If not provided, the UI will have no model options.
        builtin_tools: Optional list of builtin tools. If not provided, no tools will be available.
        toolsets: Optional list of toolsets (e.g., MCP servers). These provide additional tools.
        deps: Optional dependencies to use for all requests.
        model_settings: Optional settings to use for all model requests.
        instructions: Optional extra instructions to pass to each agent run.

    Returns:
        A list of Starlette Route objects for the API endpoints.
    """
    # Build model ID → original reference mapping and ModelInfo list for frontend
    model_id_to_ref: dict[str, Model | str] = {}
    model_infos: list[ModelInfo] = []
    builtin_tools = builtin_tools or []
    builtin_tool_types = {type(tool) for tool in builtin_tools}

    if models is not None:
        items = list(models.items()) if isinstance(models, Mapping) else [(None, m) for m in models]
        for label, model_ref in items:
            model = infer_model(model_ref)
            # Use original string if provided to preserve openai-chat: vs openai-responses: distinction
            model_id = model_ref if isinstance(model_ref, str) else f'{model.system}:{model.model_name}'
            display_name = label or model.label
            model_supported_tools = model.profile.supported_builtin_tools
            supported_tool_ids = [t.kind for t in (model_supported_tools & builtin_tool_types)]

            model_id_to_ref[model_id] = model_ref  # Store original reference
            model_infos.append(ModelInfo(id=model_id, name=display_name, builtin_tools=supported_tool_ids))

    model_ids = set(model_id_to_ref.keys())
    allowed_tool_ids = {tool.unique_id for tool in builtin_tools}
    toolsets = list(toolsets) if toolsets else None

    async def options_chat(request: Request) -> Response:
        """Handle CORS preflight requests."""
        return Response()

    async def configure_frontend(request: Request) -> Response:
        """Endpoint to configure the frontend with available models and tools."""
        config = ConfigureFrontend(
            models=model_infos,
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

        model_ref = model_id_to_ref.get(extra_data.model) if extra_data.model else None
        request_builtin_tools = [tool for tool in builtin_tools if tool.unique_id in extra_data.builtin_tools]
        streaming_response = await VercelAIAdapter[AgentDepsT, OutputDataT].dispatch_request(
            request,
            agent=agent,
            model=model_ref,
            builtin_tools=request_builtin_tools,
            toolsets=toolsets,
            deps=deps,
            model_settings=model_settings,
            instructions=instructions,
        )
        return streaming_response

    return [
        Route('/api/chat', options_chat, methods=['OPTIONS']),
        Route('/api/chat', post_chat, methods=['POST']),
        Route('/api/configure', configure_frontend, methods=['GET']),
        Route('/api/health', health, methods=['GET']),
    ]
