try:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import StreamingResponse
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.model import Model
    from openai.types.responses import Response as OpenAIResponse
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` and `fastapi` packages to enable the fastapi openai compatible endpoint, '
        'you can use the `chat-completion` optional group — `pip install "pydantic-ai-slim[chat-completion]"`'
    ) from _import_error

from pydantic_ai.fastapi.api import AgentChatCompletionsAPI, AgentModelsAPI, AgentResponsesAPI
from pydantic_ai.fastapi.data_models import (
    ChatCompletionRequest,
    ModelsResponse,
    ResponsesRequest,
)
from pydantic_ai.fastapi.registry import AgentRegistry


def create_agent_router(
    agent_registry: AgentRegistry,
    disable_responses_api: bool = False,
    disable_completions_api: bool = False,
    api_router: APIRouter | None = None,
) -> APIRouter:
    """FastAPI Router factory for Pydantic Agent exposure as OpenAI endpoint."""
    if api_router is None:
        api_router = APIRouter()
    responses_api = AgentResponsesAPI(agent_registry)
    completions_api = AgentChatCompletionsAPI(agent_registry)
    models_api = AgentModelsAPI(agent_registry)
    enable_responses_api = not disable_responses_api
    enable_completions_api = not disable_completions_api

    if enable_completions_api:

        @api_router.post('/v1/chat/completions', response_model=ChatCompletion)
        async def chat_completions(  # type: ignore[reportUnusedFunction]
            request: ChatCompletionRequest,
        ) -> ChatCompletion | StreamingResponse:
            if getattr(request, 'stream', False):
                return StreamingResponse(
                    completions_api.create_streaming_completion(request),
                    media_type='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'Content-Type': 'text/plain; charset=utf-8',
                    },
                )
            else:
                return await completions_api.create_completion(request)

    if enable_responses_api:

        @api_router.post('/v1/responses', response_model=OpenAIResponse)
        async def responses(  # type: ignore[reportUnusedFunction]
            request: ResponsesRequest,
        ) -> OpenAIResponse:
            if getattr(request, 'stream', False):
                # TODO: add streaming support for responses api
                raise HTTPException(status_code=501)
            else:
                return await responses_api.create_response(request)

    @api_router.get('/v1/models', response_model=ModelsResponse)
    async def get_models() -> ModelsResponse:  # type: ignore[reportUnusedFunction]
        return await models_api.list_models()

    @api_router.get('/v1/models' + '/{model_id}', response_model=Model)
    async def get_model(model_id: str) -> Model:  # type: ignore[reportUnusedFunction]
        return await models_api.get_model(model_id)

    return api_router
