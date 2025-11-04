from typing import Any

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


class AgentAPIRouter(APIRouter):
    """FastAPI Router for Pydantic Agent."""

    def __init__(
        self,
        agent_registry: AgentRegistry,
        disable_response_api: bool = False,
        disable_completions_api: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.registry = agent_registry
        self.responses_api = AgentResponsesAPI(self.registry)
        self.completions_api = AgentChatCompletionsAPI(self.registry)
        self.models_api = AgentModelsAPI(self.registry)
        self.enable_responses_api = not disable_response_api
        self.enable_completions_api = not disable_completions_api

        # Registers OpenAI/v1 API routes
        self._register_routes()

    def _register_routes(self) -> None:
        if self.enable_completions_api:

            @self.post('/v1/chat/completions', response_model=ChatCompletion)
            async def chat_completions(
                request: ChatCompletionRequest,
            ) -> ChatCompletion | StreamingResponse:
                if getattr(request, 'stream', False):
                    return StreamingResponse(
                        self.completions_api.create_streaming_completion(request),
                        media_type='text/event-stream',
                        headers={
                            'Cache-Control': 'no-cache',
                            'Connection': 'keep-alive',
                            'Content-Type': 'text/plain; charset=utf-8',
                        },
                    )
                else:
                    return await self.completions_api.create_completion(request)

        if self.enable_responses_api:

            @self.post('/v1/responses', response_model=OpenAIResponse)
            async def responses(
                request: ResponsesRequest,
            ) -> OpenAIResponse:
                if getattr(request, 'stream', False):
                    # TODO: add streaming support for responses api
                    raise HTTPException(status_code=501)
                else:
                    return await self.responses_api.create_response(request)

        @self.get('/v1/models', response_model=ModelsResponse)
        async def get_models() -> ModelsResponse:
            return await self.models_api.list_models()

        @self.get('/v1/models' + '/{model_id}', response_model=Model)
        async def get_model(model_id: str) -> Model:
            return await self.models_api.get_model(model_id)
