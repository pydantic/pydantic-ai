import logging
from typing import Any

try:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import StreamingResponse
    from openai.types import ErrorObject
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.model import Model
    from openai.types.responses import Response
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to enable the fastapi openai compatible endpoint, '
        'you can use the `openai` and `fastapi` optional group — `pip install "pydantic-ai-slim[openai,fastapi]"`'
    ) from _import_error

from pydantic_ai.fastapi.api import AgentChatCompletionsAPI, AgentModelsAPI, AgentResponsesAPI
from pydantic_ai.fastapi.data_models import (
    ChatCompletionRequest,
    ErrorResponse,
    ModelsResponse,
    ResponsesRequest,
)
from pydantic_ai.fastapi.registry import AgentRegistry

logger = logging.getLogger(__name__)


class AgentAPIRouter(APIRouter):
    """FastAPI Router for Pydantic Agent."""

    def __init__(
        self,
        agent_registry: AgentRegistry,
        disable_response_api: bool = False,
        disable_completions_api: bool = False,
        *args: tuple[Any],
        **kwargs: tuple[Any],
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

    def _register_routes(self) -> None:  # noqa: C901
        if self.enable_completions_api:

            @self.post(
                '/v1/chat/completions',
                response_model=ChatCompletion,
            )
            async def chat_completions(  # type: ignore
                request: ChatCompletionRequest,
            ) -> ChatCompletion | StreamingResponse:
                if not request.messages:
                    raise HTTPException(
                        status_code=400,
                        detail=ErrorResponse(
                            error=ErrorObject(
                                type='invalid_request_error',
                                message='Messages cannot be empty',
                            ),
                        ).model_dump(),
                    )
                try:
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
                except Exception as e:
                    logger.error(f'Error in chat completion: {e}', exc_info=True)
                    raise HTTPException(
                        status_code=500,
                        detail=ErrorResponse(
                            error=ErrorObject(
                                type='internal_server_error',
                                message=str(e),
                            ),
                        ).model_dump(),
                    )

        if self.enable_responses_api:

            @self.post(
                '/v1/responses',
                response_model=Response,
            )
            async def responses(  # type: ignore
                request: ResponsesRequest,
            ) -> Response:
                if not request.input:
                    raise HTTPException(
                        status_code=400,
                        detail=ErrorResponse(
                            error=ErrorObject(
                                type='invalid_request_error',
                                message='Messages cannot be empty',
                            ),
                        ).model_dump(),
                    )
                try:
                    if getattr(request, 'stream', False):
                        # TODO: add streaming support for responses api
                        raise HTTPException(status_code=501)
                    else:
                        return await self.responses_api.create_response(request)
                except Exception as e:
                    logger.error(f'Error in responses: {e}', exc_info=True)
                    raise HTTPException(
                        status_code=500,
                        detail=ErrorResponse(
                            error=ErrorObject(
                                type='internal_server_error',
                                message=str(e),
                            ),
                        ).model_dump(),
                    )

        @self.get('/v1/models', response_model=ModelsResponse)
        async def get_models() -> ModelsResponse:  # type: ignore
            try:
                return await self.models_api.list_models()
            except Exception as e:
                logger.error(f'Error listing models: {e}', exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=ErrorResponse(
                        error=ErrorObject(
                            type='internal_server_error',
                            message=f'Error retrieving models: {str(e)}',
                        ),
                    ).model_dump(),
                )

        @self.get('/v1/models' + '/{model_id}', response_model=Model)
        async def get_model(model_id: str) -> Model:  # type: ignore
            try:
                return await self.models_api.get_model(model_id)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f'Error fetching model info: {e}', exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=ErrorResponse(
                        error=ErrorObject(
                            type='internal_server_error',
                            message=f'Error retrieving model: {str(e)}',
                        ),
                    ).model_dump(),
                )
