import logging
from collections.abc import AsyncGenerator

try:
    from fastapi import HTTPException
    from openai.types import ErrorObject
    from openai.types.responses import Response
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to enable the fastapi openai compatible endpoint, '
        'you can use the `openai` and `fastapi` optional group — `pip install "pydantic-ai-slim[openai,fastapi]"`'
    ) from _import_error

from pydantic_ai import Agent
from pydantic_ai.fastapi.convert import (
    openai_responses_input_to_pai,
    pai_result_to_openai_responses,
)
from pydantic_ai.fastapi.data_models import ErrorResponse, ResponsesRequest
from pydantic_ai.fastapi.registry import AgentRegistry
from pydantic_ai.models.openai import (
    OpenAIResponsesModelSettings,
)

logger = logging.getLogger(__name__)


class AgentResponsesAPI:
    """Responses API openai <-> pydantic-ai conversion."""

    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry

    def get_agent(self, name: str) -> Agent:
        """Retrieves agent."""
        try:
            agent = self.registry.get_responses_agent(name)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error=ErrorObject(
                        message=f'Model {name} is not available as responses API',
                        type='not_found_error',
                    ),
                ).model_dump(),
            )

        return agent

    async def create_response(self, request: ResponsesRequest) -> Response:
        """Create a non-streaming chat completion."""
        model_name = request.model
        agent = self.get_agent(model_name)

        model_settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
        messages = openai_responses_input_to_pai(items=request.input)

        try:
            async with agent:
                result = await agent.run(
                    message_history=messages,
                    model_settings=model_settings,
                )
            return pai_result_to_openai_responses(
                result=result,
                model=model_name,
            )
        except Exception as e:
            logger.error(f'Error creating completion: {e}')
            raise

    async def create_streaming_response(self, request: ResponsesRequest) -> AsyncGenerator[str]:
        """Create a streaming chat completion."""
        raise NotImplementedError
