import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

try:
    from fastapi import HTTPException
    from openai.types import ErrorObject
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as Chunkhoice, ChoiceDelta
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to enable the fastapi openai compatible endpoint, '
        'you can use the `openai` and `fastapi` optional group — `pip install "pydantic-ai-slim[openai,fastapi]"`'
    ) from _import_error

from pydantic import TypeAdapter

from pydantic_ai import Agent, _utils
from pydantic_ai.fastapi.convert import (
    openai_chat_completions_2pai,
    pai_result_to_openai_completions,
)
from pydantic_ai.fastapi.data_models import ChatCompletionRequest, ErrorResponse
from pydantic_ai.fastapi.registry import AgentRegistry
from pydantic_ai.settings import ModelSettings

logger = logging.getLogger(__name__)


class AgentChatCompletionsAPI:
    """Chat completions API openai <-> pydantic-ai conversion."""

    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry

    def get_agent(self, name: str) -> Agent:
        """Retrieves agent."""
        try:
            agent = self.registry.get_completions_agent(name)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error=ErrorObject(
                        message=f'Model {name} is not available as chat completions API',
                        type='not_found_error',
                    ),
                ).model_dump(),
            )

        return agent

    async def create_completion(self, request: ChatCompletionRequest) -> ChatCompletion:
        """Create a non-streaming chat completion."""
        model_name = request.model
        agent = self.get_agent(model_name)

        model_settings_ta = TypeAdapter(ModelSettings)
        messages = openai_chat_completions_2pai(messages=request.messages)

        try:
            async with agent:
                result = await agent.run(
                    message_history=messages,
                    model_settings=model_settings_ta.validate_python(
                        {k: v for k, v in request.model_dump().items() if v is not None},
                    ),
                )

            return pai_result_to_openai_completions(
                result=result,
                model=model_name,
            )

        except Exception as e:
            logger.error(f'Error creating completion: {e}')
            raise

    async def create_streaming_completion(self, request: ChatCompletionRequest) -> AsyncGenerator[str]:
        """Create a streaming chat completion."""
        model_name = request.model
        agent = self.get_agent(model_name)
        messages = openai_chat_completions_2pai(messages=request.messages)

        role_sent = False

        async with (
            agent,
            agent.run_stream(
                message_history=messages,
            ) as result,
        ):
            async for chunk in result.stream_text(delta=True):
                delta = ChoiceDelta(
                    role='assistant' if not role_sent else None,
                    content=chunk,
                )
                role_sent = True

                stream_response = ChatCompletionChunk(
                    id=f'chatcmpl-{_utils.now_utc().isoformat()}',
                    created=int(_utils.now_utc().timestamp()),
                    model=model_name,
                    object='chat.completion.chunk',
                    choices=[
                        Chunkhoice(
                            index=0,
                            delta=delta,
                        ),
                    ],
                )

                yield f'data: {stream_response.model_dump_json()}\n\n'

            final_chunk: dict[str, Any] = {
                'id': f'chatcmpl-{int(time.time())}',
                'object': 'chat.completion.chunk',
                'model': model_name,
                'choices': [
                    {
                        'index': 0,
                        'delta': {},
                        'finish_reason': 'stop',
                    },
                ],
            }
            yield f'data: {json.dumps(final_chunk)}\n\n'
