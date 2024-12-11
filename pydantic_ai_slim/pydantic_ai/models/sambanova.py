from __future__ import annotations as _annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Literal, Union

from httpx import AsyncClient as AsyncHTTPClient

from ..messages import Message, ModelAnyResponse, ModelStructuredResponse, ModelTextResponse, ToolCall
from ..tools import ToolDefinition
from . import (
    AgentModel,
    Model,
    cached_async_http_client,
    check_allow_model_requests,
)
from .openai import OpenAIAgentModel

try:
    from openai import AsyncOpenAI, AsyncStream
    from openai.types import chat
    from openai.types.chat import ChatCompletionChunk
except ImportError as e:
    raise ImportError(
        'Please install `openai` to use the OpenAI client, '
        "you can use the `openai` optional group — `pip install 'pydantic-ai[openai]'`"
    ) from e


SambaNovaModelNames = Literal[
    'Meta-Llama-3.1-8B-Instruct',
    'Meta-Llama-3.1-70B-Instruct',
    'Meta-Llama-3.1-405B-Instruct',
    'Meta-Llama-3.2-1B-Instruct',
    'Meta-Llama-3.2-3B-Instruct',
    'Llama-3.2-11B-Vision-Instruct',
    'Llama-3.2-90B-Vision-Instruct',
    'Meta-Llama-Guard-3-8B',
    'Qwen2.5-Coder-32B-Instruct',
    'Qwen2.5-72B-Instruct',
]

SambaNovaModelName = Union[SambaNovaModelNames, str]


@dataclass(init=False)
class SambaNovaModel(Model):
    """A model that uses the SambaNova models thought OpenAI client.

    Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact
    with the SambaNova APIs

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncOpenAI = field(repr=False)

    def __init__(
        self,
        model_name: SambaNovaModelName,
        *,
        base_url: str | None = 'https://api.sambanova.ai/v1/',
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        """Initialize an SambaNovaModel model.

        SambaNova models have built-in compatibility with OpenAI chat completions API, so we use the
        OpenAI client.

        Args:
            model_name: The name of the SambaNova model to use. List of model names available
                [here](https://cloud.sambanova.ai)
            base_url: The base url for the SambaNova requests. The default value is the SambaNovaCloud URL
            api_key: The API key to use for authentication, if not provided, the `SAMBANOVA_API_KEY` environment variable
                will be used if available.
            openai_client: An existing
                [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
                client to use, if provided,`base_url`, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        self.model_name: SambaNovaModelName = model_name
        if api_key is None:
            api_key = os.environ.get('SAMBANOVA_API_KEY')
        if openai_client is not None:
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self.client = openai_client
        elif http_client is not None:
            self.client = AsyncOpenAI(api_key=api_key, http_client=http_client)
        else:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=cached_async_http_client())

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        check_allow_model_requests()
        tools = [self._map_tool_definition(r) for r in function_tools]
        if result_tools:
            tools += [self._map_tool_definition(r) for r in result_tools]
        return SambaNovaAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            tools,
        )

    def name(self) -> str:
        return f'sambanova:{self.model_name}'

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> chat.ChatCompletionToolParam:
        return {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description,
                'parameters': f.parameters_json_schema,
            },
        }


@dataclass
class SambaNovaAgentModel(OpenAIAgentModel):
    """Implementation of `AgentModel` for SambaNova models.

    SambaNova models have built-in compatibility with OpenAI chat completions API,
    so we inherit from[`OpenAIModelAgent`][pydantic_ai.models.openai.OpenAIModel] here.
    """

    @staticmethod
    def _process_response(response: chat.ChatCompletion) -> ModelAnyResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        choice = response.choices[0]
        if choice.message.tool_calls is not None:
            calls: List[ToolCall] = []
            for tool_call in choice.message.tool_calls:
                if isinstance(tool_call.function.arguments, dict):
                    calls.append(
                        ToolCall.from_json(
                            tool_call.function.name, json.dumps(tool_call.function.arguments), tool_call.id
                        )
                    )
                else:
                    calls.append(
                        ToolCall.from_json(tool_call.function.name, tool_call.function.arguments, tool_call.id)
                    )
            return ModelStructuredResponse(calls, timestamp=timestamp)
        else:
            assert choice.message.content is not None, choice
            return ModelTextResponse(choice.message.content, timestamp=timestamp)

    async def _completions_create(
        self, messages: list[Message], stream: bool
        ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
        if stream:
            if self.tools:
                raise NotImplementedError('tool calling when streaming not supported')
        return await super()._completions_create(messages, stream)
