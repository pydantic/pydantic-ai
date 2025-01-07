from __future__ import annotations as _annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Union

from httpx import AsyncClient as AsyncHTTPClient

from ..messages import ModelResponse, ModelResponsePart, TextPart, ToolCallPart
from ..tools import ToolDefinition
from . import (
    AgentModel,
    Model,
    cached_async_http_client,
)

try:
    from openai import AsyncOpenAI
    from openai.types import chat

    from .openai import OpenAIAgentModel
except ImportError as e:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        "you can use the `openai` optional group — `pip install 'pydantic-ai-slim[openai]'`"
    ) from e


from .openai import OpenAIModel

CommonOllamaModelNames = Literal[
    'codellama',
    'gemma',
    'gemma2',
    'llama3',
    'llama3.1',
    'llama3.2',
    'llama3.2-vision',
    'llama3.3',
    'mistral',
    'mistral-nemo',
    'mixtral',
    'phi3',
    'qwq',
    'qwen',
    'qwen2',
    'qwen2.5',
    'starcoder2',
]
"""This contains just the most common ollama models.

For a full list see [ollama.com/library](https://ollama.com/library).
"""
OllamaModelName = Union[CommonOllamaModelNames, str]
"""Possible ollama models.

Since Ollama supports hundreds of models, we explicitly list the most models but
allow any name in the type hints.
"""


class NestedJSONDecoder(json.JSONDecoder):
    """Modification of the built-in json decoder to enable decoding of nested models provided by the Ollama API."""

    def decode(self, s, _w=json.decoder.WHITESPACE.match):  # type: ignore
        parsed = super().decode(s)
        return self._decode_nested(parsed)

    def _decode_nested(self, obj: dict[str, Any] | list[dict[str, Any] | str] | str) -> str | dict[str, Any]:
        if isinstance(obj, dict):
            return {key: self._decode_nested(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._decode_nested(item) for item in obj]  # type: ignore
        elif isinstance(obj, str):
            try:
                return self._decode_nested(json.loads(obj))
            except json.JSONDecodeError:
                return obj


@dataclass
class OllamaAgentModel(OpenAIAgentModel):
    """Agent model for the Ollama API. Contains special handling for the escape characters in ollama responses."""

    @staticmethod
    def _process_response(response: chat.ChatCompletion) -> ModelResponse:
        """Override that deals with the extra escape characters in ollama responses."""
        timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        choice = response.choices[0]
        items: list[ModelResponsePart] = []

        if choice.message.content is not None:
            items.append(TextPart(choice.message.content))

        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                try:
                    items.append(
                        ToolCallPart.from_raw_args(
                            c.function.name, NestedJSONDecoder().decode(c.function.arguments), c.id
                        )
                    )
                except json.JSONDecodeError:
                    items.append(ToolCallPart.from_raw_args(c.function.name, c.function.arguments, c.id))

        return ModelResponse(items, timestamp=timestamp)


@dataclass(init=False)
class OllamaModel(Model):
    """A model that implements Ollama using the OpenAI API.

    Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact with the Ollama server.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    model_name: OllamaModelName
    openai_model: OpenAIModel

    def __init__(
        self,
        model_name: OllamaModelName,
        *,
        base_url: str | None = 'http://localhost:11434/v1/',
        api_key: str = 'ollama',
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        """Initialize an Ollama model.

        Ollama has built-in compatability for the OpenAI chat completions API ([source](https://ollama.com/blog/openai-compatibility)), so we reuse the
        [`OpenAIModel`][pydantic_ai.models.openai.OpenAIModel] here.

        Args:
            model_name: The name of the Ollama model to use. List of models available [here](https://ollama.com/library)
                You must first download the model (`ollama pull <MODEL-NAME>`) in order to use the model
            base_url: The base url for the ollama requests. The default value is the ollama default
            api_key: The API key to use for authentication. Defaults to 'ollama' for local instances,
                but can be customized for proxy setups that require authentication
            openai_client: An existing
                [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
                client to use, if provided, `base_url` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        self.model_name = model_name
        if openai_client is not None:
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            self.openai_model = OpenAIModel(model_name=model_name, openai_client=openai_client)
        else:
            # API key is not required for ollama but a value is required to create the client
            http_client_ = http_client or cached_async_http_client()
            oai_client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client_)
            self.openai_model = OpenAIModel(model_name=model_name, openai_client=oai_client)

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

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        tools = [self._map_tool_definition(r) for r in function_tools]
        if result_tools:
            tools += [self._map_tool_definition(r) for r in result_tools]

        return OllamaAgentModel(
            client=self.openai_model.client,
            model_name=self.openai_model.model_name,
            allow_text_result=allow_text_result,
            tools=tools,
        )

    def name(self) -> str:
        return f'ollama:{self.model_name}'
