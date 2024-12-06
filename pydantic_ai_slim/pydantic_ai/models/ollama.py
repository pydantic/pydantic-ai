from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from httpx import AsyncClient as AsyncHTTPClient

from . import (
    AbstractToolDefinition,
    AgentModel,
    Model,
    cached_async_http_client,
    check_allow_model_requests,
)

try:
    from openai import AsyncOpenAI
    from openai.types import ChatModel, chat
except ImportError as e:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        "you can use the `openai` optional group — `pip install 'pydantic-ai[openai]'`"
    ) from e

from .openai import OpenAIAgentModel


# Inherits from this model
@dataclass(init=False)
class OllamaModel(Model):
    """A model that uses the OpenAI API.

    Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    model_name: ChatModel
    client: AsyncOpenAI = field(repr=False)

    def __init__(
        self,
        model_name: ChatModel,
        *,
        base_url: str = 'http://localhost:11434/v1/',
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        """Initialize an Ollama model.

        Ollama has built-in compatability with the OpenAI chat completions API ([source](https://ollama.com/blog/openai-compatibility)), and so these models are built ontop of that

        Args:
            model_name: The name of the OpenAI model to use. List of model names available
                [here](https://github.com/openai/openai-python/blob/v1.54.3/src/openai/types/chat_model.py#L7)
                (Unfortunately, despite being ask to do so, OpenAI do not provide `.inv` files for their API).
            base_url: The base url for the ollama requests. The default value is the ollama default
            openai_client: An existing
                [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
                client to use, if provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        self.model_name: ChatModel = model_name
        if openai_client is not None:
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            self.client = openai_client
        elif http_client is not None:
            # API key is not required for ollama but a value is required to create the client
            self.client = AsyncOpenAI(base_url=base_url, api_key='ollama', http_client=http_client)
        else:
            # API key is not required for ollama but a value is required to create the client
            self.client = AsyncOpenAI(base_url=base_url, api_key='ollama', http_client=cached_async_http_client())

    async def agent_model(
        self,
        function_tools: Mapping[str, AbstractToolDefinition],
        allow_text_result: bool,
        result_tools: Sequence[AbstractToolDefinition] | None,
    ) -> AgentModel:
        check_allow_model_requests()
        tools = [self._map_tool_definition(r) for r in function_tools.values()]
        if result_tools is not None:
            tools += [self._map_tool_definition(r) for r in result_tools]
        return OpenAIAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            tools,
        )

    def name(self) -> str:
        return f'ollama:{self.model_name}'

    @staticmethod
    def _map_tool_definition(f: AbstractToolDefinition) -> chat.ChatCompletionToolParam:
        return {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description,
                'parameters': f.json_schema,
            },
        }
