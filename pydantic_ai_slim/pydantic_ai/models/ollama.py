from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Literal, Union

from httpx import AsyncClient as AsyncHTTPClient

from ..tools import ToolDefinition
from . import (
    AgentModel,
    Model,
    cached_async_http_client,
    check_allow_model_requests,
)

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        "you can use the `openai` optional group — `pip install 'pydantic-ai-slim[openai]'`"
    ) from e


from .openai import OpenAIModel

CommonOllamaModelNames = Literal[
    'codellama',
    'deepseek-r1',
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
    'phi4',
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

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        check_allow_model_requests()
        return await self.openai_model.agent_model(
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )

    def name(self) -> str:
        return f'ollama:{self.model_name}'
