from __future__ import annotations as _annotations

import os
from dataclasses import dataclass
from typing import Literal

from httpx import AsyncClient as AsyncHTTPClient

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        "you can use the `openai` optional group — `pip install 'pydantic-ai-slim[openai]'`"
    ) from e

from openai import OpenAIError

from ..tools import ToolDefinition
from . import AgentModel, Model
from .openai import OpenAIModel

DeepSeekModelName = Literal['deepseek-chat']
"""
DeepSeek follows the OpenAI chat completions API. 
For details see [DeepSeek API documentation](https://api-docs.deepseek.com).
"""


@dataclass(init=False)
class DeepSeekModel(Model):
    """A model that implements DeepSeek using the OpenAI API.

    Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact with the DeepSeek API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    model_name: DeepSeekModelName
    openai_model: OpenAIModel

    def __init__(
        self,
        model_name: DeepSeekModelName,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        """Initialize a DeepSeek model.

        DeepSeek has built-in compatibility with the OpenAI chat completions API, so we reuse the
        [`OpenAIModel`][pydantic_ai.models.openai.OpenAIModel] here.

        Args:
            model_name: The name of the DeepSeek model to use. List of models available in the
                [DeepSeek API documentation](https://platform.deepseek.com/docs)
            api_key: The API key to use for authentication. If not provided, the `DEEPSEEK_API_KEY` environment variable
                will be used if available. This parameter is ignored if `openai_client` is provided.
            openai_client: An existing [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
                client to use. If provided, it must be configured with the DeepSeek base URL
                (https://api.deepseek.com/v1) and a valid API key.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        self.model_name = model_name
        api_key = api_key or os.environ.get('DEEPSEEK_API_KEY')

        if openai_client is not None:
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            self.openai_model = OpenAIModel(
                model_name=model_name,
                openai_client=openai_client,
            )
        elif http_client is not None:
            assert openai_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is not None, 'Must provide `api_key` when using `http_client`'
            client = AsyncOpenAI(
                base_url='https://api.deepseek.com/v1',
                api_key=api_key,
                http_client=http_client,
            )
            self.openai_model = OpenAIModel(
                model_name=model_name,
                openai_client=client,
            )
        else:
            if api_key is None:
                raise OpenAIError(
                    'DeepSeek API key not found. Please provide it either through the api_key parameter '
                    'or set it as the DEEPSEEK_API_KEY environment variable.'
                )
            self.openai_model = OpenAIModel(
                model_name=model_name,
                base_url='https://api.deepseek.com/v1',
                api_key=api_key,
            )

    def name(self) -> str:
        return str(self.model_name)

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        return await self.openai_model.agent_model(
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )
