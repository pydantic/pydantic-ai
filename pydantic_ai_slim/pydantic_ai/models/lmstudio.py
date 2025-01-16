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

    from .openai import OpenAIModel
except ImportError as e:
    raise ImportError(
        'Please install `openai` to use the LMStudio, '
        "you can use the `openai` optional group — `pip install 'pydantic-ai-slim[openai]'`"
    ) from e


CommonLMStudioModelNames = Literal[
    'mistral-nemo-instruct-2407',
    'llama-3.2-3b-instruct',
]
"""This contains just the most common LMStudio models.

For a full list see [LMStudio search models](https://lmstudio.ai/models).
"""
LMStudioModelName = Union[CommonLMStudioModelNames, str]
"""Possible LMStudio model names or see this model's API identifier in LMStudio."""


@dataclass(init=False)
class LMStudioModel(Model):
    """A model that implements LMStudio using the OpenAI API.

    Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact with the LMStudio server.
    """

    model_name: LMStudioModelName
    openai_model: OpenAIModel

    def __init__(
        self,
        model_name: LMStudioModelName,
        *,
        base_url: str | None = 'http://127.0.0.1:1234/v1/',
        api_key: str = 'lmstudio',
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        """Initialize an LMStudio model.

        LMStudio has built-in compatability for the OpenAI chat completions API ([LMStudiosource](https://lmstudio.ai/docs/api/openai-api)), so we reuse the
        [`OpenAIModel`][pydantic_ai.models.openai.OpenAIModel] here.

        Args:
            model_name: The name of the LMStudio model to use. List of models available [here](https://lmstudio.ai/models)
                You must first download the model. [LMStudio](https://lmstudio.ai/) and [Docs](https://lmstudio.ai/docs).
            base_url: The base url for the LMStudio requests. The default value is the LMStudio default.
            api_key: The API key to use for authentication. Defaults to 'LMStudio' don't use it,
                but can be customized for proxy setups that require authentication.
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
            # API key is not required for LMStudio but a value is required to create the client
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
        return f'lmstudio:{self.model_name}'
