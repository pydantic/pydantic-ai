from __future__ import annotations as _annotations

import os
from typing import overload

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the DeepSeek provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class DeepSeekProvider(Provider[AsyncOpenAI]):
    """Provider for DeepSeek API."""

    @property
    def name(self) -> str:
        return 'deepseek'

    @property
    def base_url(self) -> str:
        return 'https://api.deepseek.com'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        profile = deepseek_model_profile(model_name)

        # As DeepSeekProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly.
        # This was not the case when using a DeepSeek model with another model class (e.g. BedrockConverseModel or GroqModel),
        # so we won't do this in `deepseek_model_profile` unless we learn it's always needed.
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `DEEPSEEK_API_KEY` environment variable or pass it via `DeepSeekProvider(api_key=...)`'
                'to use the DeepSeek provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='deepseek')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
