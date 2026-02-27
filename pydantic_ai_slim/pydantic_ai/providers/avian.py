from __future__ import annotations as _annotations

import os
from typing import overload

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Avian provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class AvianProvider(Provider[AsyncOpenAI]):
    """Provider for [Avian API](https://avian.io).

    Avian provides an OpenAI-compatible API for accessing a curated set of
    frontier models including DeepSeek, Kimi, GLM, and MiniMax.
    """

    @property
    def name(self) -> str:
        return 'avian'

    @property
    def base_url(self) -> str:
        return 'https://api.avian.io/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            supports_json_object_output=True,
        )

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
        api_key = api_key or os.getenv('AVIAN_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `AVIAN_API_KEY` environment variable or pass it via `AvianProvider(api_key=...)` '
                'to use the Avian provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='avian')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
