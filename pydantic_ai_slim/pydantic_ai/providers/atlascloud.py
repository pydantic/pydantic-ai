from __future__ import annotations as _annotations

import os
from typing import overload

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Atlas Cloud provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class AtlasCloudProvider(Provider[AsyncOpenAI]):
    """Provider for Atlas Cloud API."""

    @property
    def name(self) -> str:
        return 'atlascloud'

    @property
    def base_url(self) -> str:
        return 'https://api.atlascloud.ai/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        # Atlas Cloud serves many OpenAI-compatible models. DeepSeek models are the
        # main family, so reuse `deepseek_model_profile` for them and fall back to
        # `openai_model_profile` for everything else.
        if model_name.lower().startswith('deepseek'):
            profile = deepseek_model_profile(model_name)
        else:
            profile = openai_model_profile(model_name)

        # As the Atlas Cloud API is OpenAI-compatible, let's assume we also need
        # OpenAIJsonSchemaTransformer, unless json_schema_transformer is set explicitly.
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            supports_json_object_output=True,
        ).update(profile)

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('ATLASCLOUD_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `ATLASCLOUD_API_KEY` environment variable or pass it via `AtlasCloudProvider(api_key=...)`'
                ' to use the Atlas Cloud provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = create_async_http_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_http_client
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
