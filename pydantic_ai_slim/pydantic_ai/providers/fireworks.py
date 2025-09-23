from __future__ import annotations as _annotations

import os
from typing import overload

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Fireworks AI provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class FireworksProvider(Provider[AsyncOpenAI]):
    """Provider for Fireworks AI API."""

    @property
    def name(self) -> str:
        return 'fireworks'

    @property
    def base_url(self) -> str:
        return 'https://api.fireworks.ai/inference/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        prefix_to_profile = {
            'llama': meta_model_profile,
            'qwen': qwen_model_profile,
            'deepseek': deepseek_model_profile,
            'mistral': mistral_model_profile,
            'gemma': google_model_profile,
        }

        prefix = 'accounts/fireworks/models/'

        profile = None
        if model_name.startswith(prefix):
            model_name = model_name[len(prefix) :]
            for provider, profile_func in prefix_to_profile.items():
                if model_name.startswith(provider):
                    profile = profile_func(model_name)
                    break

        # As the Fireworks API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer,
        # unless json_schema_transformer is set explicitly
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
        api_key = api_key or os.getenv('FIREWORKS_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `FIREWORKS_API_KEY` environment variable or pass it via `FireworksProvider(api_key=...)`'
                'to use the Fireworks AI provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='fireworks')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
