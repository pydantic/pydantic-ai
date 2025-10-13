from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Nebius provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class NebiusProvider(Provider[AsyncOpenAI]):
    """Provider for Nebius AI Studio API."""

    @property
    def name(self) -> str:
        return 'nebius'

    @property
    def base_url(self) -> str:
        return 'https://api.studio.nebius.com/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'meta-llama': meta_model_profile,
            'deepseek-ai': deepseek_model_profile,
            'qwen': qwen_model_profile,
            'google': google_model_profile,
            'openai': harmony_model_profile,  # used for gpt-oss models on Nebius
            'mistralai': mistral_model_profile,
            'moonshotai': moonshotai_model_profile,
        }

        profile = None

        try:
            model_name = model_name.lower()
            provider, model_name = model_name.split('/', 1)
        except ValueError:
            raise UserError(f"Model name must be in 'provider/model' format, got: {model_name!r}")
        if provider in provider_to_profile:
            profile = provider_to_profile[provider](model_name)

        # As NebiusProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly
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
        api_key = api_key or os.getenv('NEBIUS_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `NEBIUS_API_KEY` environment variable or pass it via '
                '`NebiusProvider(api_key=...)` to use the Nebius AI Studio provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='nebius')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
