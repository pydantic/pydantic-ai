from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Vercel provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class VercelProvider(Provider[AsyncOpenAI]):
    """Provider for Vercel AI Gateway API."""

    @property
    def name(self) -> str:
        return 'vercel'

    @property
    def base_url(self) -> str:
        return 'https://ai-gateway.vercel.sh/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'anthropic': anthropic_model_profile,
            'bedrock': amazon_model_profile,
            'cohere': cohere_model_profile,
            'deepseek': deepseek_model_profile,
            'mistral': mistral_model_profile,
            'openai': openai_model_profile,
            'vertex': google_model_profile,
            'xai': grok_model_profile,
        }

        profile = None

        try:
            provider, model_name = model_name.split('/', 1)
        except ValueError:
            raise UserError(f"Model name must be in 'provider/model' format, got: {model_name!r}")

        if provider in provider_to_profile:
            profile = provider_to_profile[provider](model_name)

        # As VercelProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
        ).update(profile)

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
        # Support Vercel AI Gateway's standard environment variables
        api_key = api_key or os.getenv('VERCEL_AI_GATEWAY_API_KEY') or os.getenv('VERCEL_OIDC_TOKEN')

        if not api_key and openai_client is None:
            raise UserError(
                'Set the `VERCEL_AI_GATEWAY_API_KEY` or `VERCEL_OIDC_TOKEN` environment variable '
                'or pass the API key via `VercelProvider(api_key=...)` to use the Vercel provider.'
            )

        default_headers = {'http-referer': 'https://ai.pydantic.dev/', 'x-title': 'pydantic-ai'}

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
            )
        else:
            http_client = cached_async_http_client(provider='vercel')
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
            )
