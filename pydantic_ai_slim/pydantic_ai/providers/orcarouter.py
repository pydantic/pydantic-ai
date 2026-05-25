from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the OrcaRouter provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class OrcaRouterProvider(Provider[AsyncOpenAI]):
    """Provider for OrcaRouter API."""

    @property
    def name(self) -> str:
        return 'orcarouter'

    @property
    def base_url(self) -> str:
        return 'https://api.orcarouter.ai/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'anthropic': anthropic_model_profile,
            'deepseek': deepseek_model_profile,
            'google': google_model_profile,
            'meta-llama': meta_model_profile,
            'mistral': mistral_model_profile,
            'mistralai': mistral_model_profile,
            'moonshotai': moonshotai_model_profile,
            'openai': openai_model_profile,
            'qwen': qwen_model_profile,
            'x-ai': grok_model_profile,
            'xai': grok_model_profile,
        }

        profile = None

        if '/' in model_name:
            vendor, upstream = model_name.split('/', 1)
            if vendor in provider_to_profile:
                profile = provider_to_profile[vendor](upstream)

        # As OrcaRouterProvider is always used with OpenAIChatModel, the JSON schema is funneled
        # through OpenAIJsonSchemaTransformer regardless of the underlying upstream provider.
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
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
        api_key = api_key or os.getenv('ORCAROUTER_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `ORCAROUTER_API_KEY` environment variable or pass it via `OrcaRouterProvider(api_key=...)`'
                ' to use the OrcaRouter provider.'
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
