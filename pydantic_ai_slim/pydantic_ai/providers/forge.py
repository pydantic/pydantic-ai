from __future__ import annotations as _annotations

import os

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Forge provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class ForgeProvider(Provider[AsyncOpenAI]):
    """Provider for [Forge](https://github.com/TensorBlock/forge), an OpenAI-compatible LLM router.

    Forge provides unified access to 40+ AI providers through a single API.
    Model names use the format `Provider/model-name` (e.g., `OpenAI/gpt-4o`).
    """

    @property
    def name(self) -> str:
        return 'forge'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        """Get model profile based on the provider prefix in the model name.

        Forge model names use the format `Provider/model-name` (e.g., `OpenAI/gpt-4o`).
        The provider prefix is used to delegate to the appropriate profile function.
        """
        provider_to_profile = {
            'openai': openai_model_profile,
            'anthropic': anthropic_model_profile,
            'google': google_model_profile,
            'mistralai': mistral_model_profile,
            'meta-llama': meta_model_profile,
            'deepseek': deepseek_model_profile,
            'x-ai': grok_model_profile,
            'qwen': qwen_model_profile,
        }

        profile = None
        if '/' in model_name:
            provider, model = model_name.split('/', 1)
            provider_lower = provider.lower()
            if provider_lower in provider_to_profile:
                profile = provider_to_profile[provider_lower](model)

        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Forge provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `FORGE_API_KEY` environment variable
                will be used if available.
            base_url: The base URL for the Forge API. Defaults to `https://api.forge.tensorblock.co/v1`.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key` and `http_client` must
                be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
        else:
            api_key = api_key or os.getenv('FORGE_API_KEY')
            if not api_key:
                raise UserError(
                    'Set the `FORGE_API_KEY` environment variable or pass it via `ForgeProvider(api_key=...)` '
                    'to use the Forge provider.'
                )

            self._base_url = base_url or os.getenv('FORGE_API_BASE', 'https://api.forge.tensorblock.co/v1')

            http_client = http_client or cached_async_http_client(provider='forge')
            self._client = AsyncOpenAI(base_url=self._base_url, api_key=api_key, http_client=http_client)
