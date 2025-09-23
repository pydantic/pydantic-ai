from __future__ import annotations as _annotations

from typing import overload

from httpx import AsyncClient as AsyncHTTPClient
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.groq import groq_model_profile
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
        'Please install the `openai` package to use the LiteLLM provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class LiteLLMProvider(Provider[AsyncOpenAI]):
    """Provider for LiteLLM API."""

    @property
    def name(self) -> str:
        return 'litellm'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        # Map provider prefixes to their profile functions
        provider_to_profile = {
            'anthropic': anthropic_model_profile,
            'openai': openai_model_profile,
            'google': google_model_profile,
            'mistralai': mistral_model_profile,
            'mistral': mistral_model_profile,
            'cohere': cohere_model_profile,
            'amazon': amazon_model_profile,
            'bedrock': amazon_model_profile,
            'meta-llama': meta_model_profile,
            'meta': meta_model_profile,
            'groq': groq_model_profile,
            'deepseek': deepseek_model_profile,
            'moonshotai': moonshotai_model_profile,
            'x-ai': grok_model_profile,
            'qwen': qwen_model_profile,
        }

        profile = None

        # Check if model name contains a provider prefix (e.g., "anthropic/claude-3")
        if '/' in model_name:
            provider_prefix, model_suffix = model_name.split('/', 1)
            if provider_prefix in provider_to_profile:
                profile = provider_to_profile[provider_prefix](model_suffix)

        # If no profile found, default to OpenAI profile
        if profile is None:
            profile = openai_model_profile(model_name)

        # As LiteLLMProvider is used with OpenAIModel, which uses OpenAIJsonSchemaTransformer,
        # we maintain that behavior
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        http_client: AsyncHTTPClient,
    ) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        """Initialize a LiteLLM provider.

        Args:
            api_key: API key for the model provider. If None, LiteLLM will try to get it from environment variables.
            api_base: Base URL for the model provider. Use this for custom endpoints or self-hosted models.
            openai_client: Pre-configured OpenAI client. If provided, other parameters are ignored.
            http_client: Custom HTTP client to use.
        """
        if openai_client is not None:
            self._client = openai_client
            return

        # Create OpenAI client that will be used with LiteLLM's completion function
        # The actual API calls will be intercepted and routed through LiteLLM
        if http_client is not None:
            self._client = AsyncOpenAI(
                base_url=api_base, api_key=api_key or 'litellm-placeholder', http_client=http_client
            )
        else:
            http_client = cached_async_http_client(provider='litellm')
            self._client = AsyncOpenAI(
                base_url=api_base, api_key=api_key or 'litellm-placeholder', http_client=http_client
            )
