from __future__ import annotations as _annotations

from typing import overload

from httpx import AsyncClient as AsyncHTTPClient
from openai import AsyncOpenAI

from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.providers import Provider
from pydantic_ai_slim.pydantic_ai.models import cached_async_http_client

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
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        # For LiteLLM, we use a basic OpenAI profile since it's OpenAI-compatible
        # Users can override this with their own profile if needed
        profile = openai_model_profile(model_name)

        # As LiteLLMProvider is used with OpenAIModel, which used to use OpenAIJsonSchemaTransformer,
        # we maintain that behavior
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        custom_llm_provider: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        custom_llm_provider: str | None = None,
        http_client: AsyncHTTPClient,
    ) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        custom_llm_provider: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        """Initialize a LiteLLM provider.

        Args:
            api_key: API key for the model provider. If None, LiteLLM will try to get it from environment variables.
            api_base: Base URL for the model provider. Use this for custom endpoints or self-hosted models.
            custom_llm_provider: Custom LLM provider name for LiteLLM. Use this if LiteLLM can't auto-detect the provider.
            openai_client: Pre-configured OpenAI client. If provided, other parameters are ignored.
            http_client: Custom HTTP client to use.
        """
        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
            return

        # Set up LiteLLM configuration
        if api_key:
            # Store API key in LiteLLM's global config if needed
            # LiteLLM will handle provider-specific API key names
            pass

        if custom_llm_provider:
            # LiteLLM can auto-detect most providers, but this allows override
            pass

        # Use api_base if provided, otherwise use a generic base URL
        # LiteLLM doesn't actually use this URL - it routes internally
        self._base_url = api_base or 'https://api.litellm.ai/v1'

        # Create OpenAI client that will be used with LiteLLM's completion function
        # The actual API calls will be intercepted and routed through LiteLLM
        if http_client is not None:
            self._client = AsyncOpenAI(
                base_url=self._base_url, api_key=api_key or 'litellm-placeholder', http_client=http_client
            )
        else:
            http_client = cached_async_http_client(provider='litellm')
            self._client = AsyncOpenAI(
                base_url=self._base_url, api_key=api_key or 'litellm-placeholder', http_client=http_client
            )

        # Store configuration for LiteLLM
        self._api_key = api_key
        self._api_base = api_base
        self._custom_llm_provider = custom_llm_provider
