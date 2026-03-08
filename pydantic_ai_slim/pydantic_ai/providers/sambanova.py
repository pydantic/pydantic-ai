from __future__ import annotations

import os

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the SambaNova provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = ['SambaNovaProvider']


class SambaNovaProvider(Provider[AsyncOpenAI]):
    """Provider for SambaNova AI models.

    SambaNova uses an OpenAI-compatible API.
    """

    @property
    def name(self) -> str:
        """Return the provider name."""
        return 'sambanova'

    @property
    def base_url(self) -> str:
        """Return the base URL."""
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        """Return the AsyncOpenAI client."""
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        """Get model profile for SambaNova models.

        SambaNova serves models from multiple families including Meta Llama,
        DeepSeek, Qwen, and Mistral. Model profiles are matched based on
        model name prefixes.
        """
        prefix_to_profile = {
            'deepseek-': deepseek_model_profile,
            'meta-llama-': meta_model_profile,
            'llama-': meta_model_profile,
            'qwen': qwen_model_profile,
            'mistral': mistral_model_profile,
        }

        profile = None
        model_name_lower = model_name.lower()

        for prefix, profile_func in prefix_to_profile.items():
            if model_name_lower.startswith(prefix):
                profile = profile_func(model_name)
                break

        # Wrap into OpenAIModelProfile since SambaNova is OpenAI-compatible
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize SambaNova provider.

        Args:
            api_key: SambaNova API key. If not provided, reads from SAMBANOVA_API_KEY env var.
            base_url: Custom API base URL. Defaults to https://api.sambanova.ai/v1
            openai_client: Optional pre-configured OpenAI client
            http_client: Optional custom httpx.AsyncClient for making HTTP requests

        Raises:
            UserError: If API key is not provided and SAMBANOVA_API_KEY env var is not set
        """
        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
        else:
            # Get API key from parameter or environment
            api_key = api_key or os.getenv('SAMBANOVA_API_KEY')
            if not api_key:
                raise UserError(
                    'Set the `SAMBANOVA_API_KEY` environment variable or pass it via '
                    '`SambaNovaProvider(api_key=...)` to use the SambaNova provider.'
                )

            # Set base URL (default to SambaNova API endpoint)
            self._base_url = base_url or os.getenv('SAMBANOVA_BASE_URL', 'https://api.sambanova.ai/v1')

            # Create http client and AsyncOpenAI client
            http_client = http_client or cached_async_http_client(provider='sambanova')
            self._client = AsyncOpenAI(base_url=self._base_url, api_key=api_key, http_client=http_client)
