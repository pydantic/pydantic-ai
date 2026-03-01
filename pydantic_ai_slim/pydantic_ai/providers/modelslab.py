from __future__ import annotations

import os

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the ModelsLab provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = ['ModelsLabProvider']


class ModelsLabProvider(Provider[AsyncOpenAI]):
    """Provider for ModelsLab's uncensored chat API.

    ModelsLab (https://modelslab.com) provides access to open-source LLMs
    including Llama, Mistral, DeepSeek, and community models via an
    OpenAI-compatible endpoint.

    API docs: https://docs.modelslab.com
    API key: https://modelslab.com/account/api-key
    """

    @property
    def name(self) -> str:
        return 'modelslab'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        """Get model profile for ModelsLab models.

        ModelsLab hosts models from multiple families. Profiles are matched
        based on model name prefixes.
        """
        model_name_lower = model_name.lower()

        if 'llama' in model_name_lower or 'meta-llama' in model_name_lower:
            profile = meta_model_profile(model_name)
        elif 'deepseek' in model_name_lower:
            profile = deepseek_model_profile(model_name)
        elif 'mistral' in model_name_lower or 'mixtral' in model_name_lower:
            profile = mistral_model_profile(model_name)
        else:
            profile = None

        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize ModelsLab provider.

        Args:
            api_key: ModelsLab API key. If not provided, reads from MODELSLAB_API_KEY env var.
            base_url: Custom API base URL. Defaults to https://modelslab.com/api/uncensored-chat/v1
            openai_client: Optional pre-configured AsyncOpenAI client.
            http_client: Optional custom httpx.AsyncClient for making HTTP requests.

        Raises:
            UserError: If API key is not provided and MODELSLAB_API_KEY env var is not set.
        """
        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
        else:
            api_key = api_key or os.getenv('MODELSLAB_API_KEY')
            if not api_key:
                raise UserError(
                    'Set the `MODELSLAB_API_KEY` environment variable or pass it via '
                    '`ModelsLabProvider(api_key=...)` to use the ModelsLab provider.'
                )

            self._base_url = base_url or os.getenv(
                'MODELSLAB_BASE_URL',
                'https://modelslab.com/api/uncensored-chat/v1',
            )

            http_client = http_client or cached_async_http_client(provider='modelslab')
            self._client = AsyncOpenAI(base_url=self._base_url, api_key=api_key, http_client=http_client)
