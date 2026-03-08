from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the GitHub Models provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class GitHubProvider(Provider[AsyncOpenAI]):
    """Provider for GitHub Models API.

    GitHub Models provides access to various AI models through an OpenAI-compatible API.
    See <https://docs.github.com/en/github-models> for more information.
    """

    @property
    def name(self) -> str:
        return 'github'

    @property
    def base_url(self) -> str:
        return 'https://models.github.ai/inference'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'xai': grok_model_profile,
            'meta': meta_model_profile,
            'microsoft': openai_model_profile,
            'mistral-ai': mistral_model_profile,
            'cohere': cohere_model_profile,
            'deepseek': deepseek_model_profile,
        }

        profile = None

        # If the model name does not contain a provider prefix, we assume it's an OpenAI model
        if '/' not in model_name:
            return openai_model_profile(model_name)

        provider, model_name = model_name.lower().split('/', 1)
        if provider in provider_to_profile:
            model_name, *_ = model_name.split(':', 1)  # drop tags
            profile = provider_to_profile[provider](model_name)

        # As GitHubProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
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
        """Create a new GitHub Models provider.

        Args:
            api_key: The GitHub token to use for authentication. If not provided, the `GITHUB_API_KEY`
                environment variable will be used if available.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('GITHUB_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `GITHUB_API_KEY` environment variable or pass it via `GitHubProvider(api_key=...)`'
                ' to use the GitHub Models provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='github')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
