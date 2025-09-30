from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Cerebras provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class CerebrasProvider(Provider[AsyncOpenAI]):
    """Provider for Cerebras API."""

    @property
    def name(self) -> str:
        return 'cerebras'

    @property
    def base_url(self) -> str:
        return 'https://api.cerebras.ai/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        prefix_to_profile = {'llama': meta_model_profile, 'qwen': qwen_model_profile, 'gpt-oss': harmony_model_profile}

        profile = None
        for prefix, profile_func in prefix_to_profile.items():
            model_name = model_name.lower()
            if model_name.startswith(prefix):
                profile = profile_func(model_name)

        # According to https://inference-docs.cerebras.ai/resources/openai#currently-unsupported-openai-features,
        # Cerebras doesn't support some model settings.
        unsupported_model_settings = (
            'frequency_penalty',
            'logit_bias',
            'presence_penalty',
            'parallel_tool_calls',
            'service_tier',
        )
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_unsupported_model_settings=unsupported_model_settings,
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
        api_key = api_key or os.getenv('CEREBRAS_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `CEREBRAS_API_KEY` environment variable or pass it via `CerebrasProvider(api_key=...)` '
                'to use the Cerebras provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='cerebras')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
