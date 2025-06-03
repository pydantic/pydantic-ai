from __future__ import annotations as _annotations

import os
from typing import overload

from httpx import AsyncClient as AsyncHTTPClient
from openai import AsyncOpenAI

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.cerebras import CerebrasJsonSchemaTransformer
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile
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
        provider_to_profile = {
            'deepseek': deepseek_model_profile,
            'qwen': qwen_model_profile,
            'llama': meta_model_profile,
        }
        profile = None

        try:
            model_provider = model_name.split('-')[0]
            for provider, profile_func in provider_to_profile.items():
                if model_provider.startswith(provider):
                    profile = profile_func(model_name)
                    break
        except Exception as _:  # pragma: no cover
            pass

        return OpenAIModelProfile(
            json_schema_transformer=CerebrasJsonSchemaTransformer, openai_supports_strict_tool_definition=True
        ).update(profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: AsyncHTTPClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('CEREBRAS_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `CEREBRAS_API_KEY` environment variable or pass it via `CerebrasProvider(api_key=...)`'
                'to use the Cerebras provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='cerebras')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
