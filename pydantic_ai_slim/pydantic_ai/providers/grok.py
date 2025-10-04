from __future__ import annotations as _annotations

import os
from typing import Literal, overload

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Grok provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

# https://docs.x.ai/docs/models
GrokModelName = Literal[
    'grok-4',
    'grok-4-0709',
    'grok-3',
    'grok-3-mini',
    'grok-3-fast',
    'grok-3-mini-fast',
    'grok-2-vision-1212',
    'grok-2-image-1212',
]


class GrokProvider(Provider[AsyncOpenAI]):
    """Provider for Grok API."""

    @property
    def name(self) -> str:
        return 'grok'

    @property
    def base_url(self) -> str:
        return 'https://api.x.ai/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        profile = grok_model_profile(model_name)

        # As the Grok API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer,
        # unless json_schema_transformer is set explicitly.
        # Also, Grok does not support strict tool definitions: https://github.com/pydantic/pydantic-ai/issues/1846
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer, openai_supports_strict_tool_definition=False
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
        api_key = api_key or os.getenv('GROK_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `GROK_API_KEY` environment variable or pass it via `GrokProvider(api_key=...)`'
                'to use the Grok provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='grok')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
