from __future__ import annotations as _annotations

import os
from typing import Literal, overload

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import (
    OpenAIJsonSchemaTransformer,
    OpenAIModelProfile,
)
from pydantic_ai.providers import Provider

MoonshotAIModelName = Literal[
    'moonshot-v1-8k',
    'moonshot-v1-32k',
    'moonshot-v1-128k',
    'moonshot-v1-8k-vision-preview',
    'moonshot-v1-32k-vision-preview',
    'moonshot-v1-128k-vision-preview',
    'kimi-latest',
    'kimi-thinking-preview',
    'kimi-k2-0711-preview',
]


class MoonshotAIProvider(Provider[AsyncOpenAI]):
    """Provider for MoonshotAI platform (Kimi models)."""

    @property
    def name(self) -> str:
        return 'moonshotai'

    @property
    def base_url(self) -> str:
        # OpenAI-compatible endpoint, see MoonshotAI docs
        return 'https://api.moonshot.ai/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        profile = moonshotai_model_profile(model_name)

        # As the MoonshotAI API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer,
        # unless json_schema_transformer is set explicitly.
        # Also, MoonshotAI does not support strict tool definitions
        # https://platform.moonshot.ai/docs/guide/migrating-from-openai-to-kimi#about-tool_choice
        # "Please note that the current version of Kimi API does not support the tool_choice=required parameter."
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_tool_choice_required=False,
            supports_json_object_output=True,
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
        api_key = api_key or os.getenv('MOONSHOTAI_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `MOONSHOTAI_API_KEY` environment variable or pass it via '
                '`MoonshotAIProvider(api_key=...)` to use the MoonshotAI provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='moonshotai')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
