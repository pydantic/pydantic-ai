from __future__ import annotations as _annotations

import os
from typing import Literal, overload

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles import merge_profile
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
    'moonshot-v1-auto',
    'kimi-latest',
    'kimi-thinking-preview',
    'kimi-k2-0711-preview',
    'kimi-k2.5',
    'kimi-k2.6',
    'kimi-k2.7-code',
    'kimi-k2.7-code-highspeed',
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

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        profile = moonshotai_model_profile(model_name)

        # `api.moonshot.ai` rejects `reasoning_effort='none'` (it accepts minimal/low/medium/high),
        # and reasoning can't be turned off through the unified `thinking` setting (the native off
        # switch is a `thinking={'type': 'disabled'}` body object we don't send). Mark reasoning as
        # always-enabled so `thinking=False` omits `reasoning_effort` rather than sending the rejected
        # `'none'`. This is set here, not in `moonshotai_model_profile`, because that profile is also
        # routed through OpenRouter/Heroku, whose gateways don't share this endpoint quirk.
        is_reasoning = bool(profile and profile.get('supports_thinking'))

        # As the MoonshotAI API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer,
        # unless json_schema_transformer is set explicitly.
        # Also, MoonshotAI does not support strict tool definitions
        # https://platform.moonshot.ai/docs/guide/migrating-from-openai-to-kimi#about-tool_choice
        # "Please note that the current version of Kimi API does not support the tool_choice=required parameter."
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
            OpenAIModelProfile(
                openai_supports_tool_choice_required=False,
                supports_json_object_output=True,
                openai_chat_thinking_field='reasoning_content',
                openai_chat_send_back_thinking_parts='field',
                thinking_always_enabled=is_reasoning,
            ),
        )

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
            http_client = create_async_http_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_http_client
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
