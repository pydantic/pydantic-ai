from __future__ import annotations as _annotations

import os
from typing import Literal, overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import (
    OpenAIJsonSchemaTransformer,
    OpenAIModelProfile,
)

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the MoonshotAI provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )

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
    'kimi-k3',
]


class MoonshotAIProvider(_OpenAICompatibleProvider):
    """Provider for MoonshotAI platform (Kimi models)."""

    @property
    def name(self) -> str:
        return 'moonshotai'

    @property
    def base_url(self) -> str:
        return self._base_url

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
                supports_forced_tool_choice=False,
                supports_json_object_output=True,
                openai_chat_thinking_field='reasoning_content',
                openai_chat_send_back_thinking_parts='field',
                thinking_always_enabled=is_reasoning,
            ),
        )

    @overload
    def __init__(self, *, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: _OpenAIHTTPClient, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new MoonshotAI provider.

        Args:
            api_key: The API key for authentication. Defaults to the `MOONSHOTAI_API_KEY`
                environment variable.
            base_url: The base URL for requests. Defaults to the `MOONSHOTAI_BASE_URL`
                environment variable, then `https://api.moonshot.ai/v1`.
                Use `https://api.moonshot.cn/v1` for the Chinese endpoint.
            openai_client: An existing `AsyncOpenAI` client to use. Its base URL is used
                instead of `MOONSHOTAI_BASE_URL`. Cannot be combined with `base_url`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('MOONSHOTAI_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `MOONSHOTAI_API_KEY` environment variable or pass it via '
                '`MoonshotAIProvider(api_key=...)` to use the MoonshotAI provider.'
            )

        if openai_client is not None and base_url is not None:
            raise UserError('Cannot provide both `openai_client` and `base_url`')

        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
        else:
            self._base_url = base_url or os.getenv('MOONSHOTAI_BASE_URL') or 'https://api.moonshot.ai/v1'
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
