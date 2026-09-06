from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the AI Token King provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )


class AITokenKingProvider(_OpenAICompatibleProvider):
    """Provider for the AI Token King gateway."""

    @property
    def name(self) -> str:
        return 'aitokenking'

    @property
    def base_url(self) -> str:
        return 'https://api.aitokenking.com.tw/api/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        model_name = model_name.lower()

        prefix_to_profile = {
            'claude-': anthropic_model_profile,
            'deepseek-': deepseek_model_profile,
            'gemini-': google_model_profile,
            'glm-': zai_model_profile,
            'gpt-': openai_model_profile,
            'kimi-': moonshotai_model_profile,
            'qwen': qwen_model_profile,
        }

        profile = None
        for prefix, profile_func in prefix_to_profile.items():
            if model_name.startswith(prefix):
                profile = profile_func(model_name)
                break

        # Model families the gateway serves that have no upstream profile in Pydantic AI (for example
        # MiniMax and Seed) fall through with `profile is None` and get the OpenAI-compatible defaults.
        # As `AITokenKingProvider` is always used with `OpenAIChatModel`, which used to unconditionally use
        # `OpenAIJsonSchemaTransformer`, we keep that as the base and let the family profile win on top.
        return merge_profile(OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer), profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: _OpenAIHTTPClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new AI Token King provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `AITOKENKING_API_KEY` environment
                variable will be used if available.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('AITOKENKING_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `AITOKENKING_API_KEY` environment variable or pass it via '
                '`AITokenKingProvider(api_key=...)` to use the AI Token King provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
