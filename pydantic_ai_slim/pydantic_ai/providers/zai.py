from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.zai import zai_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Z.AI provider, '
        'you can use the `zai` optional group — `pip install "pydantic-ai-slim[zai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )


class ZaiProvider(_OpenAICompatibleProvider):
    """Provider for Z.AI (Zhipu AI) API.

    Z.AI provides GLM models with support for thinking/reasoning mode
    and preserved thinking across turns.
    """

    @property
    def name(self) -> str:
        return 'zai'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        profile = zai_model_profile(model_name)

        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
            OpenAIModelProfile(
                supports_json_object_output=True,
                openai_chat_thinking_field='reasoning_content',
                openai_chat_send_back_thinking_parts='field',
            ),
        )

    @overload
    def __init__(self, *, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str, base_url: str | None = None, http_client: _OpenAIHTTPClient) -> None: ...

    @overload
    def __init__(self, *, base_url: str | None = None, http_client: _OpenAIHTTPClient) -> None: ...

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
        """Create a new Z.AI provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `ZAI_API_KEY` environment variable
                will be used if available.
            base_url: The base URL to use for requests, if not provided, the general API endpoint
                `https://api.z.ai/api/paas/v4` will be used. Set this to use a different Z.AI endpoint,
                such as `https://api.z.ai/api/coding/paas/v4` for the GLM Coding Plan.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key`, `base_url` and
                `http_client` must be `None`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('ZAI_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `ZAI_API_KEY` environment variable or pass it via `ZaiProvider(api_key=...)` '
                'to use the Z.AI provider.'
            )

        self._base_url = base_url or 'https://api.z.ai/api/paas/v4'

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self._base_url, api_key=api_key, http_client=http_client)
