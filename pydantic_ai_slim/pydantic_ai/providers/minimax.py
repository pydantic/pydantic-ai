from __future__ import annotations as _annotations

import os
from typing import overload

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.minimax import minimax_model_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the MiniMax provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class MiniMaxProvider(Provider[AsyncOpenAI]):
    """Provider for MiniMax API.

    MiniMax uses an OpenAI-compatible API. Note the following constraints:

    - `temperature` must be in the range `(0.0, 1.0]` — zero is not accepted.
    - `response_format` is not supported.
    """

    @property
    def name(self) -> str:
        return 'minimax'

    @property
    def base_url(self) -> str:
        return 'https://api.minimax.io/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return OpenAIModelProfile().update(minimax_model_profile(model_name))

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('MINIMAX_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `MINIMAX_API_KEY` environment variable or pass it via `MiniMaxProvider(api_key=...)`'
                'to use the MiniMax provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='minimax')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
