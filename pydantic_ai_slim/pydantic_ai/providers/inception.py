from __future__ import annotations as _annotations

import os
from typing import Literal, overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.inception import inception_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Inception provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


InceptionModelName = Literal['mercury-2']


class InceptionProvider(Provider[AsyncOpenAI]):
    """Provider for Inception Labs API."""

    @property
    def name(self) -> str:
        return 'inception'

    @property
    def base_url(self) -> str:
        return 'https://api.inceptionlabs.ai/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        model_name = model_name.lower()

        profile = inception_model_profile(model_name) if model_name.startswith('mercury') else None

        # As the Inception API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer.
        # The API only honors `max_tokens`: `max_completion_tokens` is silently ignored.
        return merge_profile(
            OpenAIModelProfile(
                json_schema_transformer=OpenAIJsonSchemaTransformer,
                openai_chat_supports_max_completion_tokens=False,
            ),
            profile,
        )

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Inception provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `INCEPTION_API_KEY` environment variable
                will be used if available.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('INCEPTION_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `INCEPTION_API_KEY` environment variable or pass it via `InceptionProvider(api_key=...)` '
                'to use the Inception provider.'
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
