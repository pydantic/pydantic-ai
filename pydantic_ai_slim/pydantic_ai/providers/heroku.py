from __future__ import annotations as _annotations

import os
from typing import overload

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Heroku provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class HerokuProvider(Provider[AsyncOpenAI]):
    """Provider for Heroku API."""

    @property
    def name(self) -> str:
        return 'heroku'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        # As the Heroku API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer.
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer)

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
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if openai_client is not None:
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        else:
            api_key = api_key or os.getenv('HEROKU_INFERENCE_KEY')
            if not api_key:
                raise UserError(
                    'Set the `HEROKU_INFERENCE_KEY` environment variable or pass it via `HerokuProvider(api_key=...)`'
                    'to use the Heroku provider.'
                )

            base_url = base_url or os.getenv('HEROKU_INFERENCE_URL', 'https://us.inference.heroku.com')
            base_url = base_url.rstrip('/') + '/v1'

            if http_client is not None:
                self._client = AsyncOpenAI(api_key=api_key, http_client=http_client, base_url=base_url)
            else:
                http_client = cached_async_http_client(provider='heroku')
                self._client = AsyncOpenAI(api_key=api_key, http_client=http_client, base_url=base_url)
