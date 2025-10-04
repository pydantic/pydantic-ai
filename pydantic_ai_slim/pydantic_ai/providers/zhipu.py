from __future__ import annotations as _annotations

import os
from typing import Literal, overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.zhipu import zhipu_model_profile
from pydantic_ai.providers import Provider

ZhipuModelName = Literal[
    'glm-4.5',
    'glm-4.5-air',
    'glm-4.5-flash',
    'glm-4.5v',
    'glm-4.6',
    'glm-4v',
    'glm-4v-plus',
    'codegeex-4',
]

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Zhipu provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class ZhipuProvider(Provider[AsyncOpenAI]):
    """Provider for Zhipu AI API.

    Zhipu AI provides an OpenAI-compatible API, so this provider uses the OpenAI client
    with Zhipu-specific configuration.
    """

    @property
    def name(self) -> str:
        return 'zhipu'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return zhipu_model_profile(model_name)

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Zhipu AI provider.

        Args:
            api_key: The API key to use for authentication. If not provided, the `ZHIPU_API_KEY` environment variable
                will be used if available.
            base_url: The base url for the Zhipu AI requests. If not provided, defaults to Zhipu's base url.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key`, `base_url`,
                and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if openai_client is not None:
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            self._client = openai_client
        else:
            api_key = api_key or os.getenv('ZHIPU_API_KEY')
            base_url = base_url or 'https://open.bigmodel.cn/api/paas/v4/'

            if not api_key:
                raise UserError(
                    'Set the `ZHIPU_API_KEY` environment variable or pass it via `ZhipuProvider(api_key=...)`'
                    ' to use the Zhipu provider.'
                )

            if http_client is not None:
                self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
            else:
                http_client = cached_async_http_client(provider='zhipu')
                self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
