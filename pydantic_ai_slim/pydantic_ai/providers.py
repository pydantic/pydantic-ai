from __future__ import annotations as _annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from httpx import AsyncClient as AsyncHTTPClient
from openai import AsyncOpenAI

from pydantic_ai.models import cached_async_http_client

InterfaceClient = TypeVar('InterfaceClient')


class Provider(ABC, Generic[InterfaceClient]):
    """Abstract class for a provider.

    The provider is in charge of providing an authenticated client to the API.

    Each provider only supports a specific interface. A interface can be supported by multiple providers.

    For example, the OpenAIModel interface can be supported by the OpenAIProvider and the DeepSeekProvider.
    """

    _client: InterfaceClient

    @property
    @abstractmethod
    def provider(self) -> str:
        """The provider name."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def base_url(self) -> str:
        """The base URL for the provider API."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def client(self) -> InterfaceClient:
        """The client for the provider."""
        return self._client


class DeepSeekProvider(Provider[AsyncOpenAI]):
    """Provider for DeepSeek API."""

    @property
    def provider(self) -> str:
        return 'deepseek'

    @property
    def base_url(self) -> str:
        return 'https://api.deepsek.com/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def __init__(
        self,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        api_key = api_key or os.environ.get('DEEPSEEK_API_KEY')
        # This is a workaround for the OpenAI client requiring an API key, whilst locally served,
        # openai compatible models do not always need an API key, but a placeholder (non-empty) key is required.
        if api_key is None and openai_client is None:
            raise ValueError('API key is required for DeepSeek provider')

        if openai_client is not None:
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=cached_async_http_client())


class OpenAIProvider(Provider[AsyncOpenAI]):
    """Provider for OpenAI API."""

    @property
    def provider(self) -> str:
        return 'openai'

    @property
    def base_url(self) -> str:
        return 'https://api.openai.com/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def __init__(
        self,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        # This is a workaround for the OpenAI client requiring an API key, whilst locally served,
        # openai compatible models do not always need an API key, but a placeholder (non-empty) key is required.
        if api_key is None and 'OPENAI_API_KEY' not in os.environ and openai_client is None:
            api_key = 'api-key-not-set'

        if openai_client is not None:
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=cached_async_http_client())


def infer_provider(provider: str) -> Provider[Any]:
    """Infer the provider from the provider name."""
    if provider == 'openai':
        return OpenAIProvider()
    elif provider == 'deepseek':
        return DeepSeekProvider()
    else:
        raise ValueError(f'Unknown provider: {provider}')
