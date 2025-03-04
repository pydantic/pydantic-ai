"""Providers for the API clients.

The providers are in charge of providing an authenticated client to the API.
"""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

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
    def name(self) -> str:
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
        raise NotImplementedError()


def infer_provider(provider: str) -> Provider[Any]:
    """Infer the provider from the provider name."""
    if provider == 'openai':
        from .openai import OpenAIProvider

        return OpenAIProvider()
    elif provider == 'deepseek':
        from .deepseek import DeepSeekProvider

        return DeepSeekProvider()
    elif provider == 'google-vertex':
        from .google_vertex import GoogleVertexProvider

        return GoogleVertexProvider()
    elif provider == 'google-gla':
        from .google_gla import GoogleGLAProvider

        return GoogleGLAProvider()
    else:  # pragma: no cover
        raise ValueError(f'Unknown provider: {provider}')
