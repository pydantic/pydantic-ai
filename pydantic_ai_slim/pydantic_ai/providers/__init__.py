"""Providers for the API clients.

The providers are in charge of providing an authenticated client to the API.
"""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from asyncio import Lock
from collections.abc import Callable
from types import TracebackType
from typing import Any, Generic

import httpx
from typing_extensions import Self, TypeVar

from ..profiles import ModelProfile

InterfaceClient = TypeVar('InterfaceClient', default=Any)


class Provider(ABC, Generic[InterfaceClient]):
    """Abstract class for a provider.

    The provider is in charge of providing an authenticated client to the API.

    Each provider only supports a specific interface. An interface can be supported by multiple providers.

    For example, the `OpenAIChatModel` interface can be supported by the `OpenAIProvider` and the `DeepSeekProvider`.

    When used as an async context manager, providers that create their own HTTP client will close it on exit.
    This is handled automatically when using [`Agent`][pydantic_ai.agent.Agent] as a context manager.
    """

    _client: InterfaceClient
    _own_http_client: httpx.AsyncClient | None = None
    _http_client_factory: Callable[[], httpx.AsyncClient] | None = None
    _entered_count: int = 0
    _enter_lock: Lock | None = None

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

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        """The model profile for the named model, if available."""
        return None  # pragma: no cover

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        """Update the SDK client's internal HTTP client reference.

        Subclasses that manage their own HTTP client should override this to inject
        the new client into their SDK client after re-creation.
        """

    async def __aenter__(self) -> Self:
        if self._enter_lock is None:
            self._enter_lock = Lock()
        async with self._enter_lock:
            if self._entered_count == 0 and self._own_http_client is not None:
                if self._own_http_client.is_closed and self._http_client_factory is not None:
                    new_client = self._http_client_factory()
                    self._own_http_client = new_client
                    self._set_http_client(new_client)
            self._entered_count += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        if self._enter_lock is None:
            return
        async with self._enter_lock:
            self._entered_count -= 1
            if self._entered_count == 0 and self._own_http_client is not None:
                await self._own_http_client.aclose()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, base_url={self.base_url})'  # pragma: lax no cover


def infer_provider_class(provider: str) -> type[Provider[Any]]:  # noqa: C901
    """Infers the provider class from the provider name."""
    # Normalize gateway-prefixed providers (e.g. 'gateway/openai' -> 'openai')
    if provider.startswith('gateway/'):
        from .gateway import normalize_gateway_provider

        provider = normalize_gateway_provider(provider)

    # Normalize deprecated/alias provider names
    if provider == 'vertexai':
        provider = 'google-vertex'
    elif provider == 'google':
        provider = 'google-gla'

    if provider in ('openai', 'openai-chat', 'openai-responses'):
        from .openai import OpenAIProvider

        return OpenAIProvider
    elif provider == 'deepseek':
        from .deepseek import DeepSeekProvider

        return DeepSeekProvider
    elif provider == 'openrouter':
        from .openrouter import OpenRouterProvider

        return OpenRouterProvider
    elif provider == 'vercel':
        from .vercel import VercelProvider

        return VercelProvider
    elif provider == 'azure':
        from .azure import AzureProvider

        return AzureProvider
    elif provider in ('google-vertex', 'google-gla'):
        from .google import GoogleProvider

        return GoogleProvider
    elif provider == 'bedrock':
        from .bedrock import BedrockProvider

        return BedrockProvider
    elif provider == 'groq':
        from .groq import GroqProvider

        return GroqProvider
    elif provider == 'anthropic':
        from .anthropic import AnthropicProvider

        return AnthropicProvider
    elif provider == 'mistral':
        from .mistral import MistralProvider

        return MistralProvider
    elif provider == 'cerebras':
        from .cerebras import CerebrasProvider

        return CerebrasProvider
    elif provider == 'cohere':
        from .cohere import CohereProvider

        return CohereProvider
    elif provider == 'grok':
        from .grok import GrokProvider  # pyright: ignore[reportDeprecated]

        return GrokProvider  # pyright: ignore[reportDeprecated]
    elif provider == 'xai':
        from .xai import XaiProvider

        return XaiProvider
    elif provider == 'moonshotai':
        from .moonshotai import MoonshotAIProvider

        return MoonshotAIProvider
    elif provider == 'fireworks':
        from .fireworks import FireworksProvider

        return FireworksProvider
    elif provider == 'together':
        from .together import TogetherProvider

        return TogetherProvider
    elif provider == 'heroku':
        from .heroku import HerokuProvider

        return HerokuProvider
    elif provider == 'huggingface':
        from .huggingface import HuggingFaceProvider

        return HuggingFaceProvider
    elif provider == 'ollama':
        from .ollama import OllamaProvider

        return OllamaProvider
    elif provider == 'github':
        from .github import GitHubProvider

        return GitHubProvider
    elif provider == 'litellm':
        from .litellm import LiteLLMProvider

        return LiteLLMProvider
    elif provider == 'nebius':
        from .nebius import NebiusProvider

        return NebiusProvider
    elif provider == 'ovhcloud':
        from .ovhcloud import OVHcloudProvider

        return OVHcloudProvider
    elif provider == 'alibaba':
        from .alibaba import AlibabaProvider

        return AlibabaProvider
    elif provider == 'sambanova':
        from .sambanova import SambaNovaProvider

        return SambaNovaProvider
    elif provider == 'outlines':
        from .outlines import OutlinesProvider

        return OutlinesProvider
    elif provider == 'sentence-transformers':
        from .sentence_transformers import SentenceTransformersProvider

        return SentenceTransformersProvider
    elif provider == 'voyageai':
        from .voyageai import VoyageAIProvider

        return VoyageAIProvider
    else:
        raise ValueError(f'Unknown provider: {provider}')


def infer_provider(provider: str) -> Provider[Any]:
    """Infer the provider from the provider name."""
    if provider.startswith('gateway/'):
        from .gateway import gateway_provider

        upstream_provider = provider.removeprefix('gateway/')
        return gateway_provider(upstream_provider)
    elif provider in ('google-vertex', 'google-gla', 'vertexai'):
        from .google import GoogleProvider

        return GoogleProvider(vertexai=provider in ('google-vertex', 'vertexai'))
    else:
        provider_class = infer_provider_class(provider)
        return provider_class()
