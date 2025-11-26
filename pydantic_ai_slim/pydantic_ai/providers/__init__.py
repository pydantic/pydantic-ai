"""Providers for the API clients.

The providers are in charge of providing an authenticated client to the API.
"""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from ..profiles import ModelProfile

InterfaceClient = TypeVar('InterfaceClient')


class Provider(ABC, Generic[InterfaceClient]):
    """Abstract class for a provider.

    The provider is in charge of providing an authenticated client to the API.

    Each provider only supports a specific interface. A interface can be supported by multiple providers.

    For example, the `OpenAIChatModel` interface can be supported by the `OpenAIProvider` and the `DeepSeekProvider`.
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

    def model_profile(self, model_name: str) -> ModelProfile | None:
        """The model profile for the named model, if available."""
        return None  # pragma: no cover

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, base_url={self.base_url})'  # pragma: lax no cover


def infer_provider_class(provider: str) -> type[Provider[Any]]:  # noqa: C901
    """Infers the provider class from the provider name."""
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
        from .grok import GrokProvider

        return GrokProvider
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
    elif provider == 'outlines':
        from .outlines import OutlinesProvider

        return OutlinesProvider
    else:  # pragma: no cover
        raise ValueError(f'Unknown provider: {provider}')


def infer_provider(provider: str) -> Provider[Any]:
    """Infer the provider from the provider name."""
    if provider.startswith('gateway/'):
        from .gateway import gateway_provider

        upstream_provider = provider.removeprefix('gateway/')
        return gateway_provider(upstream_provider)
    elif provider in ('google-vertex', 'google-gla'):
        from .google import GoogleProvider

        return GoogleProvider(vertexai=provider == 'google-vertex')
    else:
        provider_class = infer_provider_class(provider)
        return provider_class()
