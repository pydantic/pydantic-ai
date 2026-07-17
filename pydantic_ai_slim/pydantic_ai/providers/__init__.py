"""Providers for the API clients.

The providers are in charge of providing an authenticated client to the API.
"""

from __future__ import annotations as _annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import TracebackType
from typing import Any, Generic

import anyio
import httpx
from typing_extensions import Self, TypeVar

from ..exceptions import UserError
from ..profiles import ModelProfile

InterfaceClient = TypeVar('InterfaceClient', default=Any)

_KEYLESS_HINT = (
    "To try Pydantic AI without an API key, use the built-in test model: `Agent('test')`. "
    'See https://ai.pydantic.dev/testing/'
)


def missing_api_key_error(message: str) -> UserError:
    """Build a [`UserError`][pydantic_ai.exceptions.UserError] for missing provider credentials.

    The provider-specific `message` (which environment variable to set or how to pass the key) is followed by a
    hint pointing newcomers to the keyless [test model](https://ai.pydantic.dev/testing/), so a missing key never
    dead-ends the getting-started experience.
    """
    return UserError(f'{message} {_KEYLESS_HINT}')


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

    @functools.cached_property
    def _enter_lock(self) -> anyio.Lock:
        # We use a cached_property for this because `anyio.Lock` binds to the event loop on which
        # it's first used; deferring creation until first access ensures it binds to the correct
        # running loop and avoids issues with Temporal's workflow sandbox.
        return anyio.Lock()

    @property
    @abstractmethod
    def name(self) -> str:
        """The provider name.

        The returned value flows into [`ModelMessage.provider_name`][pydantic_ai.messages.ModelMessage]
        on every part. Thinking-tag detection and native-tool detection check this value when
        the model class loads history, so silently renaming a concrete `name` value breaks
        replay of any message history captured against the old name.
        """
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
        if self._entered_count == 0:
            # No matching `__aenter__` - keep this a no-op so the provider can be re-entered cleanly.
            return
        async with self._enter_lock:
            self._entered_count -= 1
            if self._entered_count == 0 and self._own_http_client is not None:
                await self._own_http_client.aclose()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, base_url={self.base_url})'  # pragma: lax no cover


def infer_provider_class(provider: str) -> type[Provider[Any]]:  # noqa: C901
    """Infers the provider class from the provider name."""
    # Strip the `gateway/` prefix to get the canonical class-lookup name. The
    # Gateway URL route value (e.g. `google-vertex`) is a separate concern
    # handled by `_gateway_route` in `providers/gateway.py`.
    if provider.startswith('gateway/'):
        from .gateway import normalize_gateway_provider

        provider = normalize_gateway_provider(provider)

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
    elif provider in ('azure', 'azure-responses'):
        from .azure import AzureProvider

        return AzureProvider
    elif provider == 'google':
        from .google import GoogleProvider

        return GoogleProvider
    elif provider == 'google-cloud':
        from .google_cloud import GoogleCloudProvider

        return GoogleCloudProvider
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
    elif provider == 'sentence-transformers':
        from .sentence_transformers import SentenceTransformersProvider

        return SentenceTransformersProvider
    elif provider == 'voyageai':
        from .voyageai import VoyageAIProvider

        return VoyageAIProvider
    elif provider == 'zai':
        from .zai import ZaiProvider

        return ZaiProvider
    else:
        raise ValueError(f'Unknown provider: {provider}')


def infer_provider(provider: str) -> Provider[Any]:
    """Infer the provider from the provider name."""
    if provider.startswith('gateway/'):
        from .gateway import gateway_provider

        upstream_provider = provider.removeprefix('gateway/')
        return gateway_provider(upstream_provider)

    provider_class = infer_provider_class(provider)
    return provider_class()
