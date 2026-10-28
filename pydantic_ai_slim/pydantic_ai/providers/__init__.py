"""Providers for the API clients.

The providers are in charge of providing an authenticated client to the API.
"""

from __future__ import annotations as _annotations

import asyncio
import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self
from urllib.parse import urlsplit

import anyio
from typing_extensions import TypeVar

from .._http import AsyncHTTPClient
from ..exceptions import UserError
from ..profiles import ModelProfile

if TYPE_CHECKING:
    from ..realtime import RealtimeModelProfile

InterfaceClient = TypeVar('InterfaceClient', default=Any)

_KEYLESS_HINT = (
    "To try Pydantic AI without an API key, use the built-in test model: `Agent('test')`. "
    'See https://pydantic.dev/docs/ai/guides/testing/'
)


def missing_api_key_error(message: str) -> UserError:
    """Build a [`UserError`][pydantic_ai.exceptions.UserError] for missing provider credentials.

    The provider-specific `message` (which environment variable to set or how to pass the key) is followed by a
    hint pointing newcomers to the keyless [test model](https://pydantic.dev/docs/ai/guides/testing/), so a missing key never
    dead-ends the getting-started experience.
    """
    return UserError(f'{message} {_KEYLESS_HINT}')


_DEFAULT_PORTS = {'http': 80, 'https': 443}


def _normalized_url(url: str) -> str:
    """`url` with a lowercase scheme and host, an explicit default port and no trailing `/`, for comparing endpoints."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port or _DEFAULT_PORTS.get(parts.scheme.lower())
    except ValueError:  # a malformed stored URL, such as an unclosed IPv6 bracket or an out-of-range port
        return url
    scheme = parts.scheme.lower()
    netloc = f'{parts.hostname or ""}:{port}' if port else parts.hostname or ''
    query = f'?{parts.query}' if parts.query else ''
    return f'{scheme}://{netloc}{parts.path.rstrip("/")}{query}'


class Provider(ABC, Generic[InterfaceClient]):
    """Abstract class for a provider.

    The provider is in charge of providing an authenticated client to the API.

    Each provider only supports a specific interface. An interface can be supported by multiple providers.

    For example, the `OpenAIChatModel` interface can be supported by the `OpenAIProvider` and the `DeepSeekProvider`.

    When used as an async context manager, providers that create their own HTTP client will close it on exit.
    This is handled automatically when using [`Agent`][pydantic_ai.agent.Agent] as a context manager.
    Outside one, a run on a new event loop, while the one the client was last used on isn't running, gets a new client.
    """

    _client: InterfaceClient
    _own_http_client: AsyncHTTPClient | None = None
    _http_client_factory: Callable[[], AsyncHTTPClient] | None = None
    _own_client_loop: asyncio.AbstractEventLoop | None = None
    _entered_count: int = 0
    _model_id_namespace: str | None = None

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
    def canonical_system(self) -> str:
        """The system whose keys and format verify the stateless signed or encrypted data this provider returns.

        That data is encrypted reasoning, thinking and thought signatures, redacted thinking and OpenAI
        compaction items. A model sends it back only to a provider whose `canonical_system` matches, or
        lists it in [`compatible_systems`][pydantic_ai.providers.Provider.compatible_systems]; other
        providers get the fallback each model uses for foreign data, such as thinking rendered as text.
        Server-side state, such as response, item and container IDs, is judged by endpoint instead: see
        [`shares_server_state_with`][pydantic_ai.providers.Provider.shares_server_state_with].

        Defaults to `name`. Built-in providers return a fixed value, so a subclass that only changes
        `name` (one that authenticates differently against the same API, say) keeps sharing data with the
        provider it extends. `OpenAIProvider` is the exception: a subclass with its own `name` is assumed
        to talk to another service, and gets that name. Override this when your provider returns data
        another provider's keys verify, like a transparent proxy.
        """
        return self.name

    _canonical_system_follows_client: ClassVar[bool] = False
    """Whether `canonical_system` depends on the client the provider was created with, not only on `name`.

    Responses then always record their canonical system, so one name covering two systems (`google` on the Gemini
    API or Google Cloud, `openai` on OpenAI or Azure) is never ambiguous in new history.
    """

    @property
    def compatible_systems(self) -> frozenset[str]:
        """Other canonical systems whose stateless signed or encrypted data this provider also verifies.

        One-way: listing a system here sends its data to this provider, not this provider's data to it.
        See [`canonical_system`][pydantic_ai.providers.Provider.canonical_system]. Empty by default.
        """
        return frozenset()

    def shares_server_state_with(self, provider_url: str) -> bool:
        """Whether this provider can look up server-side state created at the endpoint `provider_url`.

        Server-side state is what the API stores and refers to by ID: stored responses (`previous_response_id`),
        output items, conversations and code execution containers. A model sends those IDs back only when the
        response that holds them has this provider's [`name`][pydantic_ai.providers.Provider.name] and this method
        accepts the response's [`provider_url`][pydantic_ai.messages.ModelResponse.provider_url]; a response
        without a `provider_url` is accepted by name alone. Other content in the response, such as text and
        client tool calls, is sent either way.

        Defaults to comparing `provider_url` with [`base_url`][pydantic_ai.providers.Provider.base_url], ignoring
        the case of the scheme and host, a default port and a trailing `/`. Override this when two URLs reach the
        same stored state, like a proxy alias for the API.
        """
        return _normalized_url(provider_url) == _normalized_url(self.base_url)

    @property
    def model_id_namespace(self) -> str:
        """The namespace used to fully qualify model IDs routed through this provider."""
        return self._model_id_namespace or self.name

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

    def _configured_model_profile(self, model_name: str) -> ModelProfile | None:
        """The model profile for the named model as this provider instance is configured, which models use.

        `model_profile`, unless a provider takes a profile when it's created, like `OpenAICompatibleProvider`.
        """
        return self.model_profile(model_name)

    @staticmethod
    def realtime_model_profile(model_name: str) -> RealtimeModelProfile | None:
        """The realtime model profile for the named model, if available."""
        return None

    def _set_http_client(self, http_client: AsyncHTTPClient) -> None:
        """Update the SDK client's internal HTTP client reference.

        Subclasses that manage their own HTTP client should override this to inject
        the new client into their SDK client after re-creation.
        """

    def _moved_to_another_event_loop(self) -> bool:
        """Record the running event loop as the one this provider's own client is used on.

        Returns whether the client was last used on a different event loop that isn't running now. Connections
        belong to the event loop that opened them, so a client can't be used from another one, and one whose
        loop has closed can't be closed either. While the loop it was last used on is still running, for example in
        another thread, its runs may still be using the client, so it isn't replaced under them. Only that last
        loop is checked: with three or more loops taking turns, a run still going on an earlier one can see the
        client replaced.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No asyncio event loop, e.g. under Trio, so there's nothing to compare.
            return False
        previous_loop, self._own_client_loop = self._own_client_loop, loop
        return previous_loop is not None and previous_loop is not loop and not previous_loop.is_running()

    def _refresh_http_client(self) -> None:
        """Replace the HTTP client this provider created if it's closed or was used on an event loop that isn't running.

        Called when the provider is entered, and at the start of a run that uses it without entering it. While
        the provider is entered, whoever entered it owns the client, so it's left alone. The replaced client
        isn't closed: either it already is, or its event loop is gone or belongs to other code. A client that
        was passed in is never replaced.
        """
        own_client = self._own_http_client
        if self._entered_count or own_client is None or self._http_client_factory is None:
            return
        if self._moved_to_another_event_loop() or own_client.is_closed:
            new_client = self._http_client_factory()
            self._own_http_client = new_client
            self._set_http_client(new_client)

    async def __aenter__(self) -> Self:
        async with self._enter_lock:
            self._refresh_http_client()
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

    if provider in ('openai', 'openai-chat', 'openai-responses', 'openai-decisions'):
        from .openai import OpenAIProvider

        return OpenAIProvider
    elif provider == 'openai-codex':
        from .openai_codex import OpenAICodexProvider

        return OpenAICodexProvider
    elif provider == 'openai-chatgpt':
        from .openai_chatgpt import OpenAIChatGPTProvider

        return OpenAIChatGPTProvider
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
    elif provider == 'bedrock-mantle':
        from .bedrock_mantle import BedrockMantleProvider

        return BedrockMantleProvider
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
    elif provider == 'system-one':
        from .system_one import SystemOneProvider

        return SystemOneProvider
    elif provider == 'crusoe':
        from .crusoe import CrusoeProvider

        return CrusoeProvider
    elif provider == 'elevenlabs':
        from .elevenlabs import ElevenLabsProvider

        return ElevenLabsProvider
    elif provider == 'xai':
        from .xai import XaiProvider

        return XaiProvider
    elif provider == 'moonshotai':
        from .moonshotai import MoonshotAIProvider

        return MoonshotAIProvider
    elif provider == 'meta':
        from .meta import MetaProvider

        return MetaProvider
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
        from .github import GitHubProvider  # pyright: ignore[reportDeprecated]

        return GitHubProvider  # pyright: ignore[reportDeprecated]
    elif provider == 'github-copilot':
        from .github_copilot import GitHubCopilotProvider

        return GitHubCopilotProvider
    elif provider == 'litellm':
        from .litellm import LiteLLMProvider

        return LiteLLMProvider
    elif provider == 'vllm':
        from .vllm import VLLMProvider

        return VLLMProvider
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
    elif provider == 'snowflake':
        from .snowflake import SnowflakeProvider

        return SnowflakeProvider
    elif provider == 'typesafe':
        from .typesafe import TypeSafeProvider

        return TypeSafeProvider
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
