"""A provider for any service with an OpenAI-compatible API."""

from __future__ import annotations as _annotations

from collections.abc import Callable, Iterable
from typing import overload

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.groq import groq_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the OpenAI-compatible provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

from ._openai_compatible import (
    AsyncHTTPClient as _OpenAIHTTPClient,
    _OpenAISDKProvider,  # pyright: ignore[reportPrivateUsage]
)

__all__ = ('OpenAICompatibleProvider',)


def _mistral_model_profile(model_name: str) -> ModelProfile:
    """Mistral's profile, on the OpenAI defaults assumed for a model no family profile describes.

    Outside Magistral, the Mistral profile only constrains tool call IDs, so the rest of the profile still
    comes from the OpenAI fallback that applied before it set anything.
    """
    profile = mistral_model_profile(model_name)
    if model_name.startswith('magistral'):
        return profile
    return merge_profile(openai_model_profile(model_name), profile)


def _vendor_prefixed_model_profile(model_name: str) -> ModelProfile:
    """The profile for a model on an OpenAI-compatible service without a provider of its own.

    A `vendor/model` name gets that vendor's family profile, and any other name the profile OpenAI-compatible
    endpoints share, without the flags only OpenAI's own API supports. Either way, with the JSON schema
    transformer `OpenAIChatModel` uses.
    """
    vendor_profiles: dict[str, Callable[[str], ModelProfile | None]] = {
        'anthropic': anthropic_model_profile,
        'openai': openai_model_profile,
        'google': google_model_profile,
        'mistralai': _mistral_model_profile,
        'mistral': _mistral_model_profile,
        'cohere': cohere_model_profile,
        'amazon': amazon_model_profile,
        'bedrock': amazon_model_profile,
        'meta-llama': meta_model_profile,
        'meta': meta_model_profile,
        'groq': groq_model_profile,
        'deepseek': deepseek_model_profile,
        'moonshotai': moonshotai_model_profile,
        'x-ai': grok_model_profile,
        'qwen': qwen_model_profile,
    }
    profile = None
    if '/' in model_name:
        vendor, model_suffix = model_name.split('/', 1)
        if vendor_profile := vendor_profiles.get(vendor):
            profile = vendor_profile(model_suffix)
    if profile is None:
        profile = openai_model_profile(model_name)
    return merge_profile(OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer), profile)


class OpenAICompatibleProvider(_OpenAISDKProvider):
    """Provider for a service with an OpenAI-compatible API that doesn't have a provider of its own.

    Use it with [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel], or with
    [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] if the service implements the
    Responses API:

    ```python {test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider('my-gateway', base_url='https://gateway.example.com/v1', api_key='...')
    agent = Agent(OpenAIChatModel('qwen/qwen3-coder', provider=provider))
    ```

    The `name` you give it identifies the service in message history and telemetry, so it decides which
    history data the service gets back: stateful data, such as response and file IDs, only goes back to a
    provider with the same name, and stateless signed or encrypted data, such as encrypted reasoning, only to
    one with the same [`canonical_system`][pydantic_ai.providers.Provider.canonical_system], which defaults to
    the name. Changing the name later stops that data from being sent back for history recorded under the old
    one.

    For OpenAI itself, or a transparent proxy to it, use
    [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] instead.
    """

    @overload
    def __init__(
        self,
        name: str,
        *,
        openai_client: AsyncOpenAI,
        profile: ModelProfile | Callable[[str], ModelProfile | None] | None = None,
        canonical_system: str | None = None,
        compatible_systems: Iterable[str] = (),
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        *,
        base_url: str,
        api_key: str | None = None,
        http_client: _OpenAIHTTPClient | None = None,
        profile: ModelProfile | Callable[[str], ModelProfile | None] | None = None,
        canonical_system: str | None = None,
        compatible_systems: Iterable[str] = (),
    ) -> None: ...

    def __init__(
        self,
        name: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
        profile: ModelProfile | Callable[[str], ModelProfile | None] | None = None,
        canonical_system: str | None = None,
        compatible_systems: Iterable[str] = (),
    ) -> None:
        """Create a new OpenAI-compatible provider.

        Args:
            name: The name of the service, such as `'my-gateway'`. It's recorded as the `provider_name` of
                every response, used in OpenTelemetry attributes, and decides which history data is sent
                back to the service (see above).
            base_url: The base URL of the service's OpenAI-compatible API, such as `'https://gateway.example.com/v1'`.
                Required unless `openai_client` is given.
            api_key: The API key for the service, if it needs one. `OPENAI_API_KEY` is never used, so an
                OpenAI key isn't sent to another service.
            openai_client: An existing `AsyncOpenAI` client to use. If given, `base_url`, `api_key` and
                `http_client` must not be.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
            profile: The model profile for the service's models. A `ModelProfile` is merged over the default for
                every model, and a function from the model name to a profile replaces it. The default gives a
                `vendor/model` name that vendor's profile, and any other name the profile OpenAI-compatible
                APIs share, without the features only OpenAI's own API supports.
            canonical_system: The [`canonical_system`][pydantic_ai.providers.Provider.canonical_system] of the
                stateless signed or encrypted data the service returns. Defaults to `name`. Set it to another
                provider's canonical system only when that provider's keys verify this service's data, as for a
                transparent proxy.
            compatible_systems: Other canonical systems whose stateless data the service also verifies; see
                [`compatible_systems`][pydantic_ai.providers.Provider.compatible_systems].
        """
        if openai_client is not None:
            if base_url is not None or api_key is not None or http_client is not None:
                raise UserError('`base_url`, `api_key` and `http_client` cannot be used with `openai_client`.')
            self._client = openai_client
        elif base_url is None:
            raise UserError(f'`{type(self).__name__}` needs a `base_url` or an `openai_client`.')
        else:
            # The OpenAI client falls back to `OPENAI_API_KEY` and requires a key, so a placeholder is sent to a
            # service that doesn't need one.
            self._client = self._create_openai_client(
                base_url=base_url, api_key=api_key or 'api-key-not-set', http_client=http_client
            )
        self._name = name
        self._model_profile_override = profile
        self._canonical_system = canonical_system or name
        self._compatible_systems = frozenset(compatible_systems)

    @property
    def name(self) -> str:
        return self._name

    @property
    def canonical_system(self) -> str:
        return self._canonical_system

    @property
    def compatible_systems(self) -> frozenset[str]:
        return self._compatible_systems

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        """The default model profile for the named model, described in `__init__`, before `profile=` applies."""
        return _vendor_prefixed_model_profile(model_name)

    def _configured_model_profile(self, model_name: str) -> ModelProfile | None:
        """The default model profile with this provider's `profile=` applied."""
        override = self._model_profile_override
        if callable(override):
            return override(model_name)
        default = self.model_profile(model_name)
        return default if override is None else merge_profile(default, override)
