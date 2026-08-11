from __future__ import annotations as _annotations

import os
from typing import TYPE_CHECKING, overload
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI
from typing_extensions import Self

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.providers import Provider
from pydantic_ai.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from pydantic_ai.realtime import RealtimeModelProfile

try:
    from openai import AsyncAzureOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Azure provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

try:
    from openai.lib.azure import API_KEY_SENTINEL as _api_key_sentinel
except ImportError:  # pragma: no cover
    # An SDK internal, imported separately from the guard above and compared by value below: if a
    # future `openai` release moves or renames it, Entra detection degrades (a key-less Entra client
    # reads as having one) instead of failing every Azure user with a misleading "install `openai`".
    _api_key_sentinel = None


class AzureProvider(Provider[AsyncOpenAI]):
    """Provider for Azure OpenAI API.

    See <https://azure.microsoft.com/en-us/products/ai-foundry> for more information.
    """

    @property
    def name(self) -> str:
        return 'azure'

    @property
    def base_url(self) -> str:
        assert self._base_url is not None
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @property
    def azure_endpoint(self) -> str:
        """The Azure resource endpoint used to derive service-specific URLs."""
        return self._azure_endpoint

    @property
    def api_key(self) -> str:
        """The Azure resource key, for transports that authenticate with one.

        Raises [`UserError`][pydantic_ai.exceptions.UserError] when the provider has no key, i.e. it
        was built from a Microsoft Entra ID client (`azure_ad_token` / `azure_ad_token_provider`).
        """
        if self._api_key is None:
            raise UserError(
                '`AzureProvider` has no API key: it was configured with Microsoft Entra ID '
                'authentication (`azure_ad_token` / `azure_ad_token_provider`). Pass `api_key` (or set '
                '`AZURE_OPENAI_API_KEY`) to use a transport that authenticates with the resource key.'
            )
        return self._api_key

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        model_name = model_name.lower()

        prefix_to_profile = {
            'llama': meta_model_profile,
            'meta-': meta_model_profile,
            'deepseek': deepseek_model_profile,
            'mistralai-': mistral_model_profile,
            'mistral': mistral_model_profile,
            'ministral': mistral_model_profile,
            'magistral': mistral_model_profile,
            'cohere-': cohere_model_profile,
            'grok': grok_model_profile,
        }

        base: ModelProfile | None = None
        is_mistral = False
        for prefix, profile_func in prefix_to_profile.items():
            if model_name.startswith(prefix):
                if prefix.endswith('-'):
                    model_name = model_name[len(prefix) :]
                # Three-layer merge: see OpenRouter for the rationale.
                base = merge_profile(
                    OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
                    profile_func(model_name),
                )
                is_mistral = profile_func is mistral_model_profile
                break
        if base is None:
            # OpenAI models are unprefixed.
            base = openai_model_profile(model_name)

        # Azure Chat Completions API doesn't support document input.
        base = merge_profile(base, OpenAIModelProfile(openai_chat_supports_document_input=False))

        # Reported in #6593 (not verified against the live API here): Azure AI Foundry's
        # Mistral gateway rejects `max_completion_tokens` with a 422 and accepts the legacy
        # `max_tokens` field, so route the `max_tokens` setting to the legacy field.
        # See https://github.com/pydantic/pydantic-ai/issues/6593
        if is_mistral:
            base = merge_profile(base, OpenAIModelProfile(openai_chat_supports_max_completion_tokens=False))

        return base

    @staticmethod
    def realtime_model_profile(model_name: str) -> RealtimeModelProfile:
        return OpenAIProvider.realtime_model_profile(model_name)

    @classmethod
    def for_realtime(
        cls,
        *,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        entra_authenticated: bool = False,
        http_client: httpx.AsyncClient | None = None,
    ) -> Self:
        """Create an Azure provider for the GA realtime API.

        The realtime transport always uses Azure's `/openai/v1` protocol and does not send an
        `api_version`. When neither `api_version` nor `OPENAI_API_VERSION` is set, a bare resource
        endpoint is therefore normalized to its `/openai/v1` form before constructing the provider.
        Explicit arguments otherwise follow the same environment fallbacks and validation as the
        standard constructor.

        Args:
            azure_endpoint: The Azure resource endpoint. Falls back to `AZURE_OPENAI_ENDPOINT`.
            api_version: The API version for endpoints that require one. Falls back to
                `OPENAI_API_VERSION`.
            api_key: The Azure resource key. Falls back to `AZURE_OPENAI_API_KEY`.
            entra_authenticated: Set when every request is authenticated with a Microsoft Entra ID
                credential instead of the resource key (see
                [`AzureRealtimeModel(credential=...)`][pydantic_ai.realtime.azure.AzureRealtimeModel]).
                The key is then neither required nor sent, and `api_key` raises its usual explanatory
                error if anything asks — the same state a provider built from an Entra-authenticated
                `openai_client` lands in.
            http_client: An existing `httpx.AsyncClient` used to construct the provider client.
        """
        if entra_authenticated and not api_key and 'AZURE_OPENAI_API_KEY' not in os.environ:
            # The SDK's own placeholder for "Entra-authenticated, no key", which `__init__` normalizes
            # back to `None` below. Satisfies the key requirement without inventing a credential.
            api_key = _api_key_sentinel
        endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        resolved_api_version = api_version or os.getenv('OPENAI_API_VERSION')
        if endpoint and not resolved_api_version and _openai_compatible_v1_base_url(endpoint) is None:
            endpoint = endpoint.rstrip('/') + '/openai/v1'
        return cls(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
            http_client=http_client,
        )

    @overload
    def __init__(self, *, openai_client: AsyncAzureOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncAzureOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Azure provider.

        Args:
            azure_endpoint: The Azure endpoint to use for authentication, if not provided, the `AZURE_OPENAI_ENDPOINT`
                environment variable will be used if available.
            api_version: The API version to use for authentication, if not provided, the `OPENAI_API_VERSION`
                environment variable will be used if available. Not required (and not sent) when
                `azure_endpoint` targets the [Azure OpenAI v1 GA API](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle)
                (i.e. a path ending in `/v1`, such as `https://<resource>.openai.azure.com/openai/v1/`)
                or an Azure AI Foundry serverless model endpoint (`*.models.ai.azure.com`), both of
                which reject the `api-version` query parameter.
            api_key: The API key to use for authentication, if not provided, the `AZURE_OPENAI_API_KEY` environment variable
                will be used if available.
            openai_client: An existing
                [`AsyncAzureOpenAI`](https://github.com/openai/openai-python#microsoft-azure-openai)
                client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if openai_client is not None:
            assert azure_endpoint is None, 'Cannot provide both `openai_client` and `azure_endpoint`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._base_url = str(openai_client.base_url)
            self._client = openai_client
            self._azure_endpoint = self._base_url.partition('/openai/')[0].rstrip('/')
            # An Entra-authenticated client (`azure_ad_token`/`azure_ad_token_provider`) has no API key,
            # but the SDK still fills `api_key` with a truthy placeholder. Treat it as absent, so
            # `api_key` raises its usual explanatory error instead of realtime sending the placeholder
            # as a credential and getting an opaque auth failure back.
            self._api_key = None if openai_client.api_key == _api_key_sentinel else openai_client.api_key or None
        else:
            azure_endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
            if not azure_endpoint:
                raise UserError(
                    'Must provide one of the `azure_endpoint` argument or the `AZURE_OPENAI_ENDPOINT` environment variable'
                )

            if not api_key and 'AZURE_OPENAI_API_KEY' not in os.environ:  # pragma: no cover
                raise UserError(
                    'Must provide one of the `api_key` argument or the `AZURE_OPENAI_API_KEY` environment variable'
                )

            self._azure_endpoint = azure_endpoint.rstrip('/')
            resolved_api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
            # The SDK's Entra placeholder means "no key" here too, exactly as on the `openai_client`
            # branch above, so `api_key` reports its absence rather than handing the placeholder out.
            self._api_key = None if resolved_api_key == _api_key_sentinel else resolved_api_key

            if http_client is None:
                http_client = create_async_http_client()
                self._own_http_client = http_client
                self._http_client_factory = create_async_http_client

            # The Azure OpenAI v1 GA API and Azure AI Foundry serverless model
            # endpoints expose an OpenAI-compatible `/v1` API that rejects the
            # `api-version` query parameter that `AsyncAzureOpenAI` always
            # injects, so we use a plain `AsyncOpenAI` client instead.
            if (v1_base_url := _openai_compatible_v1_base_url(azure_endpoint)) is not None:
                if api_version is not None:
                    raise UserError(
                        '`api_version` must not be set when `azure_endpoint` targets the Azure OpenAI '
                        'v1 API or an Azure AI Foundry serverless model endpoint, which do not accept it.'
                    )
                self._client = AsyncOpenAI(
                    base_url=v1_base_url,
                    api_key=api_key or os.getenv('AZURE_OPENAI_API_KEY'),
                    http_client=http_client,
                )
                self._base_url = str(self._client.base_url)
            else:
                if not api_version and 'OPENAI_API_VERSION' not in os.environ:  # pragma: no cover
                    raise UserError(
                        'Must provide one of the `api_version` argument or the `OPENAI_API_VERSION` environment variable'
                    )

                self._client = AsyncAzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version=api_version,
                    http_client=http_client,
                )
                self._base_url = str(self._client.base_url)

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]


def _openai_compatible_v1_base_url(endpoint: str) -> str | None:
    """Return the `/v1` base URL for Azure endpoints that expose the OpenAI-compatible API, or `None`.

    These endpoints reject the `api-version` query parameter that
    `AsyncAzureOpenAI` always injects, so callers need a plain `AsyncOpenAI`
    client instead. Matches:

    - Any endpoint whose path ends with `/v1` — explicit opt-in to the
      [Azure OpenAI v1 GA API](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle),
      e.g. `https://<resource>.openai.azure.com/openai/v1/` or
      `https://<resource>.services.ai.azure.com/openai/v1/`.
    - Any `*.models.ai.azure.com` host — Azure AI Foundry serverless
      model-per-endpoint deployments, which always serve an OpenAI-compatible
      `/v1` API at the root.
    """
    stripped = endpoint.rstrip('/')
    if stripped.endswith('/v1'):
        return stripped
    host = urlparse(stripped).hostname or ''
    if host.endswith('.models.ai.azure.com'):
        return f'{stripped}/v1'
    return None
