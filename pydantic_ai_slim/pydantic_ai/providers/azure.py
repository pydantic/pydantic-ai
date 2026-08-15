from __future__ import annotations as _annotations

import os
from typing import TYPE_CHECKING, overload
from urllib.parse import urlparse

from typing_extensions import Self

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile

if TYPE_CHECKING:
    from pydantic_ai.realtime import RealtimeModelProfile

try:
    from openai import AsyncAzureOpenAI, AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Azure provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from pydantic_ai.providers.openai import OpenAIProvider

    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )

try:
    from openai.lib.azure import API_KEY_SENTINEL as _api_key_sentinel
except ImportError:  # pragma: no cover
    # An SDK internal, imported separately from the guard above and compared by value below: if a
    # future `openai` release moves or renames it, Entra detection degrades (a key-less Entra client
    # reads as having one) instead of failing every Azure user with a misleading "install `openai`".
    _api_key_sentinel = None


_DEFAULT_VOICE_LIVE_API_VERSION = '2026-04-10'
"""Default Azure AI Voice Live API version when neither `AZURE_VOICELIVE_API_VERSION` nor an argument is set."""


class AzureProvider(_OpenAICompatibleProvider):
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

    # Azure AI Voice Live is a distinct resource. `AzureRealtimeModel` (which requires an `AzureProvider`)
    # reads these when `azure_voice_live=True` so the endpoint, key, and API version always come from one
    # coherent set — the `AZURE_VOICELIVE_*` variables, falling back to the resolved Azure OpenAI values —
    # rather than being mixed across resources.

    @property
    def voice_live_endpoint(self) -> str:
        """The Azure AI Voice Live endpoint (`AZURE_VOICELIVE_ENDPOINT`, else the Azure OpenAI endpoint)."""
        return self._voice_live_endpoint

    @property
    def voice_live_api_key(self) -> str:
        """The Azure AI Voice Live key (`AZURE_VOICELIVE_API_KEY`, else the Azure OpenAI key)."""
        if self._voice_live_api_key is None:
            raise UserError('Azure AI Voice Live requires API-key authentication.')
        return self._voice_live_api_key

    @property
    def voice_live_api_version(self) -> str:
        """The Azure AI Voice Live API version (`AZURE_VOICELIVE_API_VERSION`, else a supported default)."""
        return self._voice_live_api_version

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
        # The GA-vs-Voice-Live serving API is stamped by `AzureRealtimeModel.profile`, which (unlike this
        # static provider method) can also read the session settings that select between the two.
        return OpenAIProvider.realtime_model_profile(model_name)

    @classmethod
    def for_realtime(
        cls,
        *,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        entra_authenticated: bool = False,
        http_client: _OpenAIHTTPClient | None = None,
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
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` used to construct the provider client.
        """
        if entra_authenticated and not api_key and not os.getenv('AZURE_OPENAI_API_KEY'):
            # The SDK's own placeholder for "Entra-authenticated, no key", which `__init__` normalizes
            # back to `None` below. Satisfies the key requirement without inventing a credential. A
            # truthiness check (not membership) so an empty `AZURE_OPENAI_API_KEY=""` still takes the
            # placeholder rather than resolving to an empty key that trips the key-required error.
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
        voice_live_endpoint: str | None = None,
        voice_live_api_key: str | None = None,
        voice_live_api_version: str | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        voice_live_endpoint: str | None = None,
        voice_live_api_key: str | None = None,
        voice_live_api_version: str | None = None,
        openai_client: AsyncAzureOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
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
            voice_live_endpoint: The [Azure AI Voice Live](https://learn.microsoft.com/azure/ai-services/speech-service/voice-live)
                endpoint, used only by [`AzureRealtimeModel`][pydantic_ai.realtime.azure.AzureRealtimeModel]
                with `azure_voice_live=True`. Voice Live is a distinct Azure resource, so when this is
                not provided the `AZURE_VOICELIVE_ENDPOINT` environment variable is used, and finally
                `azure_endpoint` as a fallback.
            voice_live_api_key: The Voice Live API key; falls back to `AZURE_VOICELIVE_API_KEY`, then `api_key`.
            voice_live_api_version: The Voice Live API version; falls back to `AZURE_VOICELIVE_API_VERSION`,
                then a supported default. Deliberately *not* derived from `api_version`, which versions
                the Azure OpenAI data plane on an unrelated schedule.
            openai_client: An existing
                [`AsyncAzureOpenAI`](https://github.com/openai/openai-python#microsoft-azure-openai)
                client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
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
            # Resolve the Voice Live endpoint/key from their own environment prefix here too, exactly as the
            # `else` branch does before `_resolve_voice_live_credentials` (which expects them pre-resolved and
            # only falls back to the Azure OpenAI values). Without this, `AzureProvider(openai_client=...)`
            # would ignore `AZURE_VOICELIVE_ENDPOINT` / `AZURE_VOICELIVE_API_KEY` and silently point Voice Live
            # at the Azure OpenAI client's resource.
            self._resolve_voice_live_credentials(
                voice_live_endpoint or os.getenv('AZURE_VOICELIVE_ENDPOINT'),
                voice_live_api_key or os.getenv('AZURE_VOICELIVE_API_KEY'),
                voice_live_api_version,
            )
        else:
            # Azure AI Voice Live (used by `AzureRealtimeModel`) is a distinct resource with its own
            # credentials. Each set is resolved from its own sources — explicit argument, then that set's
            # environment prefix — and the two only meet at the atomic fallback below.
            azure_openai_endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
            azure_openai_api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
            azure_openai_api_version = api_version or os.getenv('OPENAI_API_VERSION')
            voice_live_endpoint = voice_live_endpoint or os.getenv('AZURE_VOICELIVE_ENDPOINT')
            voice_live_api_key = voice_live_api_key or os.getenv('AZURE_VOICELIVE_API_KEY')

            # All-or-nothing: the Voice Live set supplies the data-plane client only when *nothing* about
            # the Azure OpenAI resource was configured — a Voice-Live-only setup, where that client is a
            # formality (a Voice Live resource doesn't serve the Azure OpenAI data plane) and its values
            # merely let construction succeed. Filling individual gaps instead would point an Azure OpenAI
            # resource at another resource's key or version.
            is_voice_live_resource = not (azure_openai_endpoint or azure_openai_api_key or azure_openai_api_version)

            azure_endpoint = azure_openai_endpoint or (voice_live_endpoint if is_voice_live_resource else None)
            if not azure_endpoint:
                raise UserError(
                    'Must provide the `azure_endpoint` argument or set the `AZURE_OPENAI_ENDPOINT` '
                    '(or, for a Voice-Live-only provider, `voice_live_endpoint` / `AZURE_VOICELIVE_ENDPOINT`) '
                    'environment variable'
                )

            api_key = azure_openai_api_key or (voice_live_api_key if is_voice_live_resource else None)
            if not api_key:
                raise UserError(
                    'Must provide the `api_key` argument or set the `AZURE_OPENAI_API_KEY` '
                    '(or, for a Voice-Live-only provider, `voice_live_api_key` / `AZURE_VOICELIVE_API_KEY`) '
                    'environment variable'
                )

            self._azure_endpoint = azure_endpoint.rstrip('/')
            # The SDK's Entra placeholder means "no key" here too, exactly as on the `openai_client`
            # branch above, so `api_key` reports its absence rather than handing the placeholder out.
            # Normalized before the Voice Live set resolves, since that falls back to this value.
            self._api_key = None if api_key == _api_key_sentinel else api_key
            self._resolve_voice_live_credentials(voice_live_endpoint, voice_live_api_key, voice_live_api_version)

            http_client = self._get_http_client(http_client)

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
                    api_key=api_key,
                    http_client=http_client,  # pyright: ignore[reportArgumentType]
                )
                self._base_url = str(self._client.base_url)
            else:
                api_version = azure_openai_api_version
                if not api_version and is_voice_live_resource:
                    # A Voice-Live-only configuration has no data-plane version to offer, and the client
                    # built from it is a formality, so its own resolved version (which has a default) lets
                    # construction succeed. Never borrowed for a real Azure OpenAI resource: the two
                    # version schemes are unrelated, so the data plane would be called with a version it
                    # doesn't recognize.
                    api_version = self._voice_live_api_version
                if not api_version:
                    raise UserError(
                        'Must provide the `api_version` argument or set the `OPENAI_API_VERSION` '
                        'environment variable (`AZURE_VOICELIVE_API_VERSION` stands in only when the '
                        'whole resource is configured through the `AZURE_VOICELIVE_*` variables)'
                    )

                self._client = AsyncAzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version=api_version,
                    http_client=http_client,  # pyright: ignore[reportArgumentType]
                )
                self._base_url = str(self._client.base_url)

    def _resolve_voice_live_credentials(
        self,
        voice_live_endpoint: str | None,
        voice_live_api_key: str | None,
        voice_live_api_version: str | None,
    ) -> None:
        """Resolve the Azure AI Voice Live credential set (endpoint/key/version) as one coherent group.

        `voice_live_endpoint`/`voice_live_api_key` arrive already resolved from the explicit argument or
        `AZURE_VOICELIVE_*`; anything still unset falls back to the resolved Azure OpenAI endpoint/key, so
        a single Azure AI Foundry resource serving both needs configuring only once. Read by
        `AzureRealtimeModel`'s Voice Live path (`azure_voice_live=True`).

        The version deliberately does not fall back to `api_version`: that versions the Azure OpenAI data
        plane, on a schedule unrelated to Voice Live's beta API, so inheriting it would dial a version
        Voice Live does not serve. It defaults instead.
        """
        self._voice_live_endpoint = (voice_live_endpoint or self._azure_endpoint).rstrip('/')
        self._voice_live_api_key = voice_live_api_key or self._api_key
        self._voice_live_api_version = (
            voice_live_api_version or os.getenv('AZURE_VOICELIVE_API_VERSION') or _DEFAULT_VOICE_LIVE_API_VERSION
        )


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
