from __future__ import annotations as _annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.models import DEFAULT_HTTP_TIMEOUT, create_async_http_client, get_user_agent
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.providers import Provider, missing_api_key_error

if TYPE_CHECKING:
    from pydantic_ai.realtime import RealtimeModelProfile

try:
    from google.genai.client import Client
    from google.genai.types import HttpOptions, HttpRetryOptions
except ImportError as _import_error:
    raise ImportError(
        'Please install the `google-genai` package to use the Google provider, '
        'you can use the `google` optional group — `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error


class BaseGoogleProvider(Provider[Client], ABC):
    """Common base for the Gemini API and Google Cloud providers.

    Abstract — instantiate [`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider] for the
    Gemini API or [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] for
    Google Cloud. Subclasses share `base_url`, `client`, `_set_http_client`, and model-profile
    lookup; each subclass owns its own `Client` construction.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def base_url(self) -> str:
        return str(self._client._api_client._http_options.base_url)  # pyright: ignore[reportPrivateUsage]

    @property
    def client(self) -> Client:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return google_model_profile(model_name)

    @staticmethod
    def realtime_model_profile(model_name: str) -> RealtimeModelProfile:
        return {
            'supports_image_input': True,
            # Verified live on both currently served Live models: a session asking for `TEXT` is closed
            # with `1007 The requested combination of response modalities (TEXT) is not supported by the
            # model`, matching Google's note that native-audio models only support the `AUDIO` modality.
            # Output transcription (enabled by default) is how a Live session gets text.
            'supports_text_output': False,
            'supports_session_seeding': True,
            'supports_seeding_images': True,
            'supports_seeding_audio': False,
            'audio_input_sample_rate': 16000,
            'audio_output_sample_rate': 24000,
            # Search grounding only. Google's Live tool matrix lists code execution and URL context as
            # unsupported for every Live model, and that matches the live behavior:
            # `gemini-2.5-flash-native-audio-latest` closes the session with `1007 Code Execution tool
            # is not supported for this model`, and its URL-context grounding answers "I was unable to
            # access the page"; `gemini-3.1-flash-live-preview` accepts both declarations and then
            # produces neither a code-execution part nor URL-context metadata. Advertising them makes
            # a `WebFetch()` or `CodeExecutionTool()` a silent no-op; leaving them out means a `local=`
            # fallback is used instead, or the shared `UserError` points at one.
            'supported_native_tools': frozenset({WebSearchTool}),
            # Every current Gemini Live model takes a thinking config (verified live for both
            # `gemini-2.5-flash-native-audio-latest` and `gemini-3.1-flash-live-preview`), which Google
            # documents as `thinkingBudget` on the 2.5 family and `thinkingLevel` on 3.x.
            'supports_thinking': True,
            # Only the native-audio models actually honor `Behavior.NON_BLOCKING`; verified live with
            # a slow tool, where `gemini-2.5-flash-native-audio-latest` keeps speaking throughout and
            # `gemini-3.1-flash-live-preview` accepts the flag but still goes silent until the result
            # lands. This gates the opt-in `google_async_tool_calls` setting; it is not enabled by
            # merely being supported.
            'supports_async_tool_calls': 'native-audio' in model_name,
            # Gemini Live takes a tool's return schema natively, as the function declaration's
            # `response` schema (matching the classic `GoogleModel`'s `response_json_schema`).
            'supports_tool_return_schema': True,
        }

    def _build_http_options(
        self,
        *,
        http_client: httpx.AsyncClient | None,
        base_url: str | None,
        retry_options: HttpRetryOptions | None = None,
    ) -> HttpOptions:
        """Build `HttpOptions` and record ownership of the httpx client if we created it.

        Subclasses call this before constructing their `Client(...)` to keep timeout / user-agent /
        ownership wiring consistent.
        """
        if http_client is None:
            http_client = create_async_http_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_http_client
        # google-genai's `HttpOptions.timeout` defaults to None, which makes the SDK pass
        # `timeout=None` to httpx and override any timeout on the supplied client. Pin the timeout
        # here (ms) so requests actually time out.
        timeout_seconds = http_client.timeout.read or DEFAULT_HTTP_TIMEOUT
        timeout_ms = int(timeout_seconds * 1000)
        return HttpOptions(
            base_url=base_url,
            headers={'User-Agent': get_user_agent()},
            httpx_async_client=http_client,
            timeout=timeout_ms,
            retry_options=retry_options,
        )

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        api_client = self._client._api_client  # pyright: ignore[reportPrivateUsage]
        api_client._async_httpx_client = http_client  # pyright: ignore[reportPrivateUsage]
        api_client._http_options.httpx_async_client = http_client  # pyright: ignore[reportPrivateUsage]


class GoogleProvider(BaseGoogleProvider):
    """Provider for the Gemini API (formerly Google AI Studio / Google GLA)."""

    @property
    def name(self) -> str:
        # Must not change: persisted in ModelMessage.provider_name and checked during history replay.
        return 'google'

    @overload
    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
        retry_options: HttpRetryOptions | None = None,
    ) -> None: ...

    @overload
    def __init__(self, *, client: Client) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Client | None = None,
        http_client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
        retry_options: HttpRetryOptions | None = None,
    ) -> None:
        """Create a new Google provider for the Gemini API.

        Args:
            api_key: The [API key](https://ai.google.dev/gemini-api/docs/api-key) to
                use for authentication. It can also be set via the `GOOGLE_API_KEY` environment variable.
            client: A pre-initialized client to use.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
            base_url: The base URL for the Gemini API.
            retry_options: HTTP retry options for transient errors (429, 5xx, etc.).
                See `google.genai.types.HttpRetryOptions` for available fields.
        """
        if client is not None:
            self._client = client
            return

        # NOTE: We are keeping GEMINI_API_KEY for backwards compatibility.
        api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if api_key is None:
            raise missing_api_key_error(
                'Set the `GOOGLE_API_KEY` environment variable or pass it via `GoogleProvider(api_key=...)`'
                ' to use the Gemini API.'
            )
        http_options = self._build_http_options(http_client=http_client, base_url=base_url, retry_options=retry_options)
        self._client = Client(vertexai=False, api_key=api_key, http_options=http_options)


GoogleCloudLocation = Literal[
    'asia-east1',
    'asia-east2',
    'asia-northeast1',
    'asia-northeast3',
    'asia-south1',
    'asia-southeast1',
    'australia-southeast1',
    'europe-central2',
    'europe-north1',
    'europe-southwest1',
    'europe-west1',
    'europe-west2',
    'europe-west3',
    'europe-west4',
    'europe-west6',
    'europe-west8',
    'europe-west9',
    'me-central1',
    'me-central2',
    'me-west1',
    'northamerica-northeast1',
    'southamerica-east1',
    'us-central1',
    'us-east1',
    'us-east4',
    'us-east5',
    'us-south1',
    'us-west1',
    'us-west4',
]
"""Regions available for Google Cloud.
More details [here](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#genai-locations).

This lists single-region values only. `GoogleCloudProvider` also accepts the `'global'` location and the
`'us'`/`'eu'` multi-regions (routed to the `aiplatform.{us,eu}.rep.googleapis.com` data-residency endpoints)
as separate union members on its `location` parameter.
"""
