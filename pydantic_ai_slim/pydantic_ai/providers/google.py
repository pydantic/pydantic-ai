from __future__ import annotations as _annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, overload

import httpx2

from pydantic_ai import ModelProfile
from pydantic_ai._http import (
    DEFAULT_HTTP_TIMEOUT,
    AsyncHTTPClient,
    create_async_httpx2_client,
    warn_if_legacy_httpx_client,
)
from pydantic_ai.models import get_user_agent
from pydantic_ai.profiles.google import google_model_profile, google_realtime_model_profile
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


def _create_google_httpx2_client() -> httpx2.AsyncClient:
    """Create the HTTP client a Google provider owns, with a 20-minute timeout instead of `DEFAULT_HTTP_TIMEOUT`.

    Google [recommends](https://ai.google.dev/gemini-api/docs/generate-content/long-decode) a 20-minute read timeout
    for long decode. The Gateway's Google route creates this client too.
    """
    return create_async_httpx2_client(timeout=httpx2.Timeout(20 * 60, connect=5))


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

    _canonical_system_follows_client = True

    @property
    def canonical_system(self) -> str:
        """`'google-cloud'` with a Google Cloud (Vertex AI) client, `'google'` with a Gemini API client.

        Read off the client rather than `name`, as either provider accepts either client. The two transports
        reject each other's thought signatures.
        """
        return 'google-cloud' if self._client.vertexai else 'google'

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
        return google_realtime_model_profile(model_name)

    def _build_http_options(
        self,
        *,
        http_client: AsyncHTTPClient | None,
        base_url: str | None,
        retry_options: HttpRetryOptions | None = None,
    ) -> HttpOptions:
        """Build `HttpOptions` and record ownership of the httpx client if we created it.

        Subclasses call this before constructing their `Client(...)` to keep timeout / user-agent /
        ownership wiring consistent.
        """
        if http_client is None:
            http_client = _create_google_httpx2_client()
            self._own_http_client = http_client
            self._http_client_factory = _create_google_httpx2_client
        else:
            # 3 frames up from the helper: this method, the provider `__init__` calling it, and the
            # user's `GoogleProvider(...)` call, which is where the warning should land.
            warn_if_legacy_httpx_client(http_client, consumer='Google providers', stacklevel=3)
        # google-genai's `HttpOptions.timeout` defaults to None, which makes the SDK pass
        # `timeout=None` to httpx and override any timeout on the supplied client. Pin the timeout
        # here (ms) so requests actually time out. The SDK sends it as a scalar, which would set the
        # connect and pool timeouts too; a client we created caps those at its own (see `_http.py`).
        timeout_seconds = http_client.timeout.read or DEFAULT_HTTP_TIMEOUT
        timeout_ms = int(timeout_seconds * 1000)
        return HttpOptions(
            base_url=base_url,
            headers={'User-Agent': get_user_agent()},
            httpx_async_client=http_client,
            timeout=timeout_ms,
            retry_options=retry_options,
        )

    def _set_http_client(self, http_client: AsyncHTTPClient) -> None:
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
        api_key: str | None = None,
        http_client: AsyncHTTPClient | None = None,
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
        http_client: AsyncHTTPClient | None = None,
        base_url: str | None = None,
        retry_options: HttpRetryOptions | None = None,
    ) -> None:
        """Create a new Google provider for the Gemini API.

        Args:
            api_key: The [API key](https://ai.google.dev/gemini-api/docs/api-key) to
                use for authentication. It can also be set via the `GOOGLE_API_KEY` environment variable,
                or the legacy `GEMINI_API_KEY` environment variable (`GOOGLE_API_KEY` takes precedence).
            client: A pre-initialized client to use. Stored as-is, so a client built for Google Cloud
                (`Client(vertexai=True, ...)`, or its current spelling `Client(enterprise=True, ...)`)
                is used on that transport even though `name` stays `'google'`. `GoogleModel` reads the
                transport off the client rather than the provider name, so it routes correctly either way.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
            base_url: The base URL for the Gemini API.
            retry_options: HTTP retry options for transient errors (429, 5xx, etc.).
                See `google.genai.types.HttpRetryOptions` for available fields.
        """
        # The credentials the provider built its client from, for `_client_for_api_version`; unknown for a
        # client the caller built.
        self._api_key: str | None = None
        self._base_url = base_url
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
        self._api_key = api_key
        http_options = self._build_http_options(http_client=http_client, base_url=base_url, retry_options=retry_options)
        self._client = Client(vertexai=False, api_key=api_key, http_options=http_options)

    def _client_for_api_version(self, api_version: str) -> Client:
        """A new client on `api_version` with this provider's credentials.

        For a request the provider's own client can't make on its API version (Gemini Live's `proactivity`
        is `v1alpha`-only). Only for a provider that built its client from an API key (`_api_key` is set):
        a client the caller built can't be reproduced. The caller owns and closes the new client.
        """
        assert self._api_key is not None
        # Its own `HttpOptions` and HTTP clients: nothing it holds is shared with the provider's client.
        http_options = HttpOptions(
            base_url=self._base_url, api_version=api_version, headers={'User-Agent': get_user_agent()}
        )
        return Client(vertexai=False, api_key=self._api_key, http_options=http_options)


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
