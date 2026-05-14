from __future__ import annotations as _annotations

import os
import warnings
from abc import ABC, abstractmethod
from typing import Literal, overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import DEFAULT_HTTP_TIMEOUT, create_async_http_client, get_user_agent
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.providers import Provider

try:
    from google.auth.credentials import Credentials
    from google.genai.client import Client
    from google.genai.types import HttpOptions
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

    def _build_http_options(
        self,
        *,
        http_client: httpx.AsyncClient | None,
        base_url: str | None,
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
        )

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        api_client = self._client._api_client  # pyright: ignore[reportPrivateUsage]
        api_client._async_httpx_client = http_client  # pyright: ignore[reportPrivateUsage]
        api_client._http_options.httpx_async_client = http_client  # pyright: ignore[reportPrivateUsage]


class GoogleProvider(BaseGoogleProvider):
    """Provider for the Gemini API (formerly Google AI Studio / Google GLA)."""

    @property
    def name(self) -> str:
        return 'google'

    @overload
    def __init__(
        self, *, api_key: str, http_client: httpx.AsyncClient | None = None, base_url: str | None = None
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        credentials: Credentials | None = None,
        project: str | None = None,
        location: GoogleCloudLocation | Literal['global'] | str | None = None,
        http_client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
    ) -> None: ...

    @overload
    def __init__(self, *, client: Client) -> None: ...

    @overload
    def __init__(
        self,
        *,
        vertexai: bool = False,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        credentials: Credentials | None = None,
        project: str | None = None,
        location: GoogleCloudLocation | Literal['global'] | str | None = None,
        vertexai: bool | None = None,
        client: Client | None = None,
        http_client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
    ) -> None:
        """Create a new Google provider for the Gemini API.

        Args:
            api_key: The [API key](https://ai.google.dev/gemini-api/docs/api-key) to
                use for authentication. It can also be set via the `GOOGLE_API_KEY` environment variable.
            credentials: Deprecated. Use [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] instead.
            project: Deprecated. Use [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] instead.
            location: Deprecated. Use [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] instead.
            vertexai: Deprecated. Use [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] instead.
            client: A pre-initialized client to use.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
            base_url: The base URL for the Gemini API.
        """
        if client is not None:
            self._client = client
            return

        vertex_ai_args_used = bool(location or project or credentials)
        if vertexai is True or vertex_ai_args_used:
            # Cloud-args usage on `GoogleProvider` is deprecated in 1.x; forward to `GoogleCloudProvider`
            # so the Cloud-specific construction logic stays in one place.
            warnings.warn(
                '`GoogleProvider(...)` with Google Cloud (formerly known as Vertex AI) arguments '
                '(`vertexai=True`, `location=`, `project=`, or `credentials=`) is deprecated and will '
                'be removed in v2.0. Use `GoogleCloudProvider(...)` instead, which only accepts the '
                'Google Cloud arguments.',
                PydanticAIDeprecationWarning,
                stacklevel=2,
            )
            from .google_cloud import GoogleCloudProvider

            delegate = GoogleCloudProvider(
                api_key=api_key,
                credentials=credentials,
                project=project,
                location=location,
                http_client=http_client,
                base_url=base_url,
            )
            self._client = delegate._client
            self._own_http_client = delegate._own_http_client
            self._http_client_factory = delegate._http_client_factory
            return

        if vertexai is False:
            warnings.warn(
                '`GoogleProvider(vertexai=False, ...)` is redundant and will be removed in v2.0; '
                "drop the explicit `vertexai=False` (it's the default).",
                PydanticAIDeprecationWarning,
                stacklevel=2,
            )

        # NOTE: We are keeping GEMINI_API_KEY for backwards compatibility.
        api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if api_key is None:
            raise UserError(
                'Set the `GOOGLE_API_KEY` environment variable or pass it via `GoogleProvider(api_key=...)`'
                ' to use the Gemini API.'
            )
        http_options = self._build_http_options(http_client=http_client, base_url=base_url)
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
"""

VertexAILocation = GoogleCloudLocation
"""Deprecated alias for `GoogleCloudLocation`."""
