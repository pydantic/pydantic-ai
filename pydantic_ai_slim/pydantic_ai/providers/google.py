from __future__ import annotations as _annotations

import os
from typing import Literal, overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client, get_user_agent
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


class GoogleProvider(Provider[Client]):
    """Provider for Google."""

    @property
    def name(self) -> str:
        return 'google-vertex' if self._client._api_client.vertexai else 'google-gla'  # type: ignore[reportPrivateUsage]

    @property
    def base_url(self) -> str:
        return str(self._client._api_client._http_options.base_url)  # type: ignore[reportPrivateUsage]

    @property
    def client(self) -> Client:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return google_model_profile(model_name)

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
        location: VertexAILocation | Literal['global'] | str | None = None,
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
        location: VertexAILocation | Literal['global'] | str | None = None,
        vertexai: bool | None = None,
        client: Client | None = None,
        http_client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
    ) -> None:
        """Create a new Google provider.

        Args:
            api_key: The `API key <https://ai.google.dev/gemini-api/docs/api-key>`_ to
                use for authentication. It can also be set via the `GOOGLE_API_KEY` environment variable.
            credentials: The credentials to use for authentication when calling the Vertex AI APIs. Credentials can be
                obtained from environment variables and default credentials. For more information, see Set up
                Application Default Credentials. Applies to the Vertex AI API only.
            project: The Google Cloud project ID to use for quota. Can be obtained from environment variables
                (for example, GOOGLE_CLOUD_PROJECT). Applies to the Vertex AI API only.
            location: The location to send API requests to (for example, us-central1). Can be obtained from environment variables.
                Applies to the Vertex AI API only.
            vertexai: Force the use of the Vertex AI API. If `False`, the Google Generative Language API will be used.
                Defaults to `False` unless `location`, `project`, or `credentials` are provided.
            client: A pre-initialized client to use.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
            base_url: The base URL for the Google API.
        """
        if client is None:
            # NOTE: We are keeping GEMINI_API_KEY for backwards compatibility.
            api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')

            vertex_ai_args_used = bool(location or project or credentials)
            if vertexai is None:
                vertexai = vertex_ai_args_used

            http_client = http_client or cached_async_http_client(
                provider='google-vertex' if vertexai else 'google-gla'
            )
            http_options = HttpOptions(
                base_url=base_url,
                headers={'User-Agent': get_user_agent()},
                httpx_async_client=http_client,
            )
            if not vertexai:
                if api_key is None:
                    raise UserError(
                        'Set the `GOOGLE_API_KEY` environment variable or pass it via `GoogleProvider(api_key=...)`'
                        'to use the Google Generative Language API.'
                    )
                self._client = Client(vertexai=False, api_key=api_key, http_options=http_options)
            else:
                if vertex_ai_args_used:
                    api_key = None

                if api_key is None:
                    project = project or os.getenv('GOOGLE_CLOUD_PROJECT')
                    # From https://github.com/pydantic/pydantic-ai/pull/2031/files#r2169682149:
                    # Currently `us-central1` supports the most models by far of any region including `global`, but not
                    # all of them. `us-central1` has all google models but is missing some Anthropic partner models,
                    # which use `us-east5` instead. `global` has fewer models but higher availability.
                    # For more details, check: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
                    location = location or os.getenv('GOOGLE_CLOUD_LOCATION') or 'us-central1'

                self._client = Client(
                    vertexai=True,
                    api_key=api_key,
                    project=project,
                    location=location,
                    credentials=credentials,
                    http_options=http_options,
                )
        else:
            self._client = client  # pragma: no cover


VertexAILocation = Literal[
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
"""Regions available for Vertex AI.
More details [here](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#genai-locations).
"""
