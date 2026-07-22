from __future__ import annotations as _annotations

import os
from typing import Literal

import httpx

try:
    from google.auth.credentials import Credentials
    from google.genai.client import Client
    from google.genai.types import HttpRetryOptions

    from pydantic_ai.providers.google import BaseGoogleProvider, GoogleCloudLocation
except ImportError as _import_error:
    raise ImportError(
        'Please install the `google-genai` package to use the Google Cloud provider, '
        'you can use the `google` optional group — `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error


class GoogleCloudProvider(BaseGoogleProvider):
    """Provider for Google Cloud (formerly known as Vertex AI)."""

    @property
    def name(self) -> str:
        return 'google-cloud'

    def __init__(
        self,
        *,
        api_key: str | None = None,
        credentials: Credentials | None = None,
        project: str | None = None,
        location: GoogleCloudLocation | Literal['global'] | str | None = None,
        client: Client | None = None,
        http_client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
        retry_options: HttpRetryOptions | None = None,
    ) -> None:
        """Create a new Google Cloud provider.

        Args:
            api_key: The [Vertex AI Express Mode API key](https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys?usertype=expressmode)
                to use for authentication. It can also be set via the `GOOGLE_API_KEY` environment variable.
                Cannot be combined with `credentials`/`project`/`location` (those use Application Default Credentials).
            credentials: The credentials to use for authentication when calling the Google Cloud APIs. Credentials can
                be obtained from environment variables and default credentials. For more information, see
                [Set up Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials).
            project: The Google Cloud project ID to use for quota. Can be obtained from environment variables
                (for example, `GOOGLE_CLOUD_PROJECT`).
            location: The location to send API requests to (for example, `us-central1`). Can be obtained from
                the `GOOGLE_CLOUD_LOCATION` environment variable.
            client: A pre-initialized client to use.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
            base_url: The base URL for the Google Cloud API.
            retry_options: HTTP retry options for transient errors (429, 5xx, etc.).
                See `google.genai.types.HttpRetryOptions` for available fields.
        """
        if client is not None:
            self._client = client
            return

        # NOTE: We are keeping GEMINI_API_KEY for backwards compatibility.
        api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')

        # ADC kwargs take precedence over API-key auth. With none provided and only an api_key,
        # the SDK uses Vertex AI Express Mode.
        if credentials is not None or project is not None or location is not None:
            api_key = None

        if api_key is None:
            project = project or os.getenv('GOOGLE_CLOUD_PROJECT')
            # From https://github.com/pydantic/pydantic-ai/pull/2031/files#r2169682149:
            # Currently `us-central1` supports the most models by far of any region including `global`, but not
            # all of them. `us-central1` has all google models but is missing some Anthropic partner models,
            # which use `us-east5` instead. `global` has fewer models but higher availability.
            # For more details, check: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
            location = location or os.getenv('GOOGLE_CLOUD_LOCATION') or 'us-central1'

        http_options = self._build_http_options(http_client=http_client, base_url=base_url, retry_options=retry_options)
        self._client = Client(
            vertexai=True,
            api_key=api_key,
            project=project,
            location=location,
            credentials=credentials,
            http_options=http_options,
        )
