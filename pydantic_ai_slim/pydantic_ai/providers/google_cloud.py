from __future__ import annotations as _annotations

from typing import Literal

import httpx

try:
    from google.auth.credentials import Credentials
    from google.genai.client import Client

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
        """
        self._init_client(
            vertexai=True,
            api_key=api_key,
            credentials=credentials,
            project=project,
            location=location,
            client=client,
            http_client=http_client,
            base_url=base_url,
        )
