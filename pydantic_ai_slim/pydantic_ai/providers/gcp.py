from __future__ import annotations as _annotations

import warnings
from typing import Literal

import httpx

from pydantic_ai.providers.google import GoogleProvider, VertexAILocation

try:
    from google.auth.credentials import Credentials
    from google.genai.client import Client
except ImportError as _import_error:
    raise ImportError(
        'Please install the `google-genai` package to use the GCP provider, '
        'you can use the `google` optional group — `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error


class GCPProvider(GoogleProvider):
    """Provider for Google Cloud Platform (Vertex AI)."""

    @property
    def name(self) -> str:
        return 'gcp'

    def __init__(
        self,
        *,
        credentials: Credentials | None = None,
        project: str | None = None,
        location: VertexAILocation | Literal['global'] | str | None = None,
        client: Client | None = None,
        http_client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
    ) -> None:
        """Create a new GCP provider.

        Args:
            credentials: The credentials to use for authentication when calling the Vertex AI APIs. Credentials can be
                obtained from environment variables and default credentials. For more information, see
                [Set up Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials).
            project: The Google Cloud project ID to use for quota. Can be obtained from environment variables
                (for example, `GOOGLE_CLOUD_PROJECT`).
            location: The location to send API requests to (for example, `us-central1`). Can be obtained from
                the `GOOGLE_CLOUD_LOCATION` environment variable.
            client: A pre-initialized client to use.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
            base_url: The base URL for the Vertex AI API.
        """
        with warnings.catch_warnings():
            # Internal forward; the user-visible deprecation is the `'google-vertex:'`
            # prefix or the `GoogleProvider(vertexai=...)` kwarg, not this delegation.
            warnings.simplefilter('ignore', DeprecationWarning)
            super().__init__(  # pyright: ignore[reportCallIssue]
                vertexai=True,
                credentials=credentials,
                project=project,
                location=location,
                client=client,
                http_client=http_client,
                base_url=base_url,
            )
