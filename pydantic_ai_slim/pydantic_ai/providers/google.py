from __future__ import annotations as _annotations

import os
from typing import Literal, overload

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import get_user_agent
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.providers import Provider

try:
    from google import genai
    from google.auth.credentials import Credentials
except ImportError as _import_error:
    raise ImportError(
        'Please install the `google-genai` package to use the Google provider, '
        'you can use the `google` optional group — `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error


class GoogleProvider(Provider[genai.Client]):
    """Provider for Google."""

    @property
    def name(self) -> str:
        return 'google-vertex' if self._client._api_client.vertexai else 'google-gla'  # type: ignore[reportPrivateUsage]

    @property
    def base_url(self) -> str:
        return str(self._client._api_client._http_options.base_url)  # type: ignore[reportPrivateUsage]

    @property
    def client(self) -> genai.Client:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return google_model_profile(model_name)

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(
        self,
        *,
        credentials: Credentials | None = None,
        project: str | None = None,
        location: VertexAILocation | Literal['global'] | None = None,
    ) -> None: ...

    @overload
    def __init__(self, *, client: genai.Client) -> None: ...

    @overload
    def __init__(self, *, vertexai: bool = False) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        credentials: Credentials | None = None,
        project: str | None = None,
        location: VertexAILocation | Literal['global'] | None = None,
        client: genai.Client | None = None,
        vertexai: bool | None = None,
    ) -> None:
        """Create a new Google provider.

        Args:
            api_key: The `API key <https://ai.google.dev/gemini-api/docs/api-key>`_ to
                use for authentication. It can also be set via the `GOOGLE_API_KEY` environment variable.
                Applies to the Gemini Developer API only.
            credentials: The credentials to use for authentication when calling the Vertex AI APIs. Credentials can be
                obtained from environment variables and default credentials. For more information, see Set up
                Application Default Credentials. Applies to the Vertex AI API only.
            project: The Google Cloud project ID to use for quota. Can be obtained from environment variables
                (for example, GOOGLE_CLOUD_PROJECT). Applies to the Vertex AI API only.
            location: The location to send API requests to (for example, us-central1). Can be obtained from environment variables.
                Applies to the Vertex AI API only.
            client: A pre-initialized client to use.
            vertexai: Force the use of the Vertex AI API. If `False`, the Google Generative Language API will be used.
                Defaults to `False`.
        """
        if client is None:
            # NOTE: We are keeping GEMINI_API_KEY for backwards compatibility.
            api_key = api_key or os.environ.get('GOOGLE_API_KEY')

            if vertexai is None:  # pragma: lax no cover
                vertexai = bool(location or project or credentials)

            if not vertexai:
                if api_key is None:
                    raise UserError(  # pragma: no cover
                        'Set the `GOOGLE_API_KEY` environment variable or pass it via `GoogleProvider(api_key=...)`'
                        'to use the Google Generative Language API.'
                    )
                self._client = genai.Client(
                    vertexai=vertexai,
                    api_key=api_key,
                    http_options={'headers': {'User-Agent': get_user_agent()}},
                )
            else:
                self._client = genai.Client(
                    vertexai=vertexai,
                    project=project or os.environ.get('GOOGLE_CLOUD_PROJECT'),
                    location=location or os.environ.get('GOOGLE_CLOUD_LOCATION'),
                    credentials=credentials,
                    http_options={'headers': {'User-Agent': get_user_agent()}},
                )
        else:
            self._client = client  # pragma: lax no cover


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
