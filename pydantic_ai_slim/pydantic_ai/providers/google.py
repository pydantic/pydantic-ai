from __future__ import annotations as _annotations

from pydantic_ai.providers import Provider

try:
    from google import genai
except ImportError as _import_error:
    raise ImportError(
        'Please install the `google-genai` package to use the Google Vertex AI provider, '
        'you can use the `gemini` optional group — `pip install "pydantic-ai-slim[gemini]"`'
    ) from _import_error


class GoogleProvider(Provider[genai.Client]):
    """Provider for Google API."""

    @property
    def name(self) -> str:
        return self._name

    @property
    def base_url(self) -> str:
        return str(self._client._api_client._httpx_client.base_url)  # type: ignore[reportPrivateUsage]

    @property
    def client(self) -> genai.Client:
        return self._client

    def __init__(self, vertexai: bool = False, client: genai.Client | None = None) -> None:
        """Create a new Google GLA provider."""
        self._name = 'google-vertex' if vertexai else 'google-gla'
        self._client = client or genai.Client(vertexai=vertexai)
