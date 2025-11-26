from __future__ import annotations as _annotations

import os

import httpx
from typing_extensions import deprecated

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.providers import Provider


@deprecated('`GoogleGLAProvider` is deprecated, use `GoogleProvider` with `GoogleModel` instead.')
class GoogleGLAProvider(Provider[httpx.AsyncClient]):
    """Provider for Google Generative Language AI API."""

    @property
    def name(self):
        return 'google-gla'

    @property
    def base_url(self) -> str:
        return 'https://generativelanguage.googleapis.com/v1beta/models/'

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return google_model_profile(model_name)

    def __init__(self, api_key: str | None = None, http_client: httpx.AsyncClient | None = None) -> None:
        """Create a new Google GLA provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `GEMINI_API_KEY` environment variable
                will be used if available.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise UserError(
                'Set the `GEMINI_API_KEY` environment variable or pass it via `GoogleGLAProvider(api_key=...)`'
                'to use the Google GLA provider.'
            )

        self._client = http_client or cached_async_http_client(provider='google-gla')
        self._client.base_url = self.base_url
        # https://cloud.google.com/docs/authentication/api-keys-use#using-with-rest
        self._client.headers['X-Goog-Api-Key'] = api_key
