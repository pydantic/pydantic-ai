from __future__ import annotations as _annotations

import os

import httpx
from typing_extensions import deprecated

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
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

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
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
                ' to use the Google GLA provider.'
            )

        self._api_key = api_key

        if http_client is None:
            http_client = create_async_http_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_http_client
        self._client = http_client
        self._client.base_url = self.base_url
        # https://cloud.google.com/docs/authentication/api-keys-use#using-with-rest
        self._client.headers['X-Goog-Api-Key'] = api_key

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client = http_client
        self._client.base_url = self.base_url
        self._client.headers['X-Goog-Api-Key'] = self._api_key
