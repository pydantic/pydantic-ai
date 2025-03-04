from __future__ import annotations as _annotations

import os

import httpx

from pydantic_ai.models import cached_async_http_client
from pydantic_ai.providers import Provider


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

    def __init__(self, api_key: str | None = None, http_client: httpx.AsyncClient | None = None) -> None:
        api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_GLA_API_KEY')
        if api_key is None:
            raise ValueError('API key is required for Google GLA provider')

        self._client = http_client or cached_async_http_client()
        self._client.base_url = self.base_url
        # https://cloud.google.com/docs/authentication/api-keys-use#using-with-rest
        self._client.headers['X-Goog-Api-Key'] = api_key
