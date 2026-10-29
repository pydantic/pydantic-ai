from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.profiles.nace import nace_model_profile
from pydantic_ai.providers import Provider, missing_api_key_error

try:
    from nace_sdk import AsyncNaceClient, RetryPolicy
except ImportError as _import_error:
    raise ImportError(
        'Please install the `nace-sdk` package to use the Nace provider, '
        'you can use the `nace` optional group — `pip install "pydantic-ai-slim[nace]"`'
    ) from _import_error

_DEFAULT_BASE_URL = 'https://console.nace.ai'


class NaceProvider(Provider[AsyncNaceClient]):
    """Provider for the [Nace](https://console.nace.ai/docs) API, which serves the Drex models.

    The SDK speaks `httpx`, not the `httpx2` client this process injects into other providers, so this provider does
    not replace the SDK's HTTP client after construction.
    """

    @property
    def name(self) -> str:
        return 'nace'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncNaceClient:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return nace_model_profile(model_name)

    @overload
    def __init__(self, *, nace_client: AsyncNaceClient) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        nace_client: AsyncNaceClient | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Nace provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `NACE_API_KEY` environment
                variable will be used if available.
            base_url: The base URL for the Nace API, if not provided, the `NACE_BASE_URL` environment
                variable will be used if available, and `https://console.nace.ai` otherwise.
            nace_client: An existing `AsyncNaceClient` to use. If provided, `api_key`, `base_url` and
                `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if nace_client is not None:
            assert api_key is None, 'Cannot provide both `nace_client` and `api_key`'
            assert base_url is None, 'Cannot provide both `nace_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `nace_client` and `http_client`'
            self._client = nace_client
            self._base_url = nace_client.base_url
            return

        api_key = api_key or os.getenv('NACE_API_KEY')
        if not api_key:
            raise missing_api_key_error(
                'Set the `NACE_API_KEY` environment variable or pass it via `NaceProvider(api_key=...)`'
                ' to use the Nace provider.'
            )
        self._base_url = base_url or os.getenv('NACE_BASE_URL') or _DEFAULT_BASE_URL
        self._client = AsyncNaceClient(
            api_key=api_key,
            base_url=self._base_url,
            http_client=http_client,
            client_info='pydantic-ai-nace/0.1.0',
            timeout=60,
            retry=RetryPolicy(timeout=180.0),
        )
