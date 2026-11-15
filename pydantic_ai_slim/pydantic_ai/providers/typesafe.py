from __future__ import annotations as _annotations

import os
from typing import overload

import httpx2

from pydantic_ai import ModelProfile
from pydantic_ai._http import AsyncHTTPClient, create_async_httpx2_client
from pydantic_ai.profiles.typesafe import typesafe_model_profile
from pydantic_ai.providers import Provider, missing_api_key_error

try:
    from typesafe_sdk import AsyncTypeSafeClient
    from typesafe_sdk.constants import DEFAULT_BASE_URL
except ImportError as _import_error:
    raise ImportError(
        'Please install the `typesafe-sdk` package to use the TypeSafe provider, '
        'you can use the `typesafe` optional group — `pip install "pydantic-ai-slim[typesafe]"`'
    ) from _import_error


class TypeSafeProvider(Provider[AsyncTypeSafeClient]):
    """Provider for the [TypeSafe](https://typesafe.ai) API, which serves the Jev models."""

    @property
    def name(self) -> str:
        return 'typesafe'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncTypeSafeClient:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return typesafe_model_profile(model_name)

    @overload
    def __init__(self, *, typesafe_client: AsyncTypeSafeClient) -> None: ...

    @overload
    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None, http_client: httpx2.AsyncClient | None = None
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        typesafe_client: AsyncTypeSafeClient | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        """Create a new TypeSafe provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `TYPESAFE_API_KEY` environment
                variable will be used if available.
            base_url: The base URL for the TypeSafe API, if not provided, the `TYPESAFE_BASE_URL` environment
                variable will be used if available, and `https://api.typesafe.ai` otherwise.
            typesafe_client: An existing `AsyncTypeSafeClient` to use. If provided, `api_key`, `base_url` and
                `http_client` must be `None`.
            http_client: An existing `httpx2.AsyncClient` to use for making HTTP requests.
        """
        if typesafe_client is not None:
            assert api_key is None, 'Cannot provide both `typesafe_client` and `api_key`'
            assert base_url is None, 'Cannot provide both `typesafe_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `typesafe_client` and `http_client`'
            self._client = typesafe_client
            # The SDK exposes neither its base URL nor its HTTP client publicly.
            self._base_url = typesafe_client._config.base_url  # pyright: ignore[reportPrivateUsage]
            return

        api_key = api_key or os.getenv('TYPESAFE_API_KEY')
        if not api_key:
            raise missing_api_key_error(
                'Set the `TYPESAFE_API_KEY` environment variable or pass it via `TypeSafeProvider(api_key=...)`'
                ' to use the TypeSafe provider.'
            )
        self._base_url = base_url or os.getenv('TYPESAFE_BASE_URL') or DEFAULT_BASE_URL

        if http_client is None:
            http_client = create_async_httpx2_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_httpx2_client
        self._client = AsyncTypeSafeClient(api_key=api_key, base_url=self._base_url, http_client=http_client)

    def _set_http_client(self, http_client: AsyncHTTPClient) -> None:
        assert isinstance(http_client, httpx2.AsyncClient)
        self._client._http_client = http_client  # pyright: ignore[reportPrivateUsage]
