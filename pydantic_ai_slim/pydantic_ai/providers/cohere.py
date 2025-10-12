from __future__ import annotations as _annotations

import os

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.providers import Provider

try:
    from cohere import AsyncClientV2
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `cohere` package to use the Cohere provider, '
        'you can use the `cohere` optional group — `pip install "pydantic-ai-slim[cohere]"`'
    ) from _import_error


class CohereProvider(Provider[AsyncClientV2]):
    """Provider for Cohere API."""

    @property
    def name(self) -> str:
        return 'cohere'

    @property
    def base_url(self) -> str:
        client_wrapper = self.client._client_wrapper  # type: ignore
        return str(client_wrapper.get_base_url())

    @property
    def client(self) -> AsyncClientV2:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return cohere_model_profile(model_name)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        cohere_client: AsyncClientV2 | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Cohere provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `CO_API_KEY` environment variable
                will be used if available.
            cohere_client: An existing
                [AsyncClientV2](https://github.com/cohere-ai/cohere-python)
                client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if cohere_client is not None:
            assert http_client is None, 'Cannot provide both `cohere_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `cohere_client` and `api_key`'
            self._client = cohere_client
        else:
            api_key = api_key or os.getenv('CO_API_KEY')
            if not api_key:
                raise UserError(
                    'Set the `CO_API_KEY` environment variable or pass it via `CohereProvider(api_key=...)`'
                    'to use the Cohere provider.'
                )

            base_url = os.getenv('CO_BASE_URL')
            if http_client is not None:
                self._client = AsyncClientV2(api_key=api_key, httpx_client=http_client, base_url=base_url)
            else:
                http_client = cached_async_http_client(provider='cohere')
                self._client = AsyncClientV2(api_key=api_key, httpx_client=http_client, base_url=base_url)
