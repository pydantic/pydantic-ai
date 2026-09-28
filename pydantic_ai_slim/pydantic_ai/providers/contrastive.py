from __future__ import annotations as _annotations

import os

import httpx2

from pydantic_ai import ModelProfile
from pydantic_ai._http import AsyncHTTPClient, create_async_httpx2_client
from pydantic_ai.profiles.decision import decision_model_profile
from pydantic_ai.providers import Provider

DEFAULT_BASE_URL = 'http://127.0.0.1:8700'
"""Where `clm-serve` listens by default."""


class ContrastiveProvider(Provider[httpx2.AsyncClient]):
    """Provider for a [CLM](https://github.com/Contrastive-LM/CLM) server, which serves Contrastive Language Models.

    `clm-serve`, from the `contrastive-lm` package, serves the open-weight checkpoints such as
    [`Contrastive-LM/CLM-v0.1-8B`](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) on your own hardware.
    """

    @property
    def name(self) -> str:
        return 'contrastive'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> httpx2.AsyncClient:
        return self._client

    @property
    def api_key(self) -> str | None:
        """The key sent as a bearer token, if the server was started with `CLM_API_KEY` set."""
        return self._api_key

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return decision_model_profile(model_name)

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        """Create a new CLM provider.

        Args:
            base_url: The URL of the CLM server. If not provided, the `CLM_BASE_URL` environment variable is used
                if set, and `http://127.0.0.1:8700`, where `clm-serve` listens by default, otherwise.
            api_key: The API key the server was started with. If not provided, the `CLM_API_KEY` environment
                variable is used if set. `clm-serve` needs no key unless it was started with one.
            http_client: An existing `httpx2.AsyncClient` to use for making HTTP requests.
        """
        self._base_url = (base_url or os.getenv('CLM_BASE_URL') or DEFAULT_BASE_URL).rstrip('/')
        self._api_key = api_key or os.getenv('CLM_API_KEY')
        if http_client is None:
            http_client = create_async_httpx2_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_httpx2_client
        self._client = http_client

    def _set_http_client(self, http_client: AsyncHTTPClient) -> None:
        assert isinstance(http_client, httpx2.AsyncClient)
        self._client = http_client
