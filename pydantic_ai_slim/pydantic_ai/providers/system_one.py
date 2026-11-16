from __future__ import annotations as _annotations

import os

import httpx2

from pydantic_ai import ModelProfile
from pydantic_ai._http import AsyncHTTPClient, create_async_httpx2_client
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.decision import decision_model_profile
from pydantic_ai.providers import Provider


class SystemOneProvider(Provider[httpx2.AsyncClient]):
    """Provider for the `POST /v1/systemone` decisions API, at a URL of your choosing.

    [`SystemOneModel`][pydantic_ai.models.system_one.SystemOneModel] speaks the API; this provider says where it is
    and how to authenticate. Decision models such as [Contrastive Language Models](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) and
    [Laya](https://huggingface.co/convaiinnovations/laya) are available over this API, and
    [Ollama](https://ollama.com) runs decision models locally over it at `http://localhost:11434`.
    """

    @property
    def name(self) -> str:
        return 'system-one'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> httpx2.AsyncClient:
        return self._client

    @property
    def api_key(self) -> str | None:
        """The key sent as a bearer token, if the API needs one."""
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
        """Create a new System One provider.

        Args:
            base_url: The URL of the API, with or without a trailing `/v1`. If not provided,
                the `SYSTEM_ONE_BASE_URL` environment variable is used.
            api_key: The API key, sent as a bearer token. If not provided, the
                `SYSTEM_ONE_API_KEY` environment variable is used if set.
            http_client: An existing `httpx2.AsyncClient` to use for making HTTP requests.
        """
        base_url = base_url or os.getenv('SYSTEM_ONE_BASE_URL')
        if not base_url:
            raise UserError(
                'Set the `SYSTEM_ONE_BASE_URL` environment variable or pass it via `SystemOneProvider(base_url=...)` '
                'to point the System One provider at the API.'
            )
        self._base_url = base_url.rstrip('/')
        self._api_key = api_key or os.getenv('SYSTEM_ONE_API_KEY')
        if http_client is None:
            http_client = create_async_httpx2_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_httpx2_client
        self._client = http_client

    def _set_http_client(self, http_client: AsyncHTTPClient) -> None:
        assert isinstance(http_client, httpx2.AsyncClient)
        self._client = http_client
