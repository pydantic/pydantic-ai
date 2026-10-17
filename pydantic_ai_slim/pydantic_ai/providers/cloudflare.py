from __future__ import annotations

import os

import httpx2

from pydantic_ai import ModelProfile
from pydantic_ai._http import AsyncHTTPClient, create_async_httpx2_client
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.cloudflare import cloudflare_model_profile
from pydantic_ai.providers import Provider, missing_api_key_error


class CloudflareProvider(Provider[httpx2.AsyncClient]):
    """Provider for the account-scoped [Cloudflare Workers AI API](https://developers.cloudflare.com/workers-ai/)."""

    @property
    def name(self) -> str:
        return 'cloudflare'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> httpx2.AsyncClient:
        return self._client

    @property
    def api_key(self) -> str:
        """The API token sent as a bearer token."""
        return self._api_key

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return cloudflare_model_profile(model_name)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        account_id: str | None = None,
        base_url: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        """Create a Cloudflare provider.

        Args:
            api_key: The API token. Defaults to the `CLOUDFLARE_API_TOKEN` environment variable.
            account_id: The Cloudflare account ID. Defaults to the `CLOUDFLARE_ACCOUNT_ID` environment variable.
                Only used when `base_url` is not provided.
            base_url: An account-scoped URL, such as `https://api.cloudflare.com/client/v4/accounts/<account_id>`.
                Overrides `account_id` and its environment variable.
            http_client: An existing `httpx2.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('CLOUDFLARE_API_TOKEN')
        if not api_key:
            raise missing_api_key_error(
                'Set the `CLOUDFLARE_API_TOKEN` environment variable or pass it via '
                '`CloudflareProvider(api_key=...)` to use the Cloudflare provider.'
            )
        self._api_key = api_key
        if not base_url:
            account_id = account_id or os.getenv('CLOUDFLARE_ACCOUNT_ID')
            if not account_id:
                raise UserError(
                    'Set the `CLOUDFLARE_ACCOUNT_ID` environment variable or pass `account_id` or an account-scoped '
                    '`base_url` to `CloudflareProvider`.'
                )
            base_url = f'https://api.cloudflare.com/client/v4/accounts/{account_id}'
        self._base_url = base_url.rstrip('/')
        if http_client is None:
            http_client = create_async_httpx2_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_httpx2_client
        self._client = http_client

    def _set_http_client(self, http_client: AsyncHTTPClient) -> None:
        assert isinstance(http_client, httpx2.AsyncClient)
        self._client = http_client
