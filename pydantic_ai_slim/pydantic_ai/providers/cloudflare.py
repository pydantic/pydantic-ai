from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import httpx2
from pydantic import TypeAdapter, ValidationError

from pydantic_ai import ModelProfile
from pydantic_ai._http import AsyncHTTPClient, create_async_httpx2_client
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.profiles.cloudflare import cloudflare_model_profile
from pydantic_ai.providers import Provider, missing_api_key_error

from ._cloudflare_http import send_request

if TYPE_CHECKING:
    from ._cloudflare_openai import CloudflareOpenAIProvider


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

    @cached_property
    def _openai_provider(self) -> CloudflareOpenAIProvider:
        # Native decision/image requests do not require the optional OpenAI SDK.
        from ._cloudflare_openai import CloudflareOpenAIProvider

        return CloudflareOpenAIProvider(self)

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
        self._discovery_cache: dict[str, list[str]] = {}
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

    async def discover(
        self,
        *,
        task: Literal['Text Generation', 'Text Embeddings', 'Text-to-Image'] = 'Text Generation',
        refresh: bool = False,
    ) -> list[str]:
        """Discover current native model IDs for an existing Pydantic AI interface.

        Results are cached on this provider; pass `refresh=True` to fetch them again.
        No model constructor calls this endpoint. Text-generation discovery excludes
        unsupported Clef variants, which use the decision protocol despite their catalog task.
        """
        if not refresh and task in self._discovery_cache:
            return list(self._discovery_cache[task])
        self._refresh_http_client()
        names: set[str] = set()
        page = 1
        while True:
            request = self.client.build_request(
                'GET',
                f'{self.base_url}/ai/models/search',
                params={'task': task, 'include_deprecated': 'false', 'per_page': 100, 'page': page},
                headers={'Authorization': f'Bearer {self.api_key}'},
            )
            response = await send_request(self.client, request, model_name='cloudflare:catalog')
            try:
                parsed = _catalog_adapter.validate_json(response.content)
            except ValidationError as e:
                raise UnexpectedModelBehavior(f'Invalid Cloudflare model catalog: {e}', response.text) from e
            for model in parsed.result:
                if model.task.name != task:
                    continue
                if model.name.startswith('@cf/cloudflare/clef') and model.name not in (
                    '@cf/cloudflare/clef',
                    '@cf/cloudflare/clef-flash',
                ):
                    continue
                names.add(model.name)
            total = parsed.result_info.total_count if parsed.result_info is not None else None
            if not parsed.result or (page * 100 >= total if total is not None else len(parsed.result) < 100):
                break
            page += 1
        self._discovery_cache[task] = sorted(names)
        return list(self._discovery_cache[task])

    def _set_http_client(self, http_client: AsyncHTTPClient) -> None:
        assert isinstance(http_client, httpx2.AsyncClient)
        self._client = http_client


@dataclass
class _CatalogTask:
    name: str


@dataclass
class _CatalogModel:
    name: str
    task: _CatalogTask


@dataclass
class _CatalogPageInfo:
    total_count: int | None = None


@dataclass
class _CatalogPage:
    result: list[_CatalogModel]
    result_info: _CatalogPageInfo | None = None


_catalog_adapter = TypeAdapter(_CatalogPage)
