from __future__ import annotations as _annotations

import os
import re
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers import Provider, missing_api_key_error

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Bedrock Mantle provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


_AWS_REGION_PATTERN = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)+$')


class BedrockMantleProvider(Provider[AsyncOpenAI]):
    """Provider for the Amazon Bedrock Mantle OpenAI-compatible API."""

    @property
    def name(self) -> str:
        return 'bedrock-mantle'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        profile = openai_model_profile(model_name.removeprefix('openai.'))
        return merge_profile(profile, ModelProfile(tool_call_id_scope='response'))

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        region_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        region_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a Bedrock Mantle provider.

        Args:
            region_name: The AWS region used to construct the default `openai/v1` endpoint.
            base_url: A complete Mantle base URL. Use this for models served from another path, such as `v1`.
            api_key: A Bedrock API key. If omitted, `AWS_BEARER_TOKEN_BEDROCK` is used.
            openai_client: An existing [`AsyncOpenAI`](https://github.com/openai/openai-python) client. If provided,
                `region_name`, `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` used to make requests.
        """
        if openai_client is not None:
            assert region_name is None, 'Cannot provide both `openai_client` and `region_name`'
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
            return

        api_key = api_key or os.getenv('AWS_BEARER_TOKEN_BEDROCK')
        if not api_key:
            raise missing_api_key_error(
                'Set the `AWS_BEARER_TOKEN_BEDROCK` environment variable or pass it via '
                '`BedrockMantleProvider(api_key=...)` to use the Bedrock Mantle provider.'
            )

        if base_url is None:
            region_name = region_name or os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION')
            if not region_name:
                raise UserError(
                    'Set the `AWS_DEFAULT_REGION` or `AWS_REGION` environment variable or pass it via '
                    '`BedrockMantleProvider(region_name=...)` to use the Bedrock Mantle provider.'
                )
            if not _AWS_REGION_PATTERN.fullmatch(region_name):
                raise UserError(f'Invalid AWS region name: {region_name!r}')
            base_url = f'https://bedrock-mantle.{region_name}.api.aws/openai/v1'

        if http_client is not None:
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = create_async_http_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_http_client
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
