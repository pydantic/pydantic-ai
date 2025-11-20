from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers import Provider

try:
    from cerebras.cloud.sdk import AsyncCerebras
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `cerebras-cloud-sdk` package to use the Cerebras provider, '
        'you can use the `cerebras` optional group — `pip install "pydantic-ai-slim[cerebras]"`'
    ) from _import_error


class CerebrasProvider(Provider[AsyncCerebras]):
    """Provider for Cerebras API."""

    @property
    def name(self) -> str:
        return 'cerebras'

    @property
    def base_url(self) -> str:
        return 'https://api.cerebras.ai/v1'

    @property
    def client(self) -> AsyncCerebras:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        prefix_to_profile = {
            'llama': meta_model_profile,
            'qwen': qwen_model_profile,
            'gpt-oss': harmony_model_profile,
        }

        for prefix, profile_func in prefix_to_profile.items():
            model_name = model_name.lower()
            if model_name.startswith(prefix):
                return profile_func(model_name)

        return None

    @overload
    def __init__(self, *, cerebras_client: AsyncCerebras | None = None) -> None: ...

    @overload
    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None, http_client: httpx.AsyncClient | None = None
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        cerebras_client: AsyncCerebras | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Cerebras provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `CEREBRAS_API_KEY` environment variable
                will be used if available.
            base_url: The base url for the Cerebras requests. If not provided, defaults to Cerebras's base url.
            cerebras_client: An existing `AsyncCerebras` client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if cerebras_client is not None:
            assert http_client is None, 'Cannot provide both `cerebras_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `cerebras_client` and `api_key`'
            assert base_url is None, 'Cannot provide both `cerebras_client` and `base_url`'
            self._client = cerebras_client
        else:
            api_key = api_key or os.getenv('CEREBRAS_API_KEY')
            base_url = base_url or 'https://api.cerebras.ai/v1'

            if not api_key:
                raise UserError(
                    'Set the `CEREBRAS_API_KEY` environment variable or pass it via `CerebrasProvider(api_key=...)` '
                    'to use the Cerebras provider.'
                )
            elif http_client is not None:
                self._client = AsyncCerebras(base_url=base_url, api_key=api_key, http_client=http_client)
            else:
                http_client = cached_async_http_client(provider='cerebras')
                self._client = AsyncCerebras(base_url=base_url, api_key=api_key, http_client=http_client)
