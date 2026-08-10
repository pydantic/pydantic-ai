from __future__ import annotations as _annotations

import os
import warnings
from collections.abc import Callable, Mapping
from typing import overload

import httpx
import httpx2

from pydantic_ai import ModelProfile
from pydantic_ai._http import create_httpx2_client
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile
from pydantic_ai.providers import Provider, missing_api_key_error

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the OpenAI provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


_OpenAIHTTPClient = httpx.AsyncClient | httpx2.AsyncClient


class _OpenAICompatibleProvider(Provider[AsyncOpenAI]):
    """Shared HTTP client lifecycle for providers backed by the OpenAI SDK."""

    _own_http_client: _OpenAIHTTPClient | None = None
    _http_client_factory: Callable[[], _OpenAIHTTPClient] | None = None

    def _get_http_client(
        self, http_client: _OpenAIHTTPClient | None, *, warning_stacklevel: int = 3
    ) -> _OpenAIHTTPClient:
        if http_client is None:
            http_client = create_httpx2_client()
            self._own_http_client = http_client  # pyright: ignore[reportIncompatibleVariableOverride]
            self._http_client_factory = create_httpx2_client  # pyright: ignore[reportIncompatibleVariableOverride]
        elif isinstance(http_client, httpx.AsyncClient):
            warnings.warn(
                '`httpx.AsyncClient` support for OpenAI-compatible providers is deprecated and will be removed in v3; '
                'use `httpx2.AsyncClient` instead.',
                PydanticAIDeprecationWarning,
                stacklevel=warning_stacklevel,
            )
        return http_client

    def _create_openai_client(
        self,
        *,
        base_url: str | None,
        api_key: str | None,
        http_client: _OpenAIHTTPClient | None,
        default_headers: Mapping[str, str] | None = None,
    ) -> AsyncOpenAI:
        http_client = self._get_http_client(http_client, warning_stacklevel=4)
        # openai-python >=2.47 accepts an httpx2 client, but its type annotations still name legacy httpx.
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,  # pyright: ignore[reportArgumentType]
            default_headers=default_headers,
        )

    # The generic Provider currently only knows the legacy HTTPX client type.
    def _set_http_client(self, http_client: _OpenAIHTTPClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]


class OpenAIProvider(_OpenAICompatibleProvider):
    """Provider for OpenAI API."""

    @property
    def name(self) -> str:
        return 'openai'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        # No per-model gate on `additional_tools`. OpenAI documents a model restriction for the sibling
        # feature ("Only `gpt-5.4` and later models support `tool_search`") and states none for the item,
        # and measurement agrees: 13 models from `gpt-4o-mini` through `gpt-5.6` each called a tool that
        # only an `additional_tools` item declared, 3/3, against 0/3 for a control with the item removed.
        # An earlier list here recorded `gpt-5.4` and `gpt-5` as silently ignoring it; that came from a
        # probe whose prompt named the tool and told the model to call it, which a model that never saw
        # the declaration can satisfy from the prompt text alone.
        #
        # The flag stays here rather than moving into `openai_model_profile`, which is shared with
        # OpenAI-compatible endpoints (Azure, OpenRouter, vLLM, ...) that speak the Responses API without
        # necessarily implementing this item — the same reasoning `openai_supports_phase` documents.
        return merge_profile(
            openai_model_profile(model_name),
            OpenAIModelProfile(tool_addition_mode='with_definitions', tool_deferral_mode='with_tool_search'),
        )

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new OpenAI provider.

        Args:
            base_url: The base url for the OpenAI requests. If not provided, the `OPENAI_BASE_URL` environment variable
                will be used if available. Otherwise, defaults to OpenAI's base url.
            api_key: The API key to use for authentication, if not provided, the `OPENAI_API_KEY` environment variable
                will be used if available.
            openai_client: An existing
                [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        if api_key is None and 'OPENAI_API_KEY' not in os.environ and openai_client is None:
            if base_url is None and 'OPENAI_BASE_URL' not in os.environ:
                # When talking to OpenAI directly, a missing key would otherwise surface as a raw
                # `openai.OpenAIError`; raise our own `UserError` instead so the message is consistent with
                # other providers and points newcomers to the keyless test model.
                raise missing_api_key_error(
                    'Set the `OPENAI_API_KEY` environment variable or pass it via `OpenAIProvider(api_key=...)`'
                    ' to use the OpenAI provider.'
                )
            else:
                # This is a workaround for the OpenAI client requiring an API key, whilst locally served,
                # openai compatible models do not always need an API key, but a placeholder (non-empty) key is required.
                api_key = 'api-key-not-set'

        if openai_client is not None:
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=base_url, api_key=api_key, http_client=http_client)
