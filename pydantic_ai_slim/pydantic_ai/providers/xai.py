from __future__ import annotations as _annotations

import asyncio
import os
from typing import Any, overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.providers import Provider

try:
    from xai_sdk import AsyncClient
except ImportError as _import_error:
    raise ImportError(
        'Please install the `xai-sdk` package to use the xAI provider, '
        'you can use the `xai` optional group — `pip install "pydantic-ai-slim[xai]"`'
    ) from _import_error


class _LazyAsyncClient:
    """Wrapper that creates a fresh AsyncClient per event loop.

    gRPC async channels bind to the event loop at creation time. If the client
    is created outside an async context (e.g. at module level) and later used
    inside asyncio.run(), the loop will differ, causing RuntimeError.
    This wrapper defers client creation and recreates it when the loop changes.
    See https://github.com/grpc/grpc/issues/32480.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._client: AsyncClient | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def get_client(self) -> AsyncClient:
        running_loop: asyncio.AbstractEventLoop | None = None
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if self._client is None or (running_loop is not None and running_loop is not self._event_loop):
            self._client = AsyncClient(**self._kwargs)
            self._event_loop = running_loop

        return self._client


class XaiProvider(Provider[AsyncClient]):
    """Provider for xAI API (native xAI SDK)."""

    @property
    def name(self) -> str:
        return 'xai'

    @property
    def base_url(self) -> str:
        # Canonical pricing/identity label, not the transport host: the xAI SDK is gRPC and the actual
        # channel target is set via `api_host`. This URL is used for usage/price lookup and telemetry only.
        return 'https://api.x.ai/v1'

    @property
    def client(self) -> AsyncClient:
        if self._lazy_client is not None:
            return self._lazy_client.get_client()
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return grok_model_profile(model_name)

    @overload
    def __init__(
        self, *, api_key: str | None = None, api_host: str | None = None, timeout: float | None = None
    ) -> None: ...

    @overload
    def __init__(self, *, xai_client: AsyncClient) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_host: str | None = None,
        timeout: float | None = None,
        xai_client: AsyncClient | None = None,
    ) -> None:
        """Create a new xAI provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `XAI_API_KEY` environment variable
                will be used if available.
            api_host: The API host to use for the xAI SDK client.
            timeout: The client-level default timeout for the xAI SDK client, in seconds, applied to all requests
                made through it. This is distinct from `ModelSettings.timeout`, which overrides the timeout for an
                individual request.
            xai_client: An existing `xai_sdk.AsyncClient` to use. This takes precedence over `api_key`, `api_host`,
                and `timeout`.
        """
        self._lazy_client: _LazyAsyncClient | None = None
        if xai_client is not None:
            self._client = xai_client
        else:
            api_key = api_key or os.getenv('XAI_API_KEY')
            if not api_key:
                raise UserError(
                    'Set the `XAI_API_KEY` environment variable or pass it via `XaiProvider(api_key=...)`'
                    ' to use the xAI provider.'
                )
            client_kwargs: dict[str, str | float] = {'api_key': api_key}
            if api_host is not None:
                client_kwargs['api_host'] = api_host
            if timeout is not None:
                client_kwargs['timeout'] = timeout
            self._lazy_client = _LazyAsyncClient(**client_kwargs)
            self._client = None  # type: ignore[assignment]
