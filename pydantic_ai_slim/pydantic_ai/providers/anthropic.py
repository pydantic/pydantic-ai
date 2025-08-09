from __future__ import annotations as _annotations

import os
from typing import TypeAlias, overload

import httpx

from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.providers import Provider

try:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `anthropic` package to use the Anthropic provider, '
        'you can use the `anthropic` optional group — `pip install "pydantic-ai-slim[anthropic]"`'
    ) from _import_error


ASYNC_ANTHROPIC_CLIENT: TypeAlias = AsyncAnthropic | AsyncAnthropicBedrock


class AnthropicProvider(Provider[ASYNC_ANTHROPIC_CLIENT]):
    """Provider for Anthropic API."""

    @property
    def name(self) -> str:
        return 'anthropic'

    @property
    def base_url(self) -> str:
        return str(self._client.base_url)

    @property
    def client(self) -> ASYNC_ANTHROPIC_CLIENT:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return anthropic_model_profile(model_name)

    @overload
    def __init__(self, *, anthropic_client: ASYNC_ANTHROPIC_CLIENT | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str | None = None, http_client: httpx.AsyncClient | None = None) -> None: ...

    @overload
    def __init__(
        self,
        *,
        aws_secret_key: str | None = None,
        aws_access_key: str | None = None,
        aws_region: str | None = None,
        aws_profile: str | None = None,
        aws_session_token: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        aws_secret_key: str | None = None,
        aws_access_key: str | None = None,
        aws_region: str | None = None,
        aws_profile: str | None = None,
        aws_session_token: str | None = None,
        anthropic_client: ASYNC_ANTHROPIC_CLIENT | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Anthropic provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `ANTHROPIC_API_KEY` environment variable
                will be used if available.
            aws_secret_key: AWS secret access key for Bedrock authentication.
            aws_access_key: AWS access key ID for Bedrock authentication.
            aws_region: AWS region for Bedrock service.
            aws_profile: AWS profile name for Bedrock authentication.
            aws_session_token: AWS session token for temporary credentials.
            anthropic_client: An existing [`AsyncAnthropic`](https://github.com/anthropics/anthropic-sdk-python)
                client to use. If provided, the `api_key` and `http_client` arguments will be ignored.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if anthropic_client is not None:
            assert http_client is None, 'Cannot provide both `anthropic_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `anthropic_client` and `api_key`'
            self._client = anthropic_client
        else:
            api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
            if api_key is None:
                if http_client is not None:
                    self._client = AsyncAnthropicBedrock(
                        aws_access_key=aws_access_key,
                        aws_secret_key=aws_secret_key,
                        aws_session_token=aws_session_token,
                        aws_profile=aws_profile,
                        aws_region=aws_region,
                        http_client=http_client,
                    )
                else:
                    http_client = cached_async_http_client(provider='anthropic')
                    self._client = AsyncAnthropicBedrock(
                        aws_access_key=aws_access_key,
                        aws_secret_key=aws_secret_key,
                        aws_session_token=aws_session_token,
                        aws_profile=aws_profile,
                        aws_region=aws_region,
                        http_client=http_client,
                    )

            else:
                if http_client is not None:
                    self._client = AsyncAnthropic(api_key=api_key, http_client=http_client)
                else:
                    http_client = cached_async_http_client(provider='anthropic')
                    self._client = AsyncAnthropic(api_key=api_key, http_client=http_client)
