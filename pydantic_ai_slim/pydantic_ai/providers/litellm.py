from __future__ import annotations as _annotations

from typing import overload

from pydantic_ai import ModelProfile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the LiteLLM provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        _OpenAISDKProvider,  # pyright: ignore[reportPrivateUsage]
    )
    from .openai_compatible import _vendor_prefixed_model_profile  # pyright: ignore[reportPrivateUsage]


class LiteLLMProvider(_OpenAISDKProvider):
    """Provider for LiteLLM API."""

    @property
    def name(self) -> str:
        return 'litellm'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return _vendor_prefixed_model_profile(model_name)

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        http_client: _OpenAIHTTPClient,
    ) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Initialize a LiteLLM provider.

        Args:
            api_key: API key for the model provider. If None, LiteLLM will try to get it from environment variables.
            api_base: Base URL for the model provider. Use this for custom endpoints or self-hosted models.
            openai_client: Pre-configured OpenAI client. If provided, other parameters are ignored.
            http_client: Custom HTTP client to use.
        """
        if openai_client is not None:
            self._client = openai_client
            return

        # Create OpenAI client that will be used with LiteLLM's completion function
        # The actual API calls will be intercepted and routed through LiteLLM
        self._client = self._create_openai_client(
            base_url=api_base, api_key=api_key or 'litellm-placeholder', http_client=http_client
        )
