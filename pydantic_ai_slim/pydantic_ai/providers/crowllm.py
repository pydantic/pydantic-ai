from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai.exceptions import UserError

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the CrowLLM provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        _OpenAISDKProvider,  # pyright: ignore[reportPrivateUsage]
    )


class CrowLLMProvider(_OpenAISDKProvider):
    """Provider for the CrowLLM gateway, an OpenAI-compatible multi-model endpoint."""

    @property
    def name(self) -> str:
        return 'crowllm'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('CROWLLM_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `CROWLLM_API_KEY` environment variable or pass it via `CrowLLMProvider(api_key=...)`'
                ' to use the CrowLLM provider.'
            )

        assert openai_client is None or base_url is None, 'Cannot provide both `openai_client` and `base_url`'
        self._base_url = base_url or 'https://crowllm.com/v1'
        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
