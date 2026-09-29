from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai.exceptions import UserError
from pydantic_ai.models.openai import OpenAIModelProfile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the NVIDIA provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"``'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )


class NVIDIAProvider(_OpenAICompatibleProvider):
    """Provider for NVIDIA NIM API."""

    @property
    def name(self) -> str:
        return 'nvidia'

    @property
    def base_url(self) -> str:
        return os.getenv('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> OpenAIModelProfile | None:
        # NVIDIA NIM is OpenAI-compatible, so we use the default OpenAI model profile
        return OpenAIModelProfile()

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('NVIDIA_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `NVIDIA_API_KEY` environment variable or pass it via `NVIDIAProvider(api_key=...)`'
                ' to use the NVIDIA provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)