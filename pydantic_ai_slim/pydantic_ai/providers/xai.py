from __future__ import annotations as _annotations

import os
from typing import Literal, overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.providers import Provider

try:
    from xai_sdk import AsyncClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `xai-sdk` package to use the xAI provider, '
        'you can use the `xai` optional group — `pip install "pydantic-ai-slim[xai]"`'
    ) from _import_error


# https://docs.x.ai/docs/models
XaiModelName = Literal[
    'grok-4',
    'grok-4-0709',
    'grok-4-1-fast-reasoning',
    'grok-4-1-fast-non-reasoning',
    'grok-4-fast-reasoning',
    'grok-4-fast-non-reasoning',
    'grok-code-fast-1',
    'grok-3',
    'grok-3-mini',
    'grok-3-fast',
    'grok-3-mini-fast',
    'grok-2-vision-1212',
    'grok-2-image-1212',
]


class XaiProvider(Provider[AsyncClient]):
    """Provider for xAI API (native xAI SDK)."""

    @property
    def name(self) -> str:
        return 'xai'

    @property
    def base_url(self) -> str:
        return 'https://api.x.ai/v1'

    @property
    def client(self) -> AsyncClient:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return grok_model_profile(model_name)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, xai_client: AsyncClient) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        xai_client: AsyncClient | None = None,
    ) -> None:
        if xai_client is not None:
            self._client = xai_client
        else:
            api_key = api_key or os.getenv('XAI_API_KEY')
            if not api_key:
                raise UserError(
                    'Set the `XAI_API_KEY` environment variable or pass it via `XaiProvider(api_key=...)`'
                    'to use the xAI provider.'
                )
            self._client = AsyncClient(api_key=api_key)
