from __future__ import annotations as _annotations

import os
from typing import overload

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
        """Create a new xAI provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `XAI_API_KEY` environment variable
                will be used if available.
            xai_client: An existing `xai_sdk.AsyncClient` to use.  This takes precedence over `api_key`.
        """
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
