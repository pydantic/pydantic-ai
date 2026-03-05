from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers import Provider

try:
    from voyageai.client_async import AsyncClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `voyageai` package to use the VoyageAI provider, '
        'you can use the `voyageai` optional group — `pip install "pydantic-ai-slim[voyageai]"`'
    ) from _import_error


class VoyageAIProvider(Provider[AsyncClient]):
    """Provider for VoyageAI API."""

    @property
    def name(self) -> str:
        return 'voyageai'

    @property
    def base_url(self) -> str:
        return self._client._params.get('base_url') or 'https://api.voyageai.com/v1'  # type: ignore

    @property
    def client(self) -> AsyncClient:
        return self._client

    @overload
    def __init__(self, *, voyageai_client: AsyncClient) -> None: ...

    @overload
    def __init__(self, *, api_key: str | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        voyageai_client: AsyncClient | None = None,
    ) -> None:
        """Create a new VoyageAI provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `VOYAGE_API_KEY` environment variable
                will be used if available.
            voyageai_client: An existing
                [AsyncClient](https://github.com/voyage-ai/voyageai-python)
                client to use. If provided, `api_key` must be `None`.
        """
        if voyageai_client is not None:
            assert api_key is None, 'Cannot provide both `voyageai_client` and `api_key`'
            self._client = voyageai_client
        else:
            api_key = api_key or os.getenv('VOYAGE_API_KEY')
            if not api_key:
                raise UserError(
                    'Set the `VOYAGE_API_KEY` environment variable or pass it via `VoyageAIProvider(api_key=...)` '
                    'to use the VoyageAI provider.'
                )

            self._client = AsyncClient(api_key=api_key)
