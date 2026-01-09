from __future__ import annotations as _annotations

import os

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

    def __init__(
        self,
        *,
        api_key: str | None = None,
        voyageai_client: AsyncClient | None = None,
        base_url: str | None = None,
        max_retries: int = 0,
        timeout: float | None = None,
    ) -> None:
        """Create a new VoyageAI provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `VOYAGE_API_KEY` environment variable
                will be used if available.
            voyageai_client: An existing
                [AsyncClient](https://github.com/voyage-ai/voyageai-python)
                client to use. If provided, `api_key`, `base_url`, `max_retries`, and `timeout` must be `None`/default.
            base_url: The base URL for the VoyageAI API. Defaults to `https://api.voyageai.com/v1`.
            max_retries: Maximum number of retries for failed requests.
            timeout: Request timeout in seconds.
        """
        if voyageai_client is not None:
            assert api_key is None, 'Cannot provide both `voyageai_client` and `api_key`'
            assert base_url is None, 'Cannot provide both `voyageai_client` and `base_url`'
            assert max_retries == 0, 'Cannot provide both `voyageai_client` and `max_retries`'
            assert timeout is None, 'Cannot provide both `voyageai_client` and `timeout`'
            self._client = voyageai_client
        else:
            api_key = api_key or os.getenv('VOYAGE_API_KEY')
            if not api_key:
                raise UserError(
                    'Set the `VOYAGE_API_KEY` environment variable or pass it via `VoyageAIProvider(api_key=...)` '
                    'to use the VoyageAI provider.'
                )

            # Only pass base_url if explicitly set; otherwise use VoyageAI's default
            base_url = base_url or os.getenv('VOYAGE_BASE_URL')
            self._client = AsyncClient(
                api_key=api_key,
                max_retries=max_retries,
                timeout=timeout,
                base_url=base_url,
            )
