from __future__ import annotations as _annotations

from pydantic_ai.providers import Provider

try:
    from anthropic import AsyncAnthropicVertex
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `anthropic` package to use the Anthropic provider, '
        'you can use the `anthropic` optional group — `pip install "pydantic-ai-slim[anthropic]"`'
    ) from _import_error


class AnthropicVertexProvider(Provider[AsyncAnthropicVertex]):
    """Provider for Anthropic API."""

    @property
    def name(self) -> str:
        return 'anthropic-vertex'

    @property
    def base_url(self) -> str:
        return str(self._client.base_url)

    @property
    def client(self) -> AsyncAnthropicVertex:
        return self._client

    def __init__(
        self,
        *,
        anthropic_client: AsyncAnthropicVertex | None = None,
    ) -> None:
        """Create a new Anthropic provider.

        Args:
            anthropic_client: An existing [`AsyncAnthropic`](https://github.com/anthropics/anthropic-sdk-python)
                client to use. If provided, the `api_key` and `http_client` arguments will be ignored.
        """
        if anthropic_client:
            self._client = anthropic_client
        else:
            self._client = AsyncAnthropicVertex()
