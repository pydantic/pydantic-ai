from __future__ import annotations

import os

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Pinstripes provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = ['PinstripeProvider']


class PinstripeProvider(Provider[AsyncOpenAI]):
    """Provider for Pinstripes inference API.

    Pinstripes (https://pinstripes.io) is an OpenAI-compatible inference API
    serving models under the ``ps/`` namespace.
    """

    @property
    def name(self) -> str:
        """Return the provider name."""
        return 'pinstripes'

    @property
    def base_url(self) -> str:
        """Return the base URL."""
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        """Return the AsyncOpenAI client."""
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        """Get model profile for Pinstripes models.

        Pinstripes serves models under the ``ps/`` namespace prefix.  The
        profile is selected based on the underlying model family so that
        JSON-schema and tool-call quirks are handled correctly.
        """
        # Strip the ps/ namespace prefix before matching families.
        bare = model_name.lower().removeprefix('ps/')

        profile: ModelProfile | None = None
        if bare.startswith('qwen'):
            profile = qwen_model_profile(bare)
        elif bare.startswith('deepseek'):
            profile = deepseek_model_profile(bare)

        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize Pinstripes provider.

        Args:
            api_key: Pinstripes API key.  If not provided, reads from the
                ``PINSTRIPES_API_KEY`` environment variable.
            base_url: Custom API base URL.  Defaults to
                ``https://pinstripes.io/v1``.
            openai_client: Optional pre-configured :class:`openai.AsyncOpenAI`
                client.  When supplied, ``api_key``, ``base_url``, and
                ``http_client`` are ignored.
            http_client: Optional custom :class:`httpx.AsyncClient` for making
                HTTP requests.

        Raises:
            UserError: If no API key is available and no ``openai_client`` is
                provided.
        """
        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
        else:
            api_key = api_key or os.getenv('PINSTRIPES_API_KEY')
            if not api_key:
                raise UserError(
                    'Set the `PINSTRIPES_API_KEY` environment variable or pass it via '
                    '`PinstripeProvider(api_key=...)` to use the Pinstripes provider.'
                )

            self._base_url = base_url or os.getenv('PINSTRIPES_BASE_URL', 'https://pinstripes.io/v1')

            if http_client is None:
                http_client = create_async_http_client()
                self._own_http_client = http_client
                self._http_client_factory = create_async_http_client
            self._client = AsyncOpenAI(base_url=self._base_url, api_key=api_key, http_client=http_client)

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
