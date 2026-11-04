from __future__ import annotations as _annotations

import os
from typing import Literal, overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.meta import muse_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Meta provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        _OpenAISDKProvider,  # pyright: ignore[reportPrivateUsage]
    )

MetaModelName = Literal[
    'muse-spark-1.1',
    'muse-spark-1.2',
    'muse-spark-1.2-contributor',
    'muse-spark-1.3',
    'muse-spark-1.3-contributor',
    'muse-wonder-1.0',
]
"""Known Meta Model API models.

See <https://dev.meta.ai/docs/models> for an up to date list of models.
"""


class MetaProvider(_OpenAISDKProvider):
    """Provider for Meta's Model API (Muse models)."""

    @property
    def name(self) -> str:
        return 'meta'

    @property
    def base_url(self) -> str:
        return 'https://api.meta.ai/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        # `api.meta.ai` only accepts `tool_choice='auto'` (`none`, `required` and named functions are rejected),
        # on both the Responses and Chat Completions APIs.
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer, supports_forced_tool_choice=False),
            muse_model_profile(model_name),
        )

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: _OpenAIHTTPClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new Meta provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `META_API_KEY` environment
                variable will be used if available.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('META_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `META_API_KEY` environment variable or pass it via '
                '`MetaProvider(api_key=...)` to use the Meta provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
