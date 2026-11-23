from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles._family import ModelProfileFunc, vendor_family_model_profile
from pydantic_ai.profiles._special_tokens import special_token_route_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Nebius provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        _OpenAISDKProvider,  # pyright: ignore[reportPrivateUsage]
    )


_NEBIUS_VENDORS = frozenset({'meta-llama', 'deepseek-ai', 'qwen', 'google', 'mistralai', 'moonshotai', 'zai-org'})
"""The vendors in the shared family map whose profile Nebius ids get."""

_NEBIUS_VENDOR_PROFILES: dict[str, ModelProfileFunc] = {
    # Nebius serves gpt-oss under `openai/`.
    'openai': harmony_model_profile,
}
"""Vendors whose profile on Nebius differs from the shared family map."""


class NebiusProvider(_OpenAISDKProvider):
    """Provider for Nebius AI Studio API."""

    @property
    def name(self) -> str:
        return 'nebius'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        profile = vendor_family_model_profile(model_name, overrides=_NEBIUS_VENDOR_PROFILES, vendors=_NEBIUS_VENDORS)
        route_profile = special_token_route_profile('nebius', model_name)

        # As NebiusProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer), profile, route_profile
        )

    @overload
    def __init__(self, *, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str, base_url: str | None = None, http_client: _OpenAIHTTPClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('NEBIUS_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `NEBIUS_API_KEY` environment variable or pass it via '
                '`NebiusProvider(api_key=...)` to use the Nebius AI Studio provider.'
            )

        assert openai_client is None or base_url is None, 'Cannot provide both `openai_client` and `base_url`'
        self._base_url = base_url or 'https://api.studio.nebius.com/v1'
        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
