from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles._family import vendor_family_model_profile
from pydantic_ai.profiles._special_tokens import special_token_route_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Together AI provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        _OpenAISDKProvider,  # pyright: ignore[reportPrivateUsage]
    )


def _gpt_oss_only_profile(model_name: str) -> ModelProfile | None:
    """Together's only OpenAI chat models are gpt-oss; other `openai/*` ids keep Together's defaults."""
    return harmony_model_profile(model_name) if model_name.startswith('gpt-oss') else None


class TogetherProvider(_OpenAISDKProvider):
    """Provider for Together AI API."""

    @property
    def name(self) -> str:
        return 'together'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        profile = vendor_family_model_profile(model_name, overrides={'openai': _gpt_oss_only_profile})
        if model_name.lower().startswith('deepseek-ai/deepseek-v4-'):
            # DeepSeek's V4 models reject a forced tool choice while thinking is on, and thinking is
            # their default. Whether Together honors DeepSeek's thinking toggle is unverified, so the
            # restriction is unconditional here rather than per request as it is on `DeepSeekProvider`.
            profile = merge_profile(profile, OpenAIModelProfile(supports_forced_tool_choice=False))

        # As the Together API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer,
        # unless json_schema_transformer is set explicitly
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
            special_token_route_profile('together', model_name),
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
        api_key = api_key or os.getenv('TOGETHER_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `TOGETHER_API_KEY` environment variable or pass it via `TogetherProvider(api_key=...)`'
                ' to use the Together AI provider.'
            )

        assert openai_client is None or base_url is None, 'Cannot provide both `openai_client` and `base_url`'
        self._base_url = base_url or 'https://api.together.xyz/v1'
        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
