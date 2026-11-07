from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles._special_tokens import special_token_route_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, OpenAIReasoningEffort
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import _glm_5_3_schema_profile  # pyright: ignore[reportPrivateUsage]

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


class TogetherProvider(_OpenAISDKProvider):
    """Provider for Together AI API."""

    @property
    def name(self) -> str:
        return 'together'

    @property
    def base_url(self) -> str:
        return 'https://api.together.xyz/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'deepseek-ai': deepseek_model_profile,
            'google': google_model_profile,
            'qwen': qwen_model_profile,
            'meta-llama': meta_model_profile,
            'mistralai': mistral_model_profile,
            'moonshotai': moonshotai_model_profile,
            'openai': harmony_model_profile,  # gpt-oss models speak OpenAI's Harmony response format
            'zai-org': _glm_5_3_schema_profile,
        }

        profile = None
        route_profile = special_token_route_profile('together', model_name)

        model_name = model_name.lower()
        if '/' in model_name:
            provider, model_name = model_name.split('/', 1)
            if provider in provider_to_profile:
                profile = provider_to_profile[provider](model_name)

            if provider == 'deepseek-ai' and model_name.startswith('deepseek-v4-'):
                # DeepSeek's V4 models reject a forced tool choice while thinking is on, and thinking is
                # their default. Whether Together honors DeepSeek's thinking toggle is unverified, so the
                # restriction is unconditional here rather than per request as it is on `DeepSeekProvider`.
                profile = merge_profile(profile, OpenAIModelProfile(supports_forced_tool_choice=False))

            if provider == 'openai' and model_name.startswith('gpt-oss'):
                # gpt-oss models always reason with a graded `reasoning_effort`; the shared OpenAI
                # family profile gives them none.
                profile = merge_profile(
                    profile,
                    OpenAIModelProfile(
                        supports_thinking=True,
                        thinking_always_enabled=True,
                        openai_reasoning_efforts=frozenset[OpenAIReasoningEffort](('low', 'medium', 'high')),
                    ),
                )

        # As the Together API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer,
        # unless json_schema_transformer is set explicitly
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer), profile, route_profile
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
        api_key = api_key or os.getenv('TOGETHER_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `TOGETHER_API_KEY` environment variable or pass it via `TogetherProvider(api_key=...)`'
                ' to use the Together AI provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
