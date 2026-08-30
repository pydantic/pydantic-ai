from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use OVHcloud AI Endpoints provider.'
        'You can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )


class OVHcloudProvider(_OpenAICompatibleProvider):
    """Provider for OVHcloud AI Endpoints."""

    @property
    def name(self) -> str:
        return 'ovhcloud'

    @property
    def base_url(self) -> str:
        return 'https://oai.endpoints.kepler.ai.cloud.ovh.net/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        model_name = model_name.lower()

        prefix_to_profile = {
            'llama': meta_model_profile,
            'meta-': meta_model_profile,
            'deepseek': deepseek_model_profile,
            'mistral': mistral_model_profile,
            'gpt': harmony_model_profile,
            'qwen': qwen_model_profile,
        }

        profile = None
        for prefix, profile_func in prefix_to_profile.items():
            if model_name.startswith(prefix):
                profile = profile_func(model_name)

        # As the OVHcloud AI Endpoints API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer.
        return merge_profile(OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer), profile)

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
        api_key = api_key or os.getenv('OVHCLOUD_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `OVHCLOUD_API_KEY` environment variable or pass it via '
                '`OVHcloudProvider(api_key=...)` to use OVHcloud AI Endpoints provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
