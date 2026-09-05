from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Hubris provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )


class HubrisProvider(_OpenAICompatibleProvider):
    """Provider for Hubris API.

    Hubris is an OpenAI-compatible LLM gateway. Model names use the full
    `vendor/model` form from the catalog at https://hubris.pw/models,
    for example `anthropic/claude-sonnet-5` or `openai/gpt-5.6-luna`.
    """

    @property
    def name(self) -> str:
        return 'hubris'

    @property
    def base_url(self) -> str:
        return 'https://api.hubris.pw/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'anthropic': anthropic_model_profile,
            'openai': openai_model_profile,
            'google': google_model_profile,
            'x-ai': grok_model_profile,
            'z-ai': zai_model_profile,
            'qwen': qwen_model_profile,
            'deepseek': deepseek_model_profile,
            'moonshotai': moonshotai_model_profile,
            'mistralai': mistral_model_profile,
            'meta-llama': meta_model_profile,
            'meta': meta_model_profile,
            'amazon': amazon_model_profile,
            'cohere': cohere_model_profile,
        }

        profile = None

        model_name = model_name.lower()
        if '/' not in model_name:
            return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer)

        provider, model_name = model_name.split('/', 1)
        if provider in provider_to_profile:
            profile = provider_to_profile[provider](model_name)

        # As HubrisProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly
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
        api_key = api_key or os.getenv('HUBRIS_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `HUBRIS_API_KEY` environment variable or pass it via '
                '`HubrisProvider(api_key=...)` to use the Hubris provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
