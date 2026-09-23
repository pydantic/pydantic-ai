from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Opper provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )


class OpperProvider(_OpenAICompatibleProvider):
    """Provider for Opper, an EU-hosted OpenAI-compatible gateway."""

    @property
    def name(self) -> str:
        return 'opper'

    @property
    def base_url(self) -> str:
        return 'https://api.opper.ai/v3/compat'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        # Opper identifies models by a bare pool name, e.g. `claude-sonnet-4-6`, where a pool is
        # every provider serving that model and Opper picks the route per request. A
        # `provider/model` id such as `azure/gpt-5.5` pins one provider or region instead, so
        # strip any route prefix before matching the model family.
        _, _, bare_name = model_name.rpartition('/')
        bare_name = bare_name.lower()

        # Ordered longest-prefix-first so that e.g. `gpt-oss` resolves to the Harmony profile
        # rather than the OpenAI one.
        prefix_to_profile = (
            ('gpt-oss', harmony_model_profile),
            ('claude', anthropic_model_profile),
            ('gemini', google_model_profile),
            ('gemma', google_model_profile),
            ('deepseek', deepseek_model_profile),
            ('kimi', moonshotai_model_profile),
            ('moonshot', moonshotai_model_profile),
            ('qwen', qwen_model_profile),
            ('grok', grok_model_profile),
            ('glm', zai_model_profile),
            ('llama', meta_model_profile),
            ('meta-', meta_model_profile),
            ('mistral', mistral_model_profile),
            ('ministral', mistral_model_profile),
            ('magistral', mistral_model_profile),
            ('devstral', mistral_model_profile),
            ('codestral', mistral_model_profile),
            ('gpt', openai_model_profile),
            ('o1', openai_model_profile),
            ('o3', openai_model_profile),
            ('o4', openai_model_profile),
        )

        profile = None
        for prefix, profile_func in prefix_to_profile:
            if bare_name.startswith(prefix):
                profile = profile_func(bare_name)
                break

        # As the Opper API is OpenAI-compatible, we also need OpenAIJsonSchemaTransformer.
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
        api_key = api_key or os.getenv('OPPER_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `OPPER_API_KEY` environment variable or pass it via '
                '`OpperProvider(api_key=...)` to use the Opper provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
