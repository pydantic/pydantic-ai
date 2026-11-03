from __future__ import annotations as _annotations

import os
import re
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Fireworks AI provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        _OpenAISDKProvider,  # pyright: ignore[reportPrivateUsage]
    )

_KIMI_DOT_VERSION_RE = re.compile(r'(kimi-k\d+)p(\d+)')


class FireworksProvider(_OpenAISDKProvider):
    """Provider for Fireworks AI API."""

    @property
    def name(self) -> str:
        return 'fireworks'

    @property
    def base_url(self) -> str:
        return 'https://api.fireworks.ai/inference/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        model_name = model_name.lower()
        # Fireworks lists Kimi versions with `p` in place of the dot (`kimi-k2p6`); normalize to the
        # dotted spelling the Moonshot family profile matches.
        model_name = _KIMI_DOT_VERSION_RE.sub(r'\g<1>.\g<2>', model_name)
        prefix_to_profile = {
            'llama': meta_model_profile,
            'qwen': qwen_model_profile,
            'deepseek': deepseek_model_profile,
            'mistral': mistral_model_profile,
            'gemma': google_model_profile,
            'kimi': moonshotai_model_profile,
        }

        wrapped = False
        for wrapper in ('accounts/fireworks/models/', 'routers/'):
            if model_name.startswith(wrapper):
                model_name = model_name[len(wrapper) :]
                wrapped = True
                break

        profile = None
        # Fireworks ids are canonically wrapped; unwrapped ids are only routed for Kimi, which
        # Fireworks also serves without the `accounts/fireworks/models/` prefix.
        if wrapped or model_name.startswith('kimi'):
            for provider, profile_func in prefix_to_profile.items():
                if model_name.startswith(provider):
                    profile = profile_func(model_name)
                    break

        # As the Fireworks API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer,
        # unless json_schema_transformer is set explicitly.
        # Native output support wins on top: Fireworks enforces `response_format` JSON schemas on the
        # platform, so `NativeOutput` works for families whose own profiles don't claim support.
        # `supports_json_object_output` stays with the family profile, so `PromptedOutput` requests are unchanged.
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
            ModelProfile(supports_json_schema_output=True),
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
        api_key = api_key or os.getenv('FIREWORKS_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `FIREWORKS_API_KEY` environment variable or pass it via `FireworksProvider(api_key=...)`'
                ' to use the Fireworks AI provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
