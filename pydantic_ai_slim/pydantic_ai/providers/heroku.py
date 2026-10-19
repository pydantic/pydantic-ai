from __future__ import annotations as _annotations

import os
import re
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Heroku provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )


_HEROKU_GLM_MINOR_VERSION_RE = re.compile(r'^glm-(\d+)-(\d+)')


def _heroku_glm_model_profile(model_name: str) -> ModelProfile | None:
    # GLM is a Z.AI model family, but Heroku spells minor versions with a hyphen
    # (`glm-4-7`) where the Z.AI profile expects a dot (`glm-4.7`).
    return zai_model_profile(_HEROKU_GLM_MINOR_VERSION_RE.sub(r'glm-\1.\2', model_name))


def _heroku_kimi_model_profile(model_name: str) -> ModelProfile | None:
    # Heroku spells the Kimi 2.5 minor version with a hyphen (`kimi-k2-5`) where MoonshotAI's
    # native id uses a dot (`kimi-k2.5`), which is what `moonshotai_model_profile` matches.
    if model_name.lower() == 'kimi-k2-5':
        model_name = 'kimi-k2.5'
    return moonshotai_model_profile(model_name)


class HerokuProvider(_OpenAICompatibleProvider):
    """Provider for Heroku API."""

    @property
    def name(self) -> str:
        return 'heroku'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        # Heroku Managed Inference serves models from several families (Claude, Nova, gpt-oss,
        # Qwen, DeepSeek, Kimi, …) under bare model names with no provider prefix. Route the name
        # through the matching family profile so capabilities like `supports_thinking` are detected
        # instead of silently dropped; otherwise reasoning settings (e.g. `thinking=True`) are
        # accepted but never sent on the wire.
        prefix_to_profile = {
            'claude': anthropic_model_profile,
            'gpt-oss': harmony_model_profile,
            'qwen': qwen_model_profile,
            'deepseek': deepseek_model_profile,
            'kimi': _heroku_kimi_model_profile,
            'glm': _heroku_glm_model_profile,
            'mistral': mistral_model_profile,
            'nova': amazon_model_profile,
            'llama': meta_model_profile,
            'gemma': google_model_profile,
        }

        profile = None
        lower_model_name = model_name.lower()
        for prefix, profile_func in prefix_to_profile.items():
            if lower_model_name.startswith(prefix):
                profile = profile_func(lower_model_name)
                break

        # As the Heroku API is OpenAI-compatible, we keep the OpenAIJsonSchemaTransformer as the base
        # and layer any family-specific profile on top.
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
        )

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, base_url: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: _OpenAIHTTPClient) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: _OpenAIHTTPClient, base_url: str) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        if openai_client is not None:
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        else:
            api_key = api_key or os.getenv('HEROKU_INFERENCE_KEY')
            if not api_key:
                raise UserError(
                    'Set the `HEROKU_INFERENCE_KEY` environment variable or pass it via `HerokuProvider(api_key=...)`'
                    ' to use the Heroku provider.'
                )

            base_url = (base_url or os.getenv('HEROKU_INFERENCE_URL', 'https://us.inference.heroku.com')).rstrip('/')
            if not base_url.endswith('/v1'):
                base_url += '/v1'

            self._client = self._create_openai_client(base_url=base_url, api_key=api_key, http_client=http_client)
