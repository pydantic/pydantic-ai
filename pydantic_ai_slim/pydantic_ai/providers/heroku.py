from __future__ import annotations as _annotations

import os
from typing import overload

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
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
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Heroku provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class HerokuProvider(Provider[AsyncOpenAI]):
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
            'kimi': moonshotai_model_profile,
            'glm': moonshotai_model_profile,
            'mistral': mistral_model_profile,
            'nova': amazon_model_profile,
            'llama': meta_model_profile,
            'gemma': google_model_profile,
        }

        profile = None
        lower_model_name = model_name.lower()
        for prefix, profile_func in prefix_to_profile.items():
            if lower_model_name.startswith(prefix):
                profile = profile_func(model_name)
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
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient, base_url: str) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
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

            if http_client is not None:
                self._client = AsyncOpenAI(api_key=api_key, http_client=http_client, base_url=base_url)
            else:
                http_client = create_async_http_client()
                self._own_http_client = http_client
                self._http_client_factory = create_async_http_client
                self._client = AsyncOpenAI(api_key=api_key, http_client=http_client, base_url=base_url)

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
