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
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Requesty provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[requesty]"`'
    ) from _import_error


class RequestyProvider(Provider[AsyncOpenAI]):
    """Provider for Requesty API."""

    @property
    def name(self) -> str:
        return 'requesty'

    @property
    def base_url(self) -> str:
        return 'https://router.requesty.ai/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'google': google_model_profile,
            'openai': openai_model_profile,
            'anthropic': anthropic_model_profile,
            'mistral': mistral_model_profile,
            'qwen': qwen_model_profile,
            'xai': grok_model_profile,
            'cohere': cohere_model_profile,
            'amazon': amazon_model_profile,
            'deepseek': deepseek_model_profile,
            'meta': meta_model_profile,
            'moonshotai': moonshotai_model_profile,
        }

        profile = None

        # Requesty uses `provider/model` naming (same as OpenRouter); use the provider prefix for profile selection.
        provider, model_name = model_name.split('/', 1)
        if provider in provider_to_profile:
            model_name, *_ = model_name.split(':', 1)  # drop tags
            if provider == 'anthropic':
                model_name = model_name.replace('.', '-')
            profile = provider_to_profile[provider](model_name)

        # Two-layer merge:
        # 1. Fallback layer — `OpenAIJsonSchemaTransformer` is the default unless an upstream profile sets one.
        # 2. Upstream profile — model-specific traits from the lab's profile function.
        # 3. Gateway-specific overrides — Requesty accepts `reasoning` universally, so the gate forces
        #    `supports_thinking=True` so the unified `thinking` setting is always forwarded regardless of the
        #    upstream model's own thinking support. Requesty only accepts the older `max_tokens` field, so
        #    `openai_chat_supports_max_completion_tokens=False`.
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
            OpenAIModelProfile(
                openai_chat_supports_file_urls=True,
                openai_chat_supports_web_search=True,
                openai_chat_supports_max_completion_tokens=False,
                supports_thinking=True,
            ),
        )

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        app_url: str | None = None,
        app_title: str | None = None,
        openai_client: None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        app_url: str | None = None,
        app_title: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the provider with either an API key or prebuilt client.

        Args:
            api_key: Requesty API key. Falls back to `REQUESTY_API_KEY`
                when omitted and required unless `openai_client` is provided.
            app_url: Optional url for app attribution. Falls back to
                `REQUESTY_APP_URL` when omitted.
            app_title: Optional title for app attribution. Falls back to
                `REQUESTY_APP_TITLE` when omitted.
            openai_client: Existing `AsyncOpenAI` client to reuse instead of
                creating one internally.
            http_client: Custom `httpx.AsyncClient` to pass into the
                `AsyncOpenAI` constructor when building a client.

        Raises:
            UserError: If no API key is available and no `openai_client` is
                provided.
        """
        api_key = api_key or os.getenv('REQUESTY_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `REQUESTY_API_KEY` environment variable or pass it via `RequestyProvider(api_key=...)`'
                ' to use the Requesty provider.'
            )

        attribution_headers: dict[str, str] = {}
        if http_referer := app_url or os.getenv('REQUESTY_APP_URL'):
            attribution_headers['HTTP-Referer'] = http_referer
        if x_title := app_title or os.getenv('REQUESTY_APP_TITLE'):
            attribution_headers['X-Title'] = x_title

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=attribution_headers
            )
        else:
            http_client = create_async_http_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_http_client
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=attribution_headers
            )

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
