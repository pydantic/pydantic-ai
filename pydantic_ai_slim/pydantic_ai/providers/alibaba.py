from __future__ import annotations as _annotations

import os
from typing import overload

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Alibaba provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class AlibabaProvider(Provider[AsyncOpenAI]):
    """Provider for Alibaba Cloud Model Studio (DashScope) OpenAI-compatible API."""

    @property
    def name(self) -> str:
        return 'alibaba'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        base_profile = qwen_model_profile(model_name)

        # Wrap/merge into OpenAIModelProfile
        openai_profile = OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(base_profile)

        # For Qwen Omni models, force URI audio input encoding
        if 'omni' in model_name.lower():
            openai_profile = OpenAIModelProfile(openai_chat_audio_input_encoding='uri').update(openai_profile)

        return openai_profile

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str, base_url: str | None = None, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
        else:
            # NOTE: We support DASHSCOPE_API_KEY for compatibility with Alibaba's official docs.
            api_key = api_key or os.getenv('ALIBABA_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
            if not api_key:
                raise UserError(
                    'Set the `ALIBABA_API_KEY` environment variable or pass it via '
                    '`AlibabaProvider(api_key=...)` to use the Alibaba provider.'
                )

            self._base_url = base_url or 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'

            if http_client is None:
                http_client = cached_async_http_client(provider='alibaba')

            self._client = AsyncOpenAI(base_url=self._base_url, api_key=api_key, http_client=http_client)
