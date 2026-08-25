from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Crusoe provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )


class CrusoeProvider(_OpenAICompatibleProvider):
    """Provider for Crusoe Serverless Inference API."""

    @property
    def name(self) -> str:
        return 'crusoe'

    @property
    def base_url(self) -> str:
        return 'https://api.inference.crusoecloud.com/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        vendor_to_profile = {
            'meta-llama': meta_model_profile,
            'deepseek-ai': deepseek_model_profile,
            'qwen': qwen_model_profile,
            'google': google_model_profile,
            'openai': harmony_model_profile,  # used for gpt-oss models on Crusoe
            'moonshotai': moonshotai_model_profile,
            'zai': zai_model_profile,
        }

        profile = None

        model_name = model_name.lower()
        if '/' in model_name:
            vendor, model_name = model_name.split('/', 1)
            if vendor in vendor_to_profile:
                profile = vendor_to_profile[vendor](model_name)

        # `json_schema_transformer` is a fallback (the model family's profile wins if it sets one), as
        # `CrusoeProvider` is always used with `OpenAIChatModel`, which used to unconditionally use
        # `OpenAIJsonSchemaTransformer`.
        # The structured output flags win on top: Crusoe serves every model with guided decoding, so
        # `response_format` works even for families whose own profiles don't claim support for it.
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
            ModelProfile(supports_json_schema_output=True, supports_json_object_output=True),
        )

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: _OpenAIHTTPClient) -> None: ...

    @overload
    def __init__(self, *, http_client: _OpenAIHTTPClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new Crusoe provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `CRUSOE_API_KEY` environment
                variable will be used if available.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('CRUSOE_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `CRUSOE_API_KEY` environment variable or pass it via '
                '`CrusoeProvider(api_key=...)` to use the Crusoe provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
