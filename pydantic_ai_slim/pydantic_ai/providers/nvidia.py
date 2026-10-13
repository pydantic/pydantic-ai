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
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.nvidia import nvidia_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the NVIDIA provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )


class NVIDIAProvider(_OpenAICompatibleProvider):
    """Provider for NVIDIA NIM APIs, hosted on build.nvidia.com or self-hosted."""

    @property
    def name(self) -> str:
        return 'nvidia'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        vendor_to_profile = {
            'nvidia': nvidia_model_profile,
            'meta': meta_model_profile,
            'google': google_model_profile,
            'mistralai': mistral_model_profile,
            'deepseek-ai': deepseek_model_profile,
            'qwen': qwen_model_profile,
            'moonshotai': moonshotai_model_profile,
            'z-ai': zai_model_profile,
            'openai': harmony_model_profile,  # used for gpt-oss models on NVIDIA NIM
        }

        profile = None

        model_name = model_name.lower()
        if '/' in model_name:
            vendor, model_name = model_name.split('/', 1)
            if vendor in vendor_to_profile:
                profile = vendor_to_profile[vendor](model_name)

        # As NVIDIA NIM APIs are OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer,
        # unless json_schema_transformer is set explicitly by the model family's profile.
        return merge_profile(OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer), profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: _OpenAIHTTPClient, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new NVIDIA provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `NVIDIA_API_KEY` environment
                variable will be used if available.
            base_url: The base URL for the NVIDIA NIM API, for example a self-hosted NIM.
                Defaults to `https://integrate.api.nvidia.com/v1`.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key`, `base_url` and
                `http_client` are ignored.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('NVIDIA_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `NVIDIA_API_KEY` environment variable or pass it via '
                '`NVIDIAProvider(api_key=...)` to use the NVIDIA provider.'
            )

        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
        else:
            self._base_url = base_url or 'https://integrate.api.nvidia.com/v1'
            self._client = self._create_openai_client(base_url=self._base_url, api_key=api_key, http_client=http_client)
