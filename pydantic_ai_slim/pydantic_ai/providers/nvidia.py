from __future__ import annotations as _annotations

import os
from collections.abc import Callable
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
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile
from pydantic_ai.providers import missing_api_key_error

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
        _OpenAISDKProvider,  # pyright: ignore[reportPrivateUsage]
    )


_HOSTED_BASE_URL = 'https://integrate.api.nvidia.com/v1'


class NvidiaProvider(_OpenAISDKProvider):
    """Provider for NVIDIA NIM, hosted on `build.nvidia.com` or self-hosted."""

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
        # NVIDIA NIM serves models from many publishers under `<publisher>/<model>` ids,
        # so the publisher's family profile is chosen from the model-name prefix.
        publisher_to_profile: dict[str, Callable[[str], ModelProfile | None]] = {
            'meta': meta_model_profile,
            'deepseek-ai': deepseek_model_profile,
            'qwen': qwen_model_profile,
            'google': google_model_profile,
            'openai': harmony_model_profile,  # used for gpt-oss models on NVIDIA NIM
            'mistralai': mistral_model_profile,
            'moonshotai': moonshotai_model_profile,
            'z-ai': zai_model_profile,
        }

        publisher, _, family_model_name = model_name.lower().partition('/')
        family_profile = publisher_to_profile.get(publisher)
        profile = family_profile(family_model_name) if family_profile else None

        # As the NVIDIA NIM API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer,
        # unless json_schema_transformer is set explicitly.
        return merge_profile(OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer), profile)

    @overload
    def __init__(self, *, base_url: str | None = None) -> None: ...

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
            api_key: The API key for authentication. Defaults to the `NVIDIA_API_KEY`
                environment variable. Required for the hosted API; a self-hosted NIM deployment
                that accepts anonymous requests needs none.
            base_url: The base URL for requests. Defaults to the `NVIDIA_BASE_URL`
                environment variable, then `https://integrate.api.nvidia.com/v1`.
                Use the URL of your own NIM deployment for self-hosted models.
            openai_client: An existing `AsyncOpenAI` client to use. Its base URL is used
                instead of `NVIDIA_BASE_URL`. Cannot be combined with `base_url`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        if openai_client is not None and base_url is not None:
            raise UserError('Cannot provide both `openai_client` and `base_url`')

        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
        else:
            self._base_url = base_url or os.getenv('NVIDIA_BASE_URL') or _HOSTED_BASE_URL
            api_key = api_key or os.getenv('NVIDIA_API_KEY')
            if not api_key:
                if self._base_url.rstrip('/') == _HOSTED_BASE_URL:
                    raise missing_api_key_error(
                        'Set the `NVIDIA_API_KEY` environment variable or pass it via '
                        '`NvidiaProvider(api_key=...)` to use the NVIDIA provider.'
                    )
                # Self-hosted NIM may accept anonymous requests, but the OpenAI client requires a non-empty key.
                api_key = 'api-key-not-set'
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
