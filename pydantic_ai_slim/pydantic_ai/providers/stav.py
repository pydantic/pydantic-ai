from __future__ import annotations as _annotations

import os
from typing import Literal, overload

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
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Stav provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        _OpenAISDKProvider,  # pyright: ignore[reportPrivateUsage]
    )

__all__ = ['StavProvider']

_RUNS_IN_HEADER = 'X-Stav-Runs-In'


def _openai_model_profile(model_name: str) -> ModelProfile | None:
    # Stav serves OpenAI's open-weight gpt-oss models alongside the proprietary ones.
    if model_name.startswith('gpt-oss'):
        return harmony_model_profile(model_name)
    return openai_model_profile(model_name)


class StavProvider(_OpenAISDKProvider):
    """Provider for the [Stav](https://stav.ai) European AI inference gateway.

    Stav routes OpenAI-compatible requests to models from many labs, addressed as `creator/model`
    (for example `deepseek-ai/deepseek-v4-flash-0731`), and can keep compute inside a chosen region.
    """

    @property
    def name(self) -> str:
        return 'stav'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'anthropic': anthropic_model_profile,
            'deepseek-ai': deepseek_model_profile,
            'google': google_model_profile,
            'meta-llama': meta_model_profile,
            'mistral-ai': mistral_model_profile,
            'mistralai': mistral_model_profile,
            'moonshotai': moonshotai_model_profile,
            'openai': _openai_model_profile,
            'qwen': qwen_model_profile,
            'x-ai': grok_model_profile,
            'xai': grok_model_profile,
            'zai-org': zai_model_profile,
        }

        profile = None

        model_name = model_name.lower()
        if '/' in model_name:
            provider, model_name = model_name.split('/', 1)
            if provider in provider_to_profile:
                profile = provider_to_profile[provider](model_name)

        # As StavProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly
        return merge_profile(OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer), profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        region: Literal['eu', 'eea'] | str | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        region: Literal['eu', 'eea'] | str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new Stav provider.

        Args:
            api_key: The API key to use for authentication. If not provided, the `STAV_API_KEY` environment variable
                will be used if available.
            base_url: The base URL of the Stav API. If not provided, the `STAV_BASE_URL` environment variable
                will be used if available, falling back to `https://api.stav.ai/v1`.
            region: Where Stav may run requests: `'eu'`, `'eea'`, or an ISO 3166-1 alpha-2 country code.
                `None` (the default) lets Stav route globally. The region is sent as the `X-Stav-Runs-In` header
                on every request, and can only narrow a region already set on the API key or workspace.
                The header requires a Stav gateway release that reads it; on an older one, pass
                `extra_body={'provider': {'runs_in': 'eu'}}` in the model settings instead.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key`, `base_url`, `region`
                and `http_client` must be `None`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        if openai_client is not None:
            if region is not None:
                raise UserError(
                    'Cannot set `region` together with `openai_client`; '
                    f'set the `{_RUNS_IN_HEADER}` header on the client instead.'
                )
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
            return

        api_key = api_key or os.getenv('STAV_API_KEY')
        if not api_key:
            raise UserError(
                'Set the `STAV_API_KEY` environment variable or pass it via `StavProvider(api_key=...)` '
                'to use the Stav provider.'
            )

        self._base_url = base_url or os.getenv('STAV_BASE_URL', 'https://api.stav.ai/v1')
        default_headers = {_RUNS_IN_HEADER: region} if region is not None else None
        self._client = self._create_openai_client(
            base_url=self._base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
        )
