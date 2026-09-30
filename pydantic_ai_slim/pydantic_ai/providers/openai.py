from __future__ import annotations as _annotations

import os
from datetime import timedelta
from typing import TYPE_CHECKING, overload

from pydantic_ai import ModelProfile
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile, openai_realtime_model_profile
from pydantic_ai.providers import missing_api_key_error

if TYPE_CHECKING:
    from pydantic_ai.realtime import RealtimeModelProfile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the OpenAI provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

from ._openai_compatible import (
    AsyncHTTPClient as _OpenAIHTTPClient,
    OpenAICompatibleProvider as _OpenAICompatibleProvider,
)


class OpenAIProvider(_OpenAICompatibleProvider):
    """Provider for OpenAI API.

    Changing `base_url` does not change the OpenAI model profile selection. For another service,
    use its dedicated provider when available. For a custom gateway, subclass this provider and
    override `model_profile`, or configure the model's `profile` explicitly.
    See [OpenAI-compatible models](https://pydantic.dev/docs/ai/models/openai/#openai-compatible-models).
    """

    @property
    def name(self) -> str:
        return 'openai'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        # No per-model gate on `additional_tools`. OpenAI documents a model restriction for the sibling
        # feature ("Only `gpt-5.4` and later models support `tool_search`") and states none for the item,
        # and measurement agrees: 13 models from `gpt-4o-mini` through `gpt-5.6` each called a tool that
        # only an `additional_tools` item declared, 3/3, against 0/3 for a control with the item removed.
        # An earlier list here recorded `gpt-5.4` and `gpt-5` as silently ignoring it; that came from a
        # probe whose prompt named the tool and told the model to call it, which a model that never saw
        # the declaration can satisfy from the prompt text alone.
        #
        # The flag stays here rather than moving into `openai_model_profile`, which is shared with
        # OpenAI-compatible endpoints (Azure, OpenRouter, vLLM, ...) that speak the Responses API without
        # necessarily implementing this item — the same reasoning `openai_supports_phase` documents.
        profile = openai_model_profile(model_name)
        return merge_profile(
            profile,
            OpenAIModelProfile(
                tool_addition_mode='with_definitions',
                tool_deferral_mode='with_tool_search',
                # GPT-5.6 and later document a model-determined floor: a cached prefix stays eligible for
                # reuse for at least 30 minutes after its last write or reuse. These are the models with
                # explicit cache breakpoints. Earlier models have no honest boundary to record, because
                # their retention policy defaults to `24h` for organizations without zero data retention and
                # to `in_memory` (5-10 minutes of inactivity) for organizations with it — an org setting that
                # isn't knowable from the model. Those stay unset (`'unknown'`) rather than guess in either
                # direction: too low a boundary would declare a live cache cold and throw away a real hit, too
                # high a one would report expiry as an unexpected collapse. Explicitly requested retention,
                # such as `openai_prompt_cache_retention='24h'`, is resolved by the model's
                # `resolve_cache_retention`.
                # https://developers.openai.com/api/docs/guides/prompt-caching
                default_cache_retention=timedelta(minutes=30)
                if profile.get('openai_supports_prompt_cache_breakpoints')
                else None,
            ),
        )

    @staticmethod
    def realtime_model_profile(model_name: str) -> RealtimeModelProfile:
        return openai_realtime_model_profile(model_name)

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new OpenAI provider.

        Args:
            base_url: The base url for the OpenAI requests. If not provided, the `OPENAI_BASE_URL` environment variable
                will be used if available. Otherwise, defaults to OpenAI's base url.
            api_key: The API key to use for authentication, if not provided, the `OPENAI_API_KEY` environment variable
                will be used if available.
            openai_client: An existing
                [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        if api_key is None and 'OPENAI_API_KEY' not in os.environ and openai_client is None:
            if base_url is None and 'OPENAI_BASE_URL' not in os.environ:
                # When talking to OpenAI directly, a missing key would otherwise surface as a raw
                # `openai.OpenAIError`; raise our own `UserError` instead so the message is consistent with
                # other providers and points newcomers to the keyless test model.
                raise missing_api_key_error(
                    'Set the `OPENAI_API_KEY` environment variable or pass it via `OpenAIProvider(api_key=...)`'
                    ' to use the OpenAI provider.'
                )
            else:
                # This is a workaround for the OpenAI client requiring an API key, whilst locally served,
                # openai compatible models do not always need an API key, but a placeholder (non-empty) key is required.
                api_key = 'api-key-not-set'

        if openai_client is not None:
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=base_url, api_key=api_key, http_client=http_client)
