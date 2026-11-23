from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles._family import ModelProfileFunc, split_vendor_model_id, vendor_family_profile
from pydantic_ai.profiles._special_tokens import special_token_route_profile
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.google import google_cloud_model_profile
from pydantic_ai.profiles.groq import groq_model_profile
from pydantic_ai.profiles.openai import (
    OpenAIJsonSchemaTransformer,
    OpenAIModelProfile,
    OpenAIReasoningEffort,
    openai_model_profile,
)

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Vercel provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        _OpenAISDKProvider,  # pyright: ignore[reportPrivateUsage]
    )


_NONE_TO_HIGH: frozenset[OpenAIReasoningEffort] = frozenset({'none', 'low', 'medium', 'high'})
_LOW_TO_HIGH: frozenset[OpenAIReasoningEffort] = frozenset({'low', 'medium', 'high'})
_NONE_OR_HIGH: frozenset[OpenAIReasoningEffort] = frozenset({'none', 'high'})
_NONE_TO_HIGH_WITH_MINIMAL: frozenset[OpenAIReasoningEffort] = frozenset({'none', 'minimal', 'low', 'medium', 'high'})
_EVERY_EFFORT: frozenset[OpenAIReasoningEffort] = frozenset(
    {'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'}
)

_VERCEL_VENDORS = frozenset(
    {
        'anthropic',
        'cohere',
        'deepseek',
        'mistral',
        'moonshotai',
        'nvidia',
        'stepfun',
        'tencent',
        'thinkingmachines',
        'xai',
        'xiaomi',
        'zai',
    }
)
"""The vendors in the shared family map whose profile Vercel ids get."""

_VERCEL_VENDOR_PROFILES: dict[str, ModelProfileFunc] = {
    'bedrock': amazon_model_profile,
    'groq': groq_model_profile,
    'openai': openai_model_profile,
    'vertex': google_cloud_model_profile,
}
"""Vendors whose profile on Vercel differs from the shared family map."""

_OPEN_WEIGHT_THINKING_VENDORS = frozenset({'nvidia', 'stepfun', 'tencent', 'thinkingmachines', 'xiaomi'})

_OPEN_WEIGHT_REASONING_EFFORTS: dict[str, frozenset[OpenAIReasoningEffort]] = {
    'nvidia/nemotron-3-ultra-550b-a55b': _NONE_TO_HIGH,
    'nvidia/nemotron-3.5-lightning': _NONE_TO_HIGH,
    'nvidia/nemotron-3.5-lightning-free': _NONE_TO_HIGH,
    'nvidia/nemotron-3.5-super-vl': _NONE_TO_HIGH,
    'stepfun/step-3.7-flash': _LOW_TO_HIGH,
    'stepfun/step-5-preview': _LOW_TO_HIGH,
    'tencent/hy3': frozenset({'none', 'low', 'high'}),
    'tencent/hy4': _NONE_OR_HIGH,
    'tencent/hy4-flash': _NONE_OR_HIGH,
    'tencent/hy4-preview': _NONE_OR_HIGH,
    'thinkingmachines/inkling': _EVERY_EFFORT,
    'thinkingmachines/inkling-small': _EVERY_EFFORT,
    'xiaomi/mimo-v2.6-flash': _NONE_TO_HIGH_WITH_MINIMAL,
    'xiaomi/mimo-v2.6-pro': _NONE_TO_HIGH_WITH_MINIMAL,
    'xiaomi/mimo-v2.6-pro-ultraspeed': _NONE_TO_HIGH_WITH_MINIMAL,
}
"""The efforts Vercel's model catalog (`reasoning_options` at https://ai-gateway.vercel.sh/v1/models) lists for the
open-weight thinking families' models.

Vercel's docs don't say what it does with an effort a model doesn't list, and ask for "a value supported by the
model's catalog entry" (https://vercel.com/docs/ai-gateway/models-and-providers/reasoning), so `thinking` resolves to
the lowest listed effort at or above the requested level, else the highest. A family model missing here, such as
Nemotron 3 Super, for which the catalog lists no reasoning controls, gets no `reasoning_effort` at all.
"""


class VercelProvider(_OpenAISDKProvider):
    """Provider for Vercel AI Gateway API."""

    @property
    def name(self) -> str:
        return 'vercel'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        route_profile = special_token_route_profile('vercel', model_name)

        # Vercel matches the vendor case-sensitively, unlike the shared family map.
        vendor, model_name = split_vendor_model_id(model_name) or ('', model_name)
        family_profile = vendor_family_profile(vendor, overrides=_VERCEL_VENDOR_PROFILES, vendors=_VERCEL_VENDORS)
        profile = family_profile(model_name) if family_profile else None
        if vendor in _OPEN_WEIGHT_THINKING_VENDORS and profile is not None:
            reasoning_efforts = _OPEN_WEIGHT_REASONING_EFFORTS.get(f'{vendor}/{model_name}', frozenset())
            profile = merge_profile(profile, OpenAIModelProfile(openai_reasoning_efforts=reasoning_efforts))

        # As VercelProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer), profile, route_profile
        )

    @overload
    def __init__(self, *, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str, base_url: str | None = None, http_client: _OpenAIHTTPClient) -> None: ...

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
        # Support Vercel AI Gateway's standard environment variables
        api_key = api_key or os.getenv('VERCEL_AI_GATEWAY_API_KEY') or os.getenv('VERCEL_OIDC_TOKEN')

        if not api_key and openai_client is None:
            raise UserError(
                'Set the `VERCEL_AI_GATEWAY_API_KEY` or `VERCEL_OIDC_TOKEN` environment variable '
                'or pass the API key via `VercelProvider(api_key=...)` to use the Vercel provider.'
            )

        default_headers = {'http-referer': 'https://pydantic.dev/docs/ai/', 'x-title': 'pydantic-ai'}

        assert openai_client is None or base_url is None, 'Cannot provide both `openai_client` and `base_url`'
        self._base_url = base_url or 'https://ai-gateway.vercel.sh/v1'
        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
            )
