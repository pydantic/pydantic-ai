from __future__ import annotations as _annotations

import os
from typing import Literal, overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles._family import (
    ModelProfileFunc,
    _openai_family_profile,  # pyright: ignore[reportPrivateUsage]
    vendor_family_model_profile,
)
from pydantic_ai.profiles._special_tokens import special_token_route_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Otari provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        _OpenAISDKProvider,  # pyright: ignore[reportPrivateUsage]
    )

__all__ = ('OtariProvider', 'OtariRegion')

OtariRegion = Literal['us', 'eu']
"""A region of the hosted Otari gateway at [otari.ai](https://otari.ai)."""

_REGION_BASE_URLS: dict[OtariRegion, str] = {
    'us': 'https://api.otari.ai/api/v1',
    'eu': 'https://eu.api.otari.ai/api/v1',
}

_OTARI_VENDOR_PROFILES: dict[str, ModelProfileFunc] = {
    'azureopenai': _openai_family_profile,
    'gemini': google_model_profile,
    'moonshot': moonshotai_model_profile,
}
"""Otari provider ids that serve one vendor's models but aren't spelled as the shared vendor map spells it.

Hosts of many vendors' models, like `azure`, `bedrock`, `groq` or `vertexai`, have no entry: a `vendor/model` id after
them picks the family, and any other model name gets no family profile.
"""


def _vendor_model_name(model_name: str) -> str:
    """Return an Otari `provider:model` name, or a `vendor/model` catalogue id, as `vendor/model`."""
    vendor, separator, rest = model_name.partition(':')
    if separator and '/' not in vendor:
        return f'{vendor}/{rest}'
    return model_name


class OtariProvider(_OpenAISDKProvider):
    """Provider for [Otari](https://otari.ai), Mozilla.ai's open-source LLM gateway.

    Requests go to the hosted gateway at otari.ai (US region by default), or to a self-hosted gateway
    when `base_url` or the `OTARI_BASE_URL` environment variable is set.
    """

    @property
    def name(self) -> str:
        return 'otari'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        vendor_model_name = _vendor_model_name(model_name)
        profile = vendor_family_model_profile(vendor_model_name, overrides=_OTARI_VENDOR_PROFILES)
        if profile is None:
            # A host of many vendors' models names the vendor after it, as in `together:qwen/qwen3.8-27b`.
            profile = vendor_family_model_profile(vendor_model_name.partition('/')[2])

        # `OpenAIJsonSchemaTransformer` is the fallback for the OpenAI-compatible API; a family's own transformer wins.
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
            special_token_route_profile('otari', vendor_model_name),
        )

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        region: OtariRegion | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        region: OtariRegion | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new Otari provider.

        Args:
            api_key: An Otari API key. Defaults to the `OTARI_API_KEY` environment variable.
            base_url: The `/api/v1` URL of a self-hosted gateway. Defaults to otari.ai in
                `region` when that is set, else to the `OTARI_BASE_URL` environment variable, else to otari.ai's US region.
            region: The otari.ai region to send requests to, `'us'` or `'eu'`.
                Cannot be combined with `base_url`.
            openai_client: An existing `AsyncOpenAI` client to use. Cannot be combined with `base_url` or `region`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('OTARI_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `OTARI_API_KEY` environment variable or pass it via '
                '`OtariProvider(api_key=...)` to use the Otari provider.'
            )

        assert openai_client is None or base_url is None, 'Cannot provide both `openai_client` and `base_url`'
        if region is not None:
            if base_url is not None or openai_client is not None:
                raise UserError('Cannot set `region` together with `base_url` or `openai_client`.')
            base_url = _REGION_BASE_URLS[region]

        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
        else:
            self._base_url = base_url or os.getenv('OTARI_BASE_URL') or _REGION_BASE_URLS['us']
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
