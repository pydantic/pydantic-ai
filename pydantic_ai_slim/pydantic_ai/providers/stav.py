from __future__ import annotations as _annotations

import os
from collections.abc import Sequence
from typing import Literal, overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles._family import vendor_family_model_profile
from pydantic_ai.profiles._special_tokens import special_token_route_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

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


def _runs_in_header_value(region: str | Sequence[str] | None) -> str | None:
    if region is None:
        return None
    parts = [region] if isinstance(region, str) else list(region)
    parts = [part.strip() for part in parts]
    if not parts or not all(parts):
        raise UserError("`region` must be a region or a non-empty list of regions, like `'eu'` or `['no', 'se']`.")
    return ','.join(parts)


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
        # Stav spells Mistral's creator `mistral-ai`; the shared map knows the other creators Stav routes to.
        profile = vendor_family_model_profile(model_name, overrides={'mistral-ai': mistral_model_profile})

        # As StavProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly.
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
            special_token_route_profile('stav', model_name),
        )

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        region: Literal['eu', 'eea'] | str | Sequence[str] | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        region: Literal['eu', 'eea'] | str | Sequence[str] | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new Stav provider.

        Args:
            api_key: The API key to use for authentication. If not provided, the `STAV_API_KEY` environment variable
                will be used if available.
            base_url: The base URL of the Stav API. If not provided, the `STAV_BASE_URL` environment variable
                will be used if available, falling back to `https://api.stav.ai/v1`.
            region: The jurisdiction Stav may run requests in: `'eu'`, `'eea'`, an ISO 3166-1 alpha-2 country
                code such as `'no'`, or a list of these (`['no', 'se']`), which allows any of them. This is a
                jurisdiction, not a cloud site like Bedrock's `region_name='eu-north-1'`. `None` (the default)
                lets Stav route globally. The region is sent as the `X-Stav-Runs-In` header on every request,
                and can only narrow the workspace's own region policy. Stav refuses a request it
                cannot serve inside the region rather than sending it elsewhere.
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

        self._base_url = base_url or os.getenv('STAV_BASE_URL') or 'https://api.stav.ai/v1'
        runs_in = _runs_in_header_value(region)
        default_headers = {_RUNS_IN_HEADER: runs_in} if runs_in is not None else None
        self._client = self._create_openai_client(
            base_url=self._base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
        )
