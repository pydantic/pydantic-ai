from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles._family import ModelProfileFunc, vendor_family_model_profile
from pydantic_ai.profiles._special_tokens import special_token_route_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the bf16.ai provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        _OpenAISDKProvider,  # pyright: ignore[reportPrivateUsage]
    )

__all__ = ('BF16Provider',)


def _bf16_family_profile(model_name: str) -> ModelProfile | None:
    """Resolve bf16-namespaced quantized/tuned checkpoints (e.g. `bf16/deepseek-v4.1-flash-...`) to their base family."""
    if model_name.startswith('deepseek'):
        return deepseek_model_profile(model_name)
    return None


_BF16_VENDOR_PROFILES: dict[str, ModelProfileFunc] = {
    'bf16': _bf16_family_profile,
}


class BF16Provider(_OpenAISDKProvider):
    """Provider for the [bf16.ai](https://bf16.ai) OpenAI-compatible open-weight inference API.

    bf16.ai serves 1M-context open-weight models addressed as `creator/model`
    (for example `deepseek/deepseek-v4.1-flash-uncensored`, `zai-org/glm-5.3-uncensored`,
    and `moonshotai/kimi-k3.1-pro-uncensored`) with Zero Data Retention (ZDR).
    """

    @property
    def name(self) -> str:
        return 'bf16'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        profile = vendor_family_model_profile(model_name, overrides=_BF16_VENDOR_PROFILES)

        # `OpenAIJsonSchemaTransformer` is the fallback for the OpenAI-compatible API; a family's own transformer wins.
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
            special_token_route_profile('bf16', model_name),
        )

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
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new bf16.ai provider.

        Args:
            api_key: A bf16.ai API key. Defaults to the `BF16_API_KEY` environment variable.
            base_url: Optional custom base URL. Defaults to the `BF16_BASE_URL` environment variable,
                falling back to `https://api.bf16.ai/v1`.
            openai_client: An existing `AsyncOpenAI` client to use. Cannot be combined with `base_url`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('BF16_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `BF16_API_KEY` environment variable or pass it via '
                '`BF16Provider(api_key=...)` to use the bf16.ai provider.'
            )

        assert openai_client is None or base_url is None, 'Cannot provide both `openai_client` and `base_url`'

        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
        else:
            self._base_url = base_url or os.getenv('BF16_BASE_URL') or 'https://api.bf16.ai/v1'
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
