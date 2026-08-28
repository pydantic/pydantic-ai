from __future__ import annotations as _annotations

import os
from collections.abc import Callable
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Synthorai provider, '
        'you can use the `synthorai` optional group — `pip install "pydantic-ai-slim[synthorai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )


class SynthoraiProvider(_OpenAICompatibleProvider):
    """Provider for Synthorai, an OpenAI-compatible gateway."""

    @property
    def name(self) -> str:
        return 'synthorai'

    @property
    def base_url(self) -> str:
        return 'https://synthorai.io/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        # Synthorai model ids carry no vendor prefix - they are flat names such as
        # `claude-opus-5` or `glm-5.2` - so the family is matched on a leading
        # substring rather than split on '/' the way vendor-prefixed providers do.
        # Longest prefix first, so `gpt-oss` picks the harmony profile before the
        # shorter `gpt-` entry can claim it.
        prefix_to_profile: dict[str, Callable[[str], ModelProfile | None]] = {
            'claude-': anthropic_model_profile,
            'deepseek-': deepseek_model_profile,
            'gemini-': google_model_profile,
            'glm-': zai_model_profile,
            'kimi-': moonshotai_model_profile,
            'qwen': qwen_model_profile,
        }
        # `gpt-` is absent on purpose: it resolves to the OpenAI-compatible base, which is
        # what falling through already produces, so an entry here would only restate it.
        # The families the catalog serves that have no profile in this repository - minimax,
        # hunyuan and the Seed models - fall through for a different reason: mapping them to
        # an approximate profile would claim capabilities nobody has checked.

        profile = None
        lowered = model_name.lower()
        # Longest prefix first, so a more specific entry cannot be shadowed by a shorter one.
        for prefix in sorted(prefix_to_profile, key=len, reverse=True):
            if lowered.startswith(prefix):
                profile = prefix_to_profile[prefix](lowered)
                break

        # Supplies `json_schema_transformer` as a base so an OpenAI-shaped transform is
        # present when the family profile does not set one; a profile that does set it
        # wins, since it is merged on top.
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
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
        """Create a new Synthorai provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `SYNTHORAI_API_KEY`
                environment variable will be used if available.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('SYNTHORAI_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `SYNTHORAI_API_KEY` environment variable or pass it via '
                '`SynthoraiProvider(api_key=...)` to use the Synthorai provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
