from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Cerebras provider, '
        'you can use the `cerebras` optional group — `pip install "pydantic-ai-slim[cerebras]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )


class CerebrasProvider(_OpenAICompatibleProvider):
    """Provider for Cerebras API."""

    @property
    def name(self) -> str:
        return 'cerebras'

    @property
    def base_url(self) -> str:
        return 'https://api.cerebras.ai/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        prefix_to_profile = {
            'llama': meta_model_profile,
            'qwen': qwen_model_profile,
            'gpt-oss': harmony_model_profile,
            'zai': zai_model_profile,
        }

        # Reasoning models that support the cerebras_disable_reasoning setting
        reasoning_prefixes = ('zai', 'gpt-oss')

        profile = None
        model_name_lower = model_name.lower()
        for prefix, profile_func in prefix_to_profile.items():
            if model_name_lower.startswith(prefix):
                profile = profile_func(model_name_lower)
                break

        is_reasoning = model_name_lower.startswith(reasoning_prefixes)
        # gpt-oss reasons unconditionally on Cerebras: `disable_reasoning=True` is rejected with a 400,
        # so `thinking=False` must be silently ignored rather than emitted. zai-glm-4.7 can still disable.
        is_always_on_reasoning = model_name_lower.startswith('gpt-oss')
        # GLM requires prior reasoning to be replayed inside `<think>...</think>` tags in the assistant
        # message content, not in a separate `reasoning` field; gpt-oss follows Harmony rules and keeps `'auto'`.
        # https://inference-docs.cerebras.ai/capabilities/reasoning
        send_back_thinking_parts = 'tags' if model_name_lower.startswith('zai') else 'auto'
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
            OpenAIModelProfile(
                # Cerebras accepts `logit_bias` and validates it — a map over 100 entries is a 400 — but
                # never applies it: biasing a token by 100 in either direction leaves the returned logprobs
                # bit-identical. Forwarding it would turn today's silent no-op into a hard error on large
                # bias maps, so it stays stripped even though the API reference documents it as supported.
                openai_unsupported_model_settings=('logit_bias',),
                supports_thinking=is_reasoning,
                thinking_always_enabled=is_always_on_reasoning,
                openai_chat_send_back_thinking_parts=send_back_thinking_parts,
            ),
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
        """Create a new Cerebras provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `CEREBRAS_API_KEY` environment variable
                will be used if available.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('CEREBRAS_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `CEREBRAS_API_KEY` environment variable or pass it via `CerebrasProvider(api_key=...)` '
                'to use the Cerebras provider.'
            )

        default_headers = {'X-Cerebras-3rd-Party-Integration': 'pydantic-ai'}

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
            )
