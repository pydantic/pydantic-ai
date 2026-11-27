from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai._http import AsyncHTTPClient
from pydantic_ai._json_schema import JsonSchema, JsonSchemaTransformerWithOutputOverride
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles import InlineDefsJsonSchemaTransformer, merge_profile
from pydantic_ai.profiles._special_tokens import special_token_route_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.groq import groq_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import (
    _GptOssJsonSchemaTransformer,  # pyright: ignore[reportPrivateUsage]
    openai_model_profile,
)
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers import Provider

try:
    from groq import AsyncGroq
except ImportError as _import_error:
    raise ImportError(
        'Please install the `groq` package to use the Groq provider, '
        'you can use the `groq` optional group — `pip install "pydantic-ai-slim[groq]"`'
    ) from _import_error

# Below the guard on purpose: `groq` requires `httpx`, so without the extra the error above
# is what users should see, not `ModuleNotFoundError: httpx`.
import httpx


def groq_moonshotai_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an MoonshotAI model used with the Groq provider."""
    return merge_profile(
        ModelProfile(supports_json_object_output=True, supports_json_schema_output=True),
        moonshotai_model_profile(model_name),
    )


def meta_groq_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Meta model used with the Groq provider."""
    if model_name in {'llama-4-maverick-17b-128e-instruct', 'llama-4-scout-17b-16e-instruct'}:
        return merge_profile(
            ModelProfile(supports_json_object_output=True, supports_json_schema_output=True),
            meta_model_profile(model_name),
        )
    else:
        return meta_model_profile(model_name)


class GroqProvider(Provider[AsyncGroq]):
    """Provider for Groq API."""

    @property
    def name(self) -> str:
        return 'groq'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncGroq:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile:
        prefix_to_profile = {
            'llama': meta_model_profile,
            'meta-llama/': meta_groq_model_profile,
            'gemma': google_model_profile,
            'qwen': qwen_model_profile,
            'deepseek': deepseek_model_profile,
            'mistral': mistral_model_profile,
            'moonshotai/': groq_moonshotai_model_profile,
            'compound-': groq_model_profile,
            'openai/': openai_model_profile,
        }

        model_name = model_name.lower()
        profile: ModelProfile | None = None
        for prefix, profile_func in prefix_to_profile.items():
            if model_name.startswith(prefix):
                family_name = model_name[len(prefix) :] if prefix.endswith('/') else model_name
                profile = profile_func(family_name)
                break

        # The generic family profiles above don't know Groq's serving specifics for reasoning
        # (e.g. `qwen/qwen3-*` reasons, and Groq's `openai/gpt-oss-*` reasons, but the generic Qwen/OpenAI
        # profiles flag them differently or not at all). Groq is authoritative here, so the Groq profile's
        # reasoning flags override the family profile — it's layered *after* it. The Groq profile also turns
        # JSON-schema output on for some models, never off. The family profile still provides all its other
        # (non-`groq_`, non-reasoning) traits, which the Groq profile doesn't touch.
        # Maintenance contract: because this makes `groq_model_profile`'s reasoning detection authoritative,
        # its `is_reasoning_model` list must stay complete for every Groq-served reasoning model — a model the
        # list misses would have any family-profile `supports_thinking=True` overridden to `False` here.
        return merge_profile(
            profile,
            groq_model_profile(model_name),
            # Groq documents JSON mode for every model it serves.
            ModelProfile(
                # Groq doesn't keep a mid-conversation `system` message in place on gpt-oss, so a request that adds
                # one only reads the cached prefix up to the instructions and tools; the same text as `<system>` in
                # a user turn keeps the whole prefix cached: https://github.com/pydantic/pydantic-ai/issues/14516
                supports_inline_system_prompts=not model_name.startswith('openai/gpt-oss-'),
                supports_json_object_output=True,
            ),
            ModelProfile(json_schema_transformer=_GroqQwenStrictOutputJsonSchemaTransformer)
            if model_name.startswith('qwen/qwen3.8-')
            else None,
            ModelProfile(json_schema_transformer=_GroqQwenBestEffortOutputJsonSchemaTransformer)
            if model_name.startswith('qwen/qwen3.6-')
            else None,
            special_token_route_profile('groq', model_name),
        )

    @overload
    def __init__(self, *, groq_client: AsyncGroq | None = None) -> None: ...

    @overload
    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None, http_client: httpx.AsyncClient | None = None
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        groq_client: AsyncGroq | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Groq provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `GROQ_API_KEY` environment variable
                will be used if available.
            base_url: The base url for the Groq requests. If not provided, the `GROQ_BASE_URL` environment variable
                will be used if available. Otherwise, defaults to Groq's base url.
            groq_client: An existing
                [`AsyncGroq`](https://github.com/groq/groq-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `AsyncClient` to use for making HTTP requests.
        """
        if groq_client is not None:
            assert http_client is None, 'Cannot provide both `groq_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `groq_client` and `api_key`'
            assert base_url is None, 'Cannot provide both `groq_client` and `base_url`'
            self._client = groq_client
        else:
            api_key = api_key or os.getenv('GROQ_API_KEY')
            base_url = base_url or os.getenv('GROQ_BASE_URL', 'https://api.groq.com')

            if not api_key:
                raise UserError(
                    'Set the `GROQ_API_KEY` environment variable or pass it via `GroqProvider(api_key=...)`'
                    ' to use the Groq provider.'
                )
            elif http_client is not None:
                self._client = AsyncGroq(base_url=base_url, api_key=api_key, http_client=http_client)
            else:
                http_client = create_async_http_client()
                self._own_http_client = http_client
                self._http_client_factory = create_async_http_client
                self._client = AsyncGroq(base_url=base_url, api_key=api_key, http_client=http_client)

    def _set_http_client(self, http_client: AsyncHTTPClient) -> None:
        assert isinstance(http_client, httpx.AsyncClient)
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]


class _GroqQwenStrictOutputJsonSchemaTransformer(
    InlineDefsJsonSchemaTransformer, JsonSchemaTransformerWithOutputOverride
):
    """Qwen's family transformer for tool schemas, and gpt-oss's strict one for the output object.

    Groq lists qwen3.8 for `strict: true`, which requires every field to be required and every object to set
    `additionalProperties: false` (https://console.groq.com/docs/structured-outputs); qwen's family transformer only
    inlines `$defs`. `GroqModel` sends `strict` only on the output object's `json_schema` response format, never on a
    tool, so tool schemas keep the family transformer.
    """

    output_object_transformer = _GptOssJsonSchemaTransformer


class _GroqBestEffortJsonSchemaTransformer(InlineDefsJsonSchemaTransformer):
    """Qwen's family transformer, reporting a schema strict only when the caller asks for strict mode."""

    def walk(self) -> JsonSchema:
        schema = super().walk()
        # `_customize_output_object()` coalesces an unset `strict` to this value, so the output object goes out
        # best-effort unless `NativeOutput(..., strict=True)` is passed.
        self.is_strict_compatible = self.strict is True
        return schema


class _GroqQwenBestEffortOutputJsonSchemaTransformer(
    InlineDefsJsonSchemaTransformer, JsonSchemaTransformerWithOutputOverride
):
    """Qwen's family transformer for tool schemas, and best-effort mode for the output object.

    Groq doesn't list qwen3.6 for `strict: true` (https://console.groq.com/docs/structured-outputs), and qwen's family
    transformer doesn't produce the `additionalProperties: false` on every object that strict mode requires, so its
    output object goes out with `strict: false` unless the caller passes `NativeOutput(..., strict=True)`.
    """

    output_object_transformer = _GroqBestEffortJsonSchemaTransformer
