from __future__ import annotations as _annotations

import os
from dataclasses import replace
from typing import overload

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai._json_schema import JsonSchema, JsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the OpenRouter provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class _OpenRouterGoogleJsonSchemaTransformer(JsonSchemaTransformer):
    """Legacy Google JSON schema transformer for OpenRouter compatibility.

    OpenRouter's compatibility layer doesn't fully support modern JSON Schema features
    like $defs/$ref and anyOf for nullable types. This transformer restores v1.19.0
    behavior by inlining definitions and simplifying nullable unions.

    See: https://github.com/pydantic/pydantic-ai/issues/3617
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict, prefer_inlined_defs=True, simplify_nullable_unions=True)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # Remove properties not supported by Gemini
        schema.pop('$schema', None)
        schema.pop('title', None)
        schema.pop('discriminator', None)
        schema.pop('examples', None)
        schema.pop('exclusiveMaximum', None)
        schema.pop('exclusiveMinimum', None)

        if (const := schema.pop('const', None)) is not None:
            schema['enum'] = [const]

        # Convert enums to string type (legacy Gemini requirement)
        if enum := schema.get('enum'):
            schema['type'] = 'string'
            schema['enum'] = [str(val) for val in enum]

        # Convert oneOf to anyOf for discriminated unions
        if 'oneOf' in schema and 'type' not in schema:
            schema['anyOf'] = schema.pop('oneOf')

        # Handle string format -> description
        type_ = schema.get('type')
        if type_ == 'string' and (fmt := schema.pop('format', None)):
            description = schema.get('description')
            if description:
                schema['description'] = f'{description} (format: {fmt})'
            else:
                schema['description'] = f'Format: {fmt}'

        return schema


def _openrouter_google_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Google model accessed via OpenRouter.

    Uses the legacy transformer to maintain compatibility with OpenRouter's
    translation layer, which doesn't fully support modern JSON Schema features.
    """
    profile = google_model_profile(model_name)
    if profile is None:  # pragma: no cover
        return None
    return replace(profile, json_schema_transformer=_OpenRouterGoogleJsonSchemaTransformer)


class OpenRouterProvider(Provider[AsyncOpenAI]):
    """Provider for OpenRouter API."""

    @property
    def name(self) -> str:
        return 'openrouter'

    @property
    def base_url(self) -> str:
        return 'https://openrouter.ai/api/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'google': _openrouter_google_model_profile,
            'openai': openai_model_profile,
            'anthropic': anthropic_model_profile,
            'mistralai': mistral_model_profile,
            'qwen': qwen_model_profile,
            'x-ai': grok_model_profile,
            'cohere': cohere_model_profile,
            'amazon': amazon_model_profile,
            'deepseek': deepseek_model_profile,
            'meta-llama': meta_model_profile,
            'moonshotai': moonshotai_model_profile,
        }

        profile = None

        provider, model_name = model_name.split('/', 1)
        if provider in provider_to_profile:
            model_name, *_ = model_name.split(':', 1)  # drop tags
            profile = provider_to_profile[provider](model_name)

        # As OpenRouterProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_chat_send_back_thinking_parts='field',
            openai_chat_thinking_field='reasoning',
        ).update(profile)

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        app_url: str | None = None,
        app_title: str | None = None,
        openai_client: None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        app_url: str | None = None,
        app_title: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the provider with either an API key or prebuilt client.

        Args:
            api_key: OpenRouter API key. Falls back to ``OPENROUTER_API_KEY``
                when omitted and required unless ``openai_client`` is provided.
            app_url: Optional url for app attribution. Falls back to
                ``OPENROUTER_APP_URL`` when omitted.
            app_title: Optional title for app attribution. Falls back to
                ``OPENROUTER_APP_TITLE`` when omitted.
            openai_client: Existing ``AsyncOpenAI`` client to reuse instead of
                creating one internally.
            http_client: Custom ``httpx.AsyncClient`` to pass into the
                ``AsyncOpenAI`` constructor when building a client.

        Raises:
            UserError: If no API key is available and no ``openai_client`` is
                provided.
        """
        api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `OPENROUTER_API_KEY` environment variable or pass it via `OpenRouterProvider(api_key=...)`'
                'to use the OpenRouter provider.'
            )

        attribution_headers: dict[str, str] = {}
        if http_referer := app_url or os.getenv('OPENROUTER_APP_URL'):
            attribution_headers['HTTP-Referer'] = http_referer
        if x_title := app_title or os.getenv('OPENROUTER_APP_TITLE'):
            attribution_headers['X-Title'] = x_title

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=attribution_headers
            )
        else:
            http_client = cached_async_http_client(provider='openrouter')
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=attribution_headers
            )
