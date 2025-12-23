from __future__ import annotations as _annotations

import os
from dataclasses import dataclass
from typing import TypeAlias, overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.providers import Provider

from .._json_schema import JsonSchema, JsonSchemaTransformer

try:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicFoundry, AsyncAnthropicVertex
except ImportError as _import_error:
    raise ImportError(
        'Please install the `anthropic` package to use the Anthropic provider, '
        'you can use the `anthropic` optional group — `pip install "pydantic-ai-slim[anthropic]"`'
    ) from _import_error


AsyncAnthropicClient: TypeAlias = AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicFoundry | AsyncAnthropicVertex


class AnthropicProvider(Provider[AsyncAnthropicClient]):
    """Provider for Anthropic API."""

    @property
    def name(self) -> str:
        return 'anthropic'

    @property
    def base_url(self) -> str:
        return str(self._client.base_url)

    @property
    def client(self) -> AsyncAnthropicClient:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        profile = anthropic_model_profile(model_name)
        return ModelProfile(json_schema_transformer=AnthropicJsonSchemaTransformer).update(profile)

    @overload
    def __init__(self, *, anthropic_client: AsyncAnthropicClient | None = None) -> None: ...

    @overload
    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None, http_client: httpx.AsyncClient | None = None
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        anthropic_client: AsyncAnthropicClient | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Anthropic provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `ANTHROPIC_API_KEY` environment variable
                will be used if available.
            base_url: The base URL to use for the Anthropic API.
            anthropic_client: An existing Anthropic client to use. Accepts
                [`AsyncAnthropic`](https://github.com/anthropics/anthropic-sdk-python),
                [`AsyncAnthropicBedrock`](https://docs.anthropic.com/en/api/claude-on-amazon-bedrock),
                [`AsyncAnthropicFoundry`](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry), or
                [`AsyncAnthropicVertex`](https://docs.anthropic.com/en/api/claude-on-vertex-ai).
                If provided, the `api_key` and `http_client` arguments will be ignored.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if anthropic_client is not None:
            assert http_client is None, 'Cannot provide both `anthropic_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `anthropic_client` and `api_key`'
            self._client = anthropic_client
        else:
            api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise UserError(
                    'Set the `ANTHROPIC_API_KEY` environment variable or pass it via `AnthropicProvider(api_key=...)`'
                    'to use the Anthropic provider.'
                )
            if http_client is not None:
                self._client = AsyncAnthropic(api_key=api_key, base_url=base_url, http_client=http_client)
            else:
                http_client = cached_async_http_client(provider='anthropic')
                self._client = AsyncAnthropic(api_key=api_key, base_url=base_url, http_client=http_client)


@dataclass(init=False)
class AnthropicJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms schemas to the subset supported by Anthropic structured outputs.

    Transformation is applied when:
    - `NativeOutput` is used as the `output_type` of the Agent
    - `strict=True` is set on the `Tool`

    The behavior of this transformer differs from the OpenAI one in that it sets `Tool.strict=False` by default when not explicitly set to True.

    Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('anthropic:claude-sonnet-4-5')

        @agent.tool_plain  # -> defaults to strict=False
        def my_tool(x: str) -> dict[str, int]:
            ...
        ```

    Anthropic's SDK `transform_schema()` automatically:
    - Adds `additionalProperties: false` to all objects (required by API)
    - Removes unsupported constraints (minLength, pattern, etc.)
    - Moves removed constraints to description field
    - Removes title and $schema fields
    """

    def walk(self) -> JsonSchema:
        schema = super().walk()

        # The caller (pydantic_ai.models._customize_tool_def or _customize_output_object) coalesces
        # - output_object.strict = self.is_strict_compatible
        # - tool_def.strict = self.is_strict_compatible
        # the reason we don't default to `strict=True` is that the transformation could be lossy
        # so in order to change the behavior (default to True), we need to come up with logic that will check for lossiness
        # https://github.com/pydantic/pydantic-ai/issues/3541
        self.is_strict_compatible = self.strict is True  # not compatible when strict is False/None

        if self.strict is True:
            from anthropic import transform_schema

            return transform_schema(schema)
        else:
            return schema

    def transform(self, schema: JsonSchema) -> JsonSchema:
        schema.pop('title', None)
        schema.pop('$schema', None)
        return schema
