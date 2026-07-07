from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Snowflake provider, '
        'you can use the `snowflake` optional group — `pip install "pydantic-ai-slim[snowflake]"`'
    ) from _import_error


class SnowflakeModelProfile(OpenAIModelProfile, total=False):
    """Profile for models used with `SnowflakeModel`.

    ALL FIELDS MUST BE `snowflake_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    snowflake_supports_reasoning: bool
    """Whether the model supports the OpenRouter-style `reasoning` request object (Claude models)."""

    snowflake_reasoning_requires_temperature_1: bool
    """Whether the model requires `temperature` to be exactly 1 when reasoning is enabled.

    Cortex applies a non-1 default temperature server-side, so `SnowflakeModel` sets `temperature`
    explicitly when reasoning is enabled and the user didn't set it.
    """


class SnowflakeProvider(Provider[AsyncOpenAI]):
    """Provider for [Snowflake Cortex](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api).

    Routes requests through Snowflake's OpenAI-compatible Chat Completions API at
    `https://<account>.snowflakecomputing.com/api/v2/cortex/v1/chat/completions`, which serves
    Claude, GPT, Llama, Mistral, DeepSeek, and Snowflake's own models. All inference runs inside
    the customer's Snowflake account, so data never leaves the Snowflake security perimeter.
    """

    @property
    def name(self) -> str:
        return 'snowflake'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        model_name = model_name.lower()

        family_profile: ModelProfile | None = None
        if model_name.startswith('claude'):
            family_profile = anthropic_model_profile(model_name)
        elif model_name.startswith('openai-'):
            family_profile = openai_model_profile(model_name.removeprefix('openai-'))
        elif 'llama' in model_name:
            family_profile = meta_model_profile(model_name)
        elif model_name.startswith(('mistral', 'mixtral')):
            family_profile = mistral_model_profile(model_name)
        elif model_name.startswith('deepseek'):
            family_profile = deepseek_model_profile(model_name)

        # Cortex does not document `strict` on tool definitions, so we don't send it.
        cortex_profile = OpenAIModelProfile(openai_supports_strict_tool_definition=False)
        if model_name.startswith('claude'):
            # Claude models only support `json_schema` as the response format type, and thinking
            # is requested with an OpenRouter-style `reasoning` object (see `SnowflakeModel`).
            cortex_profile.update(
                SnowflakeModelProfile(
                    supports_json_schema_output=True,
                    supports_json_object_output=False,
                    supports_thinking=True,
                    snowflake_supports_reasoning=True,
                    snowflake_reasoning_requires_temperature_1=True,
                )
            )
        elif not model_name.startswith('openai-'):
            # Cortex accepts `tools` and `response_format` for OpenAI and Claude models only:
            # `tools` is an error and `response_format` is silently ignored for other model families.
            cortex_profile.update(
                OpenAIModelProfile(
                    supports_tools=False,
                    supports_json_schema_output=False,
                    supports_json_object_output=False,
                    default_structured_output_mode='prompted',
                )
            )

        # As `SnowflakeProvider` is always used with `SnowflakeModel`, which is based on
        # `OpenAIChatModel`, we maintain the base `OpenAIJsonSchemaTransformer` unless the family
        # profile sets one explicitly (like `meta_model_profile` does).
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            family_profile,
            cortex_profile,
        )

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        account: str | None = None,
        token: str | None = None,
        base_url: str | None = None,
        openai_client: None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        account: str | None = None,
        token: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Snowflake provider.

        Args:
            account: The [Snowflake account identifier](https://docs.snowflake.com/en/user-guide/admin-account-identifier),
                e.g. `myorg-myaccount`. Defaults to the `SNOWFLAKE_ACCOUNT` environment variable.
            token: A Snowflake [programmatic access token](https://docs.snowflake.com/en/user-guide/programmatic-access-tokens),
                OAuth token, or key-pair JWT, sent as `Authorization: Bearer <token>`.
                Defaults to the `SNOWFLAKE_TOKEN` environment variable.
            base_url: The base URL of the Cortex REST API, e.g. when connecting through
                [private connectivity](https://docs.snowflake.com/en/user-guide/private-snowflake-service).
                Defaults to `https://<account>.snowflakecomputing.com/api/v2/cortex/v1`.
            openai_client: An existing `AsyncOpenAI` client to use. Its `base_url` must already
                point at the Cortex REST API. If provided, `account`, `token`, `base_url`, and
                `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if openai_client is not None:
            assert account is None, 'Cannot provide both `openai_client` and `account`'
            assert token is None, 'Cannot provide both `openai_client` and `token`'
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
            return

        if base_url is None:
            account = account or os.getenv('SNOWFLAKE_ACCOUNT')
            if not account:
                raise UserError(
                    'Set the `SNOWFLAKE_ACCOUNT` environment variable or pass it via `SnowflakeProvider(account=...)`'
                    ' to use the Snowflake provider.'
                )
            # Be resilient to values that include more than the bare account identifier,
            # like `myorg-myaccount.snowflakecomputing.com` or a full URL.
            account = account.removeprefix('https://').removesuffix('/').removesuffix('.snowflakecomputing.com')
            base_url = f'https://{account}.snowflakecomputing.com/api/v2/cortex/v1'
        self._base_url = base_url

        token = token or os.getenv('SNOWFLAKE_TOKEN')
        if not token:
            raise UserError(
                'Set the `SNOWFLAKE_TOKEN` environment variable or pass it via `SnowflakeProvider(token=...)`'
                ' to use the Snowflake provider.'
            )

        if http_client is not None:
            self._client = AsyncOpenAI(base_url=base_url, api_key=token, http_client=http_client)
        else:
            http_client = create_async_http_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_http_client
            self._client = AsyncOpenAI(base_url=base_url, api_key=token, http_client=http_client)

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
