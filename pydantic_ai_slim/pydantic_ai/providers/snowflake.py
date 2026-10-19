from __future__ import annotations as _annotations

import os
import re
from collections.abc import Callable
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Snowflake provider, '
        'you can use the `snowflake` optional group — `pip install "pydantic-ai-slim[snowflake]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )


class SnowflakeModelProfile(OpenAIModelProfile, total=False):
    """Profile for models used with `SnowflakeModel`.

    ALL FIELDS MUST BE `snowflake_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    snowflake_supports_reasoning: bool
    """Whether the model supports the `reasoning` request object (Claude models)."""

    snowflake_reasoning_requires_temperature_1: bool
    """Whether the model requires `temperature` to be exactly 1 when reasoning is enabled.

    Cortex applies a non-1 default temperature server-side, so `SnowflakeModel` sets `temperature`
    explicitly when reasoning is enabled and the user didn't set it.
    """


class SnowflakeProvider(_OpenAICompatibleProvider):
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

        # Cortex serves bare model names without a provider-prefix delimiter, so match each model to
        # its family by prefix. `snowflake-llama-*` is Meta's Llama fine-tuned by Snowflake, and
        # `openai-*` names carry an `openai-` prefix the OpenAI profile doesn't expect.
        prefix_to_profile: dict[str, Callable[[str], ModelProfile | None]] = {
            'claude': anthropic_model_profile,
            'openai-': lambda name: openai_model_profile(name.removeprefix('openai-')),
            'snowflake-llama': meta_model_profile,
            'llama': meta_model_profile,
            'mistral': mistral_model_profile,
            'mixtral': mistral_model_profile,
            'deepseek': deepseek_model_profile,
        }

        family_profile: ModelProfile | None = None
        for prefix, profile_func in prefix_to_profile.items():
            if model_name.startswith(prefix):
                family_profile = profile_func(model_name)
                break

        # Cortex does not document `strict` on tool definitions, so we don't send it.
        cortex_profile = OpenAIModelProfile(openai_supports_strict_tool_definition=False)
        if model_name.startswith('claude'):
            # Claude models only support `json_schema` as the response format type, and thinking
            # is requested with a `reasoning` object (see `SnowflakeModel`).
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
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        account: str | None = None,
        token: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
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
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
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
            # Accept either a bare account identifier (`myorg-myaccount`) or a value that includes
            # the Snowflake hostname (`myorg-myaccount.snowflakecomputing.com`, with or without a scheme).
            account = account.removeprefix('https://').removeprefix('http://').removesuffix('/')
            account = account.removesuffix('.snowflakecomputing.com')
            # Validate that what's left is a plain account identifier, so a value like
            # `attacker.example/path` can't redirect the authenticated request to another host.
            if not re.fullmatch(r'[A-Za-z0-9._-]+', account):
                raise UserError(
                    f'Invalid Snowflake account identifier {account!r}. Pass a bare account identifier '
                    'like `myorg-myaccount`, or use the `base_url` parameter to target a custom endpoint.'
                )
            base_url = f'https://{account}.snowflakecomputing.com/api/v2/cortex/v1'
        self._base_url = base_url

        token = token or os.getenv('SNOWFLAKE_TOKEN')
        if not token:
            raise UserError(
                'Set the `SNOWFLAKE_TOKEN` environment variable or pass it via `SnowflakeProvider(token=...)`'
                ' to use the Snowflake provider.'
            )

        self._client = self._create_openai_client(base_url=base_url, api_key=token, http_client=http_client)
