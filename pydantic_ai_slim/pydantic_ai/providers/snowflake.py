"""Snowflake Cortex Inference provider."""
from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Snowflake Cortex provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


def _cortex_model_profile(model_name: str) -> ModelProfile:
    """Return a merged model profile for a Snowflake Cortex model."""
    model_lower = model_name.lower()
    family_profile: ModelProfile | None = None

    if model_lower.startswith('claude-'):
        family_profile = anthropic_model_profile(model_lower)
    elif 'llama' in model_lower:
        family_profile = meta_model_profile(model_lower)
    elif model_lower.startswith('mistral') or model_lower.startswith('mixtral'):
        family_profile = mistral_model_profile(model_lower)
    elif model_lower.startswith(('gpt-', 'openai-gpt-')):
        canonical = model_lower.removeprefix('openai-')
        family_profile = openai_model_profile(canonical)

    # Cortex-specific overrides are placed last so they take precedence over
    # any family-level profile settings (e.g. meta_model_profile sets
    # InlineDefsJsonSchemaTransformer, but we always want OpenAI format here).
    return merge_profile(
        family_profile,
        OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            supports_json_object_output=True,
            # Cortex does not support strict tool definitions.
            openai_supports_strict_tool_definition=False,
            # Cortex requires max_completion_tokens, not the deprecated max_tokens.
            openai_chat_supports_max_completion_tokens=True,
        ),
    )


class SnowflakeCortexProvider(Provider[AsyncOpenAI]):
    """Provider for Snowflake Cortex Inference.

    Routes all requests through Snowflake's OpenAI-compatible
    /api/v2/cortex/v1/chat/completions endpoint. Supported models include
    Claude, GPT, Llama, Mistral, DeepSeek, and Snowflake's own models.
    All inference runs inside the customer's Snowflake account — data never
    leaves the Snowflake security perimeter.

    Note:
        For Claude models, Snowflake also exposes an Anthropic-compatible
        /messages endpoint. To use that path, use AnthropicModel with
        AnthropicProvider pointed at your Cortex base URL.

    Example::

        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.snowflake import SnowflakeCortexProvider

        provider = SnowflakeCortexProvider()  # reads SNOWFLAKE_ACCOUNT + SNOWFLAKE_TOKEN
        agent = Agent(OpenAIChatModel('llama4-maverick', provider=provider))

    Environment variables:
        SNOWFLAKE_ACCOUNT: Snowflake account identifier, e.g. myorg-myaccount.
        SNOWFLAKE_TOKEN: A PAT (Programmatic Access Token) or OAuth token.
          Generate a PAT in Snowflake UI: Account > Security > Programmatic Access Tokens.
    """

    _account: str

    @property
    def name(self) -> str:
        return 'snowflake-cortex'

    @property
    def base_url(self) -> str:
        return f'https://{self._account}.snowflakecomputing.com/api/v2/cortex/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return _cortex_model_profile(model_name)

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        account: str | None = None,
        token: str | None = None,
        openai_client: None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        account: str | None = None,
        token: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Snowflake Cortex provider.

        Args:
            account: Snowflake account identifier, e.g. myorg-myaccount.
                Defaults to the SNOWFLAKE_ACCOUNT environment variable.
            token: Snowflake token for authentication (PAT or OAuth token).
                Sent as Authorization: Bearer <token>.
                Defaults to the SNOWFLAKE_TOKEN environment variable.
            openai_client: An existing AsyncOpenAI client. Its base_url must
                already point at the Snowflake Cortex endpoint. When provided,
                account, token, and http_client must all be None.
            http_client: An existing httpx.AsyncClient to use for HTTP requests.
        """
        if openai_client is not None:
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert account is None, 'Cannot provide both `openai_client` and `account`'
            assert token is None, 'Cannot provide both `openai_client` and `token`'
            self._client = openai_client
            raw = str(openai_client.base_url)
            self._account = raw.split('.snowflakecomputing.com')[0].lstrip('https://')
            return

        account = account or os.getenv('SNOWFLAKE_ACCOUNT')
        token = token or os.getenv('SNOWFLAKE_TOKEN')

        if not account:
            raise UserError(
                'Set the `SNOWFLAKE_ACCOUNT` environment variable or pass it via '
                '`SnowflakeCortexProvider(account=...)` to use the Snowflake Cortex provider.'
            )
        if not token:
            raise UserError(
                'Set the `SNOWFLAKE_TOKEN` environment variable or pass it via '
                '`SnowflakeCortexProvider(token=...)` to use the Snowflake Cortex provider. '
                'Generate a Programmatic Access Token (PAT) in Snowflake UI: '
                'Account > Security > Programmatic Access Tokens.'
            )

        self._account = account
        base_url = f'https://{account}.snowflakecomputing.com/api/v2/cortex/v1'

        if http_client is not None:
            self._client = AsyncOpenAI(base_url=base_url, api_key=token, http_client=http_client)
        else:
            http_client = create_async_http_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_http_client
            self._client = AsyncOpenAI(base_url=base_url, api_key=token, http_client=http_client)

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
