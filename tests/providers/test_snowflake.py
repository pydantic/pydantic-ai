from __future__ import annotations as _annotations

import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import InlineDefsJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.snowflake import SnowflakeProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_snowflake_provider():
    provider = SnowflakeProvider(account='myorg-myaccount', token='pat')
    assert provider.name == 'snowflake'
    assert provider.base_url == 'https://myorg-myaccount.snowflakecomputing.com/api/v2/cortex/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'pat'


def test_snowflake_provider_from_env(env: TestEnv) -> None:
    env.set('SNOWFLAKE_ACCOUNT', 'myorg-envaccount')
    env.set('SNOWFLAKE_TOKEN', 'env-pat')
    provider = SnowflakeProvider()
    assert provider.base_url == 'https://myorg-envaccount.snowflakecomputing.com/api/v2/cortex/v1'
    assert provider.client.api_key == 'env-pat'


def test_snowflake_provider_need_account(env: TestEnv) -> None:
    env.remove('SNOWFLAKE_ACCOUNT')
    env.remove('SNOWFLAKE_TOKEN')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `SNOWFLAKE_ACCOUNT` environment variable or pass it via `SnowflakeProvider(account=...)`'
            ' to use the Snowflake provider.'
        ),
    ):
        SnowflakeProvider()


def test_snowflake_provider_need_token(env: TestEnv) -> None:
    env.remove('SNOWFLAKE_TOKEN')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `SNOWFLAKE_TOKEN` environment variable or pass it via `SnowflakeProvider(token=...)`'
            ' to use the Snowflake provider.'
        ),
    ):
        SnowflakeProvider(account='myorg-myaccount')


@pytest.mark.parametrize(
    'account',
    [
        'myorg-myaccount',
        'myorg-myaccount.snowflakecomputing.com',
        'https://myorg-myaccount.snowflakecomputing.com',
        'https://myorg-myaccount.snowflakecomputing.com/',
    ],
)
def test_snowflake_provider_account_normalization(account: str) -> None:
    """Account values that include more than the bare identifier are normalized."""
    provider = SnowflakeProvider(account=account, token='pat')
    assert provider.base_url == 'https://myorg-myaccount.snowflakecomputing.com/api/v2/cortex/v1'


def test_snowflake_provider_base_url_override(env: TestEnv) -> None:
    """A custom `base_url` (e.g. private connectivity) does not require an account."""
    env.remove('SNOWFLAKE_ACCOUNT')
    provider = SnowflakeProvider(
        base_url='https://myorg-myaccount.privatelink.snowflakecomputing.com/api/v2/cortex/v1', token='pat'
    )
    assert provider.base_url == 'https://myorg-myaccount.privatelink.snowflakecomputing.com/api/v2/cortex/v1'


def test_snowflake_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = SnowflakeProvider(account='myorg-myaccount', token='pat', http_client=http_client)
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_snowflake_provider_reenter_recreates_http_client() -> None:
    """Re-entering the provider after its own HTTP client is closed swaps in a fresh one."""
    provider = SnowflakeProvider(account='myorg-myaccount', token='pat')

    first_http_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
    async with provider:
        assert not first_http_client.is_closed
    assert first_http_client.is_closed

    async with provider:
        second_http_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert second_http_client is not first_http_client
        assert not second_http_client.is_closed
    assert second_http_client.is_closed


def test_snowflake_provider_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(
        api_key='pat',
        base_url='https://myorg-myaccount.snowflakecomputing.com/api/v2/cortex/v1',
    )
    provider = SnowflakeProvider(openai_client=openai_client)
    assert provider.client is openai_client
    assert provider.base_url == 'https://myorg-myaccount.snowflakecomputing.com/api/v2/cortex/v1/'


def test_snowflake_provider_openai_client_excludes_other_args() -> None:
    openai_client = openai.AsyncOpenAI(
        api_key='pat',
        base_url='https://myorg-myaccount.snowflakecomputing.com/api/v2/cortex/v1',
    )
    with pytest.raises(AssertionError, match='Cannot provide both `openai_client` and `account`'):
        SnowflakeProvider(openai_client=openai_client, account='other')  # type: ignore[call-overload]


def test_snowflake_provider_model_profile_claude():
    profile = SnowflakeProvider.model_profile('claude-sonnet-4-6')
    assert profile is not None
    assert profile.get('json_schema_transformer') == OpenAIJsonSchemaTransformer
    assert profile.get('supports_tools', True) is True
    assert profile.get('supports_thinking') is True
    assert profile.get('supports_json_schema_output') is True
    assert profile.get('supports_json_object_output') is False
    assert profile.get('openai_supports_strict_tool_definition') is False


def test_snowflake_provider_model_profile_openai():
    profile = SnowflakeProvider.model_profile('openai-gpt-5.2')
    assert profile is not None
    assert profile.get('json_schema_transformer') == OpenAIJsonSchemaTransformer
    assert profile.get('supports_tools', True) is True
    assert profile.get('supports_json_schema_output') is True
    assert profile.get('openai_supports_strict_tool_definition') is False


def test_snowflake_provider_model_profile_no_tool_support():
    """Cortex only accepts `tools` and `response_format` for OpenAI and Claude models."""
    for model_name in ('llama4-maverick', 'snowflake-llama-3.3-70b', 'mistral-large2', 'deepseek-r1'):
        profile = SnowflakeProvider.model_profile(model_name)
        assert profile is not None, model_name
        assert profile.get('supports_tools') is False, model_name
        assert profile.get('supports_json_schema_output') is False, model_name
        assert profile.get('supports_json_object_output') is False, model_name
        assert profile.get('default_structured_output_mode') == 'prompted', model_name


def test_snowflake_provider_model_profile_families():
    llama_profile = SnowflakeProvider.model_profile('llama4-maverick')
    assert llama_profile is not None
    assert llama_profile.get('json_schema_transformer') == InlineDefsJsonSchemaTransformer

    deepseek_profile = SnowflakeProvider.model_profile('deepseek-r1')
    assert deepseek_profile is not None
    assert deepseek_profile.get('thinking_always_enabled') is True

    unknown_profile = SnowflakeProvider.model_profile('some-future-model')
    assert unknown_profile is not None
    assert unknown_profile.get('json_schema_transformer') == OpenAIJsonSchemaTransformer
    assert unknown_profile.get('supports_tools') is False
