"""Tests for SnowflakeCortexProvider."""
import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.snowflake import SnowflakeCortexProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


def test_provider_from_explicit_args() -> None:
    provider = SnowflakeCortexProvider(account='myorg-myacct', token='pat-secret')
    assert provider.name == 'snowflake-cortex'
    assert provider.base_url == 'https://myorg-myacct.snowflakecomputing.com/api/v2/cortex/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'pat-secret'


def test_provider_from_env(env: TestEnv) -> None:
    env.set('SNOWFLAKE_ACCOUNT', 'myorg-envacct')
    env.set('SNOWFLAKE_TOKEN', 'env-pat-token')
    provider = SnowflakeCortexProvider()
    assert provider.base_url == 'https://myorg-envacct.snowflakecomputing.com/api/v2/cortex/v1'
    assert provider.client.api_key == 'env-pat-token'


def test_provider_requires_account(env: TestEnv) -> None:
    env.remove('SNOWFLAKE_ACCOUNT')
    env.remove('SNOWFLAKE_TOKEN')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `SNOWFLAKE_ACCOUNT` environment variable or pass it via '
            '`SnowflakeCortexProvider(account=...)` to use the Snowflake Cortex provider.'
        ),
    ):
        SnowflakeCortexProvider()


def test_provider_requires_token(env: TestEnv) -> None:
    env.set('SNOWFLAKE_ACCOUNT', 'myorg-myacct')
    env.remove('SNOWFLAKE_TOKEN')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `SNOWFLAKE_TOKEN` environment variable or pass it via '
            '`SnowflakeCortexProvider(token=...)` to use the Snowflake Cortex provider.'
        ),
    ):
        SnowflakeCortexProvider()


def test_provider_with_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = SnowflakeCortexProvider(account='myorg-myacct', token='pat', http_client=http_client)
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_provider_with_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(
        api_key='pat',
        base_url='https://myorg-myacct.snowflakecomputing.com/api/v2/cortex/v1',
    )
    provider = SnowflakeCortexProvider(openai_client=openai_client)
    assert provider.client is openai_client
    assert 'myorg-myacct' in provider.base_url


def test_provider_openai_client_excludes_account() -> None:
    openai_client = openai.AsyncOpenAI(
        api_key='pat',
        base_url='https://myorg-myacct.snowflakecomputing.com/api/v2/cortex/v1',
    )
    with pytest.raises(AssertionError):
        SnowflakeCortexProvider(openai_client=openai_client, account='other')


# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------


def test_profile_llama() -> None:
    profile = SnowflakeCortexProvider.model_profile('llama4-maverick')
    assert profile is not None
    assert profile.get('json_schema_transformer') == OpenAIJsonSchemaTransformer
    assert profile.get('openai_supports_strict_tool_definition') is False
    assert profile.get('openai_chat_supports_max_completion_tokens') is True


def test_profile_snowflake_llama() -> None:
    profile = SnowflakeCortexProvider.model_profile('snowflake-llama-3.3-70b')
    assert profile is not None
    assert profile.get('openai_chat_supports_max_completion_tokens') is True


def test_profile_claude() -> None:
    profile = SnowflakeCortexProvider.model_profile('claude-sonnet-4-6')
    assert profile is not None
    assert profile.get('json_schema_transformer') == OpenAIJsonSchemaTransformer


def test_profile_mistral() -> None:
    profile = SnowflakeCortexProvider.model_profile('mistral-large2')
    assert profile is not None
    assert profile.get('openai_supports_strict_tool_definition') is False


def test_profile_openai_gpt() -> None:
    profile = SnowflakeCortexProvider.model_profile('openai-gpt-4.1')
    assert profile is not None


def test_profile_deepseek() -> None:
    # DeepSeek on Cortex has no special family profile; falls back to base Cortex profile.
    profile = SnowflakeCortexProvider.model_profile('deepseek-r1')
    assert profile is not None
    assert profile.get('openai_chat_supports_max_completion_tokens') is True


# ---------------------------------------------------------------------------
# Integration with OpenAIChatModel
# ---------------------------------------------------------------------------


def test_openai_chat_model_uses_cortex_provider() -> None:
    provider = SnowflakeCortexProvider(account='myorg-myacct', token='pat')
    model = OpenAIChatModel('llama4-maverick', provider=provider)
    assert model.system == 'snowflake-cortex'
    assert 'myorg-myacct.snowflakecomputing.com' in model.base_url


def test_provider_repr() -> None:
    provider = SnowflakeCortexProvider(account='myorg-myacct', token='pat')
    assert 'snowflake-cortex' in repr(provider)
    assert 'snowflakecomputing.com' in repr(provider)
