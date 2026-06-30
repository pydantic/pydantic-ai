"""Tests for SnowflakeCortexProvider."""

import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import anthropic
    import openai

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.snowflake import SnowflakeCortexAnthropicProvider, SnowflakeCortexProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai or anthropic not installed')


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
        SnowflakeCortexProvider(openai_client=openai_client, account='other')  # type: ignore[call-overload]


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


# ---------------------------------------------------------------------------
# SnowflakeCortexAnthropicProvider
# ---------------------------------------------------------------------------


def test_anthropic_provider_from_explicit_args() -> None:
    provider = SnowflakeCortexAnthropicProvider(account='myorg-myacct', token='pat-secret')
    assert provider.name == 'snowflake-cortex-anthropic'
    assert provider.base_url == 'https://myorg-myacct.snowflakecomputing.com/api/v2/cortex/anthropic/v1'
    assert isinstance(provider.client, anthropic.AsyncAnthropic)


def test_anthropic_provider_uses_auth_token_not_api_key() -> None:
    """Cortex /messages requires Authorization: Bearer — auth_token, not api_key."""
    provider = SnowflakeCortexAnthropicProvider(account='myorg-myacct', token='my-pat')
    # auth_token sets Authorization: Bearer, api_key sets X-Api-Key
    assert provider.client.auth_token == 'my-pat'
    assert provider.client.api_key is None


def test_anthropic_provider_sends_pat_token_type_header() -> None:
    """Both providers must send X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN."""
    ap = SnowflakeCortexAnthropicProvider(account='myorg-myacct', token='my-pat')
    assert ap.client.default_headers.get('X-Snowflake-Authorization-Token-Type') == 'PROGRAMMATIC_ACCESS_TOKEN'


def test_openai_provider_sends_pat_token_type_header() -> None:
    op = SnowflakeCortexProvider(account='myorg-myacct', token='my-pat')
    assert op.client.default_headers.get('X-Snowflake-Authorization-Token-Type') == 'PROGRAMMATIC_ACCESS_TOKEN'


def test_anthropic_provider_uses_anthropic_endpoint_path() -> None:
    """Anthropic path is /cortex/anthropic/v1, not /cortex/v1."""
    provider = SnowflakeCortexAnthropicProvider(account='myorg-myacct', token='tok')
    assert '/cortex/anthropic/v1' in provider.base_url


def test_anthropic_provider_from_env(env: TestEnv) -> None:
    env.set('SNOWFLAKE_ACCOUNT', 'myorg-envacct')
    env.set('SNOWFLAKE_TOKEN', 'env-pat')
    provider = SnowflakeCortexAnthropicProvider()
    assert 'myorg-envacct.snowflakecomputing.com' in provider.base_url


def test_anthropic_provider_requires_account(env: TestEnv) -> None:
    env.remove('SNOWFLAKE_ACCOUNT')
    env.remove('SNOWFLAKE_TOKEN')
    with pytest.raises(UserError, match='SNOWFLAKE_ACCOUNT'):
        SnowflakeCortexAnthropicProvider()


def test_anthropic_provider_requires_token(env: TestEnv) -> None:
    env.set('SNOWFLAKE_ACCOUNT', 'myorg-myacct')
    env.remove('SNOWFLAKE_TOKEN')
    with pytest.raises(UserError, match='SNOWFLAKE_TOKEN'):
        SnowflakeCortexAnthropicProvider()


def test_anthropic_provider_with_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = SnowflakeCortexAnthropicProvider(account='myorg-myacct', token='pat', http_client=http_client)
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_anthropic_provider_with_anthropic_client() -> None:
    client = anthropic.AsyncAnthropic(
        auth_token='pat',
        base_url='https://myorg-myacct.snowflakecomputing.com/api/v2/cortex/anthropic/v1',
    )
    provider = SnowflakeCortexAnthropicProvider(anthropic_client=client)
    assert provider.client is client
    assert 'myorg-myacct' in provider.base_url


def test_anthropic_provider_model_profile_claude() -> None:
    profile = SnowflakeCortexAnthropicProvider.model_profile('claude-sonnet-4-6')
    assert profile is not None


def test_anthropic_model_uses_cortex_anthropic_provider() -> None:
    provider = SnowflakeCortexAnthropicProvider(account='myorg-myacct', token='pat')
    model = AnthropicModel('claude-sonnet-4-6', provider=provider)
    assert model.system == 'snowflake-cortex-anthropic'
    assert 'myorg-myacct.snowflakecomputing.com' in model.base_url


# ---------------------------------------------------------------------------
# SnowflakeCortexModel — unified auto-routing model
# ---------------------------------------------------------------------------


def test_cortex_model_routes_llama_to_openai_path(env: TestEnv) -> None:
    env.set('SNOWFLAKE_ACCOUNT', 'myorg-myacct')
    env.set('SNOWFLAKE_TOKEN', 'tok')
    from pydantic_ai.models.snowflake import SnowflakeCortexModel

    m = SnowflakeCortexModel('llama4-maverick')
    assert m.system == 'snowflake-cortex'
    assert m.model_name == 'llama4-maverick'
    assert 'snowflakecomputing.com' in m.wrapped.base_url  # type: ignore[union-attr]


def test_cortex_model_routes_claude_to_anthropic_path(env: TestEnv) -> None:
    env.set('SNOWFLAKE_ACCOUNT', 'myorg-myacct')
    env.set('SNOWFLAKE_TOKEN', 'tok')
    from pydantic_ai.models.snowflake import SnowflakeCortexModel

    m = SnowflakeCortexModel('claude-sonnet-4-6')
    assert m.system == 'snowflake-cortex-anthropic'
    assert m.model_name == 'claude-sonnet-4-6'


def test_cortex_model_string_shorthand_llama(env: TestEnv) -> None:
    env.set('SNOWFLAKE_ACCOUNT', 'myorg-myacct')
    env.set('SNOWFLAKE_TOKEN', 'tok')
    from pydantic_ai.models import infer_model

    m = infer_model('snowflake-cortex:llama4-maverick')
    assert m.system == 'snowflake-cortex'
    assert m.model_name == 'llama4-maverick'


def test_cortex_model_string_shorthand_claude(env: TestEnv) -> None:
    env.set('SNOWFLAKE_ACCOUNT', 'myorg-myacct')
    env.set('SNOWFLAKE_TOKEN', 'tok')
    from pydantic_ai.models import infer_model

    m = infer_model('snowflake-cortex:claude-sonnet-4-6')
    assert m.system == 'snowflake-cortex-anthropic'
    assert m.model_name == 'claude-sonnet-4-6'


def test_cortex_model_explicit_account_token() -> None:
    from pydantic_ai.models.snowflake import SnowflakeCortexModel

    m = SnowflakeCortexModel('llama4-maverick', account='myorg-myacct', token='my-tok')
    assert m.system == 'snowflake-cortex'


def test_cortex_model_mistral_routes_to_openai_path(env: TestEnv) -> None:
    env.set('SNOWFLAKE_ACCOUNT', 'myorg-myacct')
    env.set('SNOWFLAKE_TOKEN', 'tok')
    from pydantic_ai.models.snowflake import SnowflakeCortexModel

    m = SnowflakeCortexModel('mistral-large2')
    assert m.system == 'snowflake-cortex'
