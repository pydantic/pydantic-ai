import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai import Agent
from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile
from pydantic_ai.profiles.perplexity import perplexity_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.cloudflare import CloudflareProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_cloudflare_provider():
    provider = CloudflareProvider(account_id='test-account-id', gateway_id='test-gateway-id', api_key='api-key')
    assert provider.name == 'cloudflare'
    assert provider.base_url == 'https://gateway.ai.cloudflare.com/v1/test-account-id/test-gateway-id/compat'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_cloudflare_provider_need_account_id(env: TestEnv) -> None:
    env.remove('CLOUDFLARE_ACCOUNT_ID')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `CLOUDFLARE_ACCOUNT_ID` environment variable '
            'or pass it via `CloudflareProvider(account_id=...)` to use the Cloudflare provider.'
        ),
    ):
        CloudflareProvider(gateway_id='test-gateway-id', api_key='api-key')  # type: ignore[call-overload]


def test_cloudflare_provider_need_gateway_id(env: TestEnv) -> None:
    env.remove('CLOUDFLARE_GATEWAY_ID')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `CLOUDFLARE_GATEWAY_ID` environment variable '
            'or pass it via `CloudflareProvider(gateway_id=...)` to use the Cloudflare provider.'
        ),
    ):
        CloudflareProvider(account_id='test-account-id', api_key='api-key')  # type: ignore[call-overload]


def test_cloudflare_provider_from_env(env: TestEnv) -> None:
    env.set('CLOUDFLARE_ACCOUNT_ID', 'env-account-id')
    env.set('CLOUDFLARE_GATEWAY_ID', 'env-gateway-id')

    # Test with explicit api_key (account_id and gateway_id from env)
    provider = CloudflareProvider(api_key='env-api-key')  # type: ignore[call-overload]
    assert provider.base_url == 'https://gateway.ai.cloudflare.com/v1/env-account-id/env-gateway-id/compat'
    assert provider.client.api_key == 'env-api-key'


def test_cloudflare_provider_with_gateway_auth_token():
    provider = CloudflareProvider(
        account_id='test-account-id',
        gateway_id='test-gateway-id',
        api_key='api-key',
        gateway_auth_token='gateway-token',
    )
    assert provider.client.default_headers['cf-aig-authorization'] == 'gateway-token'


def test_cloudflare_provider_gateway_auth_token_from_env(env: TestEnv) -> None:
    env.set('CLOUDFLARE_ACCOUNT_ID', 'test-account-id')
    env.set('CLOUDFLARE_GATEWAY_ID', 'test-gateway-id')
    env.set('CLOUDFLARE_AI_GATEWAY_AUTH', 'env-gateway-token')

    provider = CloudflareProvider(api_key='api-key')  # type: ignore[call-overload]
    assert provider.client.default_headers['cf-aig-authorization'] == 'env-gateway-token'


def test_cloudflare_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = CloudflareProvider(
        account_id='test-account-id', gateway_id='test-gateway-id', openai_client=openai_client
    )
    assert provider.client == openai_client


def test_cloudflare_provider_model_profile(mocker: MockerFixture, env: TestEnv):
    # Set dummy API keys so we can use real GroqProvider and CerebrasProvider
    env.set('GROQ_API_KEY', 'test-groq-key')
    env.set('CEREBRAS_API_KEY', 'test-cerebras-key')

    provider = CloudflareProvider(account_id='test-account-id', gateway_id='test-gateway-id', api_key='api-key')

    ns = 'pydantic_ai.providers.cloudflare'

    # Mock all profile functions
    anthropic_mock = mocker.patch(f'{ns}.anthropic_model_profile', wraps=anthropic_model_profile)
    cohere_mock = mocker.patch(f'{ns}.cohere_model_profile', wraps=cohere_model_profile)
    deepseek_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    google_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    grok_mock = mocker.patch(f'{ns}.grok_model_profile', wraps=grok_model_profile)
    mistral_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    openai_mock = mocker.patch(f'{ns}.openai_model_profile', wraps=openai_model_profile)
    perplexity_mock = mocker.patch(f'{ns}.perplexity_model_profile', wraps=perplexity_model_profile)

    # Use real GroqProvider and CerebrasProvider - they don't make API calls for model_profile()
    # We just need dummy API keys which are set via env vars above

    # Test openai provider
    profile = provider.model_profile('openai/gpt-4o')
    openai_mock.assert_called_with('gpt-4o')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test anthropic provider
    profile = provider.model_profile('anthropic/claude-3-sonnet')
    anthropic_mock.assert_called_with('claude-3-sonnet')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test cohere provider
    profile = provider.model_profile('cohere/command-r-plus')
    cohere_mock.assert_called_with('command-r-plus')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test deepseek provider
    profile = provider.model_profile('deepseek/deepseek-chat')
    deepseek_mock.assert_called_with('deepseek-chat')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test mistral provider
    profile = provider.model_profile('mistral/mistral-large')
    mistral_mock.assert_called_with('mistral-large')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test google-ai-studio provider
    profile = provider.model_profile('google-ai-studio/gemini-1.5-pro')
    google_mock.assert_called_with('gemini-1.5-pro')
    assert profile is not None
    assert profile.json_schema_transformer == GoogleJsonSchemaTransformer

    # Test grok provider
    profile = provider.model_profile('grok/grok-beta')
    grok_mock.assert_called_with('grok-beta')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test xai provider (alias for grok)
    profile = provider.model_profile('xai/grok-2')
    grok_mock.assert_called_with('grok-2')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test groq provider with llama model (delegates to GroqProvider which returns meta profile)
    # meta_model_profile uses InlineDefsJsonSchemaTransformer
    profile = provider.model_profile('groq/llama-3.3-70b-versatile')
    assert profile is not None
    assert profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    # Test groq provider with gemma model (delegates to GroqProvider which returns google profile)
    # google_model_profile uses GoogleJsonSchemaTransformer
    profile = provider.model_profile('groq/gemma-7b-it')
    assert profile is not None
    assert profile.json_schema_transformer == GoogleJsonSchemaTransformer

    # Test perplexity provider (currently returns None, falls back to OpenAI-compatible)
    profile = provider.model_profile('perplexity/llama-3.1-sonar-small-128k-online')
    perplexity_mock.assert_called_with('llama-3.1-sonar-small-128k-online')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test workers-ai provider (Cloudflare's own AI service)
    profile = provider.model_profile('workers-ai/@cf/meta/llama-3.1-8b-instruct')
    openai_mock.assert_called_with('@cf/meta/llama-3.1-8b-instruct')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test cerebras provider with llama model (delegates to CerebrasProvider which returns meta profile)
    # meta_model_profile uses InlineDefsJsonSchemaTransformer, wrapped by CerebrasProvider's OpenAIModelProfile
    profile = provider.model_profile('cerebras/llama3.1-8b')
    assert profile is not None
    assert profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    # Test cerebras provider with qwen model (delegates to CerebrasProvider which returns qwen profile)
    # qwen_model_profile uses InlineDefsJsonSchemaTransformer, wrapped by CerebrasProvider's OpenAIModelProfile
    profile = provider.model_profile('cerebras/qwen3.5-8b')
    assert profile is not None
    assert profile.json_schema_transformer == InlineDefsJsonSchemaTransformer


def test_cloudflare_with_http_client():
    http_client = httpx.AsyncClient()
    provider = CloudflareProvider(
        account_id='test-account-id', gateway_id='test-gateway-id', api_key='test-key', http_client=http_client
    )
    assert provider.client.api_key == 'test-key'
    assert (
        str(provider.client.base_url) == 'https://gateway.ai.cloudflare.com/v1/test-account-id/test-gateway-id/compat/'
    )


def test_cloudflare_provider_invalid_model_name():
    provider = CloudflareProvider(account_id='test-account-id', gateway_id='test-gateway-id', api_key='api-key')

    with pytest.raises(UserError, match="Model name must be in 'provider/model' format"):
        provider.model_profile('invalid-model-name')


def test_cloudflare_provider_unknown_provider():
    provider = CloudflareProvider(account_id='test-account-id', gateway_id='test-gateway-id', api_key='api-key')

    profile = provider.model_profile('unknown/gpt-4')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer


def test_cloudflare_default_headers():
    provider = CloudflareProvider(account_id='test-account-id', gateway_id='test-gateway-id', api_key='api-key')

    # Check that default headers are set
    assert provider.client.default_headers['http-referer'] == 'https://ai.pydantic.dev/'
    assert provider.client.default_headers['x-title'] == 'pydantic-ai'


def test_cloudflare_provider_stored_keys():
    """Test CF-managed keys mode - API keys stored in Cloudflare dashboard (requires authenticated gateway)."""
    provider = CloudflareProvider(
        account_id='test-account-id',
        gateway_id='test-gateway-id',
        gateway_auth_token='gateway-token',
    )
    # api_key is set to empty string for AsyncOpenAI to prevent Authorization header
    assert provider.client.api_key == ''
    assert provider.base_url == 'https://gateway.ai.cloudflare.com/v1/test-account-id/test-gateway-id/compat'
    assert provider.client.default_headers['cf-aig-authorization'] == 'gateway-token'


def test_cloudflare_provider_missing_credentials():
    """Test that error is raised when api_key is missing and not in CF-managed keys mode."""
    with pytest.raises(
        UserError,
        match=re.escape('You must provide an api_key for user-managed mode.'),
    ):
        CloudflareProvider(account_id='test-account-id', gateway_id='test-gateway-id')  # type: ignore[call-overload]


def test_cloudflare_stored_keys_no_auth_header():
    """Test that Authorization header is not sent in CF-managed keys mode (empty api_key)."""
    provider = CloudflareProvider(
        account_id='test-account-id',
        gateway_id='test-gateway-id',
        gateway_auth_token='gateway-token',
    )

    # In CF-managed keys mode, api_key is empty string which prevents OpenAI SDK from adding Authorization header
    assert provider.client.api_key == ''
    assert provider.client.default_headers['cf-aig-authorization'] == 'gateway-token'


def test_cloudflare_documented_patterns():
    """Test the exact usage patterns from the documentation work correctly.

    This test validates the examples shown in docs/models/openai.md work as documented.
    """
    from pydantic_ai.models.openai import OpenAIChatModel

    # Example 1: Basic BYOK mode (from docs)
    model = OpenAIChatModel(
        'openai/gpt-4o',
        provider=CloudflareProvider(
            account_id='your-account-id',
            gateway_id='your-gateway-id',
            api_key='your-openai-api-key',
        ),
    )
    agent = Agent(model)
    assert isinstance(agent.model, OpenAIChatModel)
    assert agent.model.model_name == 'openai/gpt-4o'

    # Example 2: Stored keys mode (from docs)
    model = OpenAIChatModel(
        'anthropic/claude-3-5-sonnet',
        provider=CloudflareProvider(
            account_id='your-account-id',
            gateway_id='your-gateway-id',
            gateway_auth_token='your-gateway-token',
        ),
    )
    agent = Agent(model)
    assert isinstance(agent.model, OpenAIChatModel)
    assert agent.model.model_name == 'anthropic/claude-3-5-sonnet'
