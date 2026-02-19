import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.vercel import VercelProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_vercel_provider():
    provider = VercelProvider(api_key='api-key')
    assert provider.name == 'vercel'
    assert provider.base_url == 'https://ai-gateway.vercel.sh/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_vercel_provider_need_api_key(env: TestEnv) -> None:
    env.remove('VERCEL_AI_GATEWAY_API_KEY')
    env.remove('VERCEL_OIDC_TOKEN')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `VERCEL_AI_GATEWAY_API_KEY` or `VERCEL_OIDC_TOKEN` environment variable '
            'or pass the API key via `VercelProvider(api_key=...)` to use the Vercel provider.'
        ),
    ):
        VercelProvider()


def test_vercel_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = VercelProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_vercel_provider_model_profile(mocker: MockerFixture):
    provider = VercelProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.vercel'

    # Mock all profile functions
    anthropic_mock = mocker.patch(f'{ns}.anthropic_model_profile', wraps=anthropic_model_profile)
    amazon_mock = mocker.patch(f'{ns}.amazon_model_profile', wraps=amazon_model_profile)
    cohere_mock = mocker.patch(f'{ns}.cohere_model_profile', wraps=cohere_model_profile)
    deepseek_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    google_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    grok_mock = mocker.patch(f'{ns}.grok_model_profile', wraps=grok_model_profile)
    mistral_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    openai_mock = mocker.patch(f'{ns}.openai_model_profile', wraps=openai_model_profile)

    # Test openai provider
    profile = provider.model_profile('openai/gpt-4o')
    openai_mock.assert_called_with('gpt-4o')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test anthropic provider
    profile = provider.model_profile('anthropic/claude-sonnet-4-5')
    anthropic_mock.assert_called_with('claude-sonnet-4-5')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test bedrock provider
    profile = provider.model_profile('bedrock/anthropic.claude-sonnet-4-5')
    amazon_mock.assert_called_with('anthropic.claude-sonnet-4-5')
    assert profile is not None
    assert profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

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

    # Test vertex provider
    profile = provider.model_profile('vertex/gemini-1.5-pro')
    google_mock.assert_called_with('gemini-1.5-pro')
    assert profile is not None
    assert profile.json_schema_transformer == GoogleJsonSchemaTransformer

    # Test xai provider
    profile = provider.model_profile('xai/grok-beta')
    grok_mock.assert_called_with('grok-beta')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer


def test_vercel_with_http_client():
    http_client = httpx.AsyncClient()
    provider = VercelProvider(api_key='test-key', http_client=http_client)
    assert provider.client.api_key == 'test-key'
    assert str(provider.client.base_url) == 'https://ai-gateway.vercel.sh/v1/'


def test_vercel_provider_invalid_model_name():
    provider = VercelProvider(api_key='api-key')

    with pytest.raises(UserError, match="Model name must be in 'provider/model' format"):
        provider.model_profile('invalid-model-name')


def test_vercel_provider_unknown_provider():
    provider = VercelProvider(api_key='api-key')

    profile = provider.model_profile('unknown/gpt-4')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer
