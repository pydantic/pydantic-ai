import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.profiles.amazon import amazon_model_profile
    from pydantic_ai.profiles.anthropic import anthropic_model_profile
    from pydantic_ai.profiles.cohere import cohere_model_profile
    from pydantic_ai.profiles.deepseek import deepseek_model_profile
    from pydantic_ai.profiles.google import google_model_profile
    from pydantic_ai.profiles.meta import meta_model_profile
    from pydantic_ai.profiles.mistral import mistral_model_profile
    from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
    from pydantic_ai.providers.edenai import EdenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='OpenAI client not installed'),
    pytest.mark.anyio,
]


def test_init_with_api_key():
    provider = EdenAIProvider(api_key='test-key')
    assert provider.name == 'edenai'
    assert provider.base_url == 'https://api.edenai.run/v3'
    assert provider.client.api_key == 'test-key'


def test_init_from_env(env: TestEnv):
    env.set('EDENAI_API_KEY', 'env-key')
    provider = EdenAIProvider()
    assert provider.client.api_key == 'env-key'


def test_init_without_api_key_raises(env: TestEnv):
    env.remove('EDENAI_API_KEY')
    with pytest.raises(UserError, match=re.escape('Set the `EDENAI_API_KEY` environment variable')):
        EdenAIProvider()


def test_init_with_openai_client():
    openai_client = AsyncOpenAI(api_key='custom-key', base_url='https://custom.example.com/v1')
    provider = EdenAIProvider(openai_client=openai_client)
    assert provider.client is openai_client


async def test_init_with_http_client():
    async with httpx.AsyncClient() as custom_client:
        provider = EdenAIProvider(api_key='test-key', http_client=custom_client)
        assert isinstance(provider.client, AsyncOpenAI)
        assert provider.client.api_key == 'test-key'


async def test_create_http_client_default(mocker: MockerFixture):
    async with httpx.AsyncClient() as mock_client:
        mock_create_func = mocker.patch(
            'pydantic_ai.providers.edenai.create_async_http_client', return_value=mock_client
        )
        provider = EdenAIProvider(api_key='test-key')
        mock_create_func.assert_called_once_with()
        assert isinstance(provider.client, AsyncOpenAI)


def test_model_profile_unknown_prefix_falls_back_to_openai(mocker: MockerFixture):
    provider = EdenAIProvider(api_key='test-key')
    mock_openai_profile = mocker.patch('pydantic_ai.providers.edenai.openai_model_profile', wraps=openai_model_profile)

    profile = provider.model_profile('unknown-vendor/some-model')

    mock_openai_profile.assert_called_once_with('unknown-vendor/some-model')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer


def test_model_profile_no_slash_falls_back_to_openai(mocker: MockerFixture):
    provider = EdenAIProvider(api_key='test-key')
    mock_openai_profile = mocker.patch('pydantic_ai.providers.edenai.openai_model_profile', wraps=openai_model_profile)

    profile = provider.model_profile('gpt-4o-mini')

    mock_openai_profile.assert_called_once_with('gpt-4o-mini')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer


def test_model_profile_routes_to_vendor_profile(mocker: MockerFixture):
    # This test verifies routing of vendor prefixes to profile functions, not catalog
    # availability — placeholder suffixes are deliberate so the test does not go stale
    # when a specific Eden AI model id is deprecated.
    provider = EdenAIProvider(api_key='test-key')

    mocks = {
        'anthropic': mocker.patch(
            'pydantic_ai.providers.edenai.anthropic_model_profile', wraps=anthropic_model_profile
        ),
        'openai': mocker.patch('pydantic_ai.providers.edenai.openai_model_profile', wraps=openai_model_profile),
        'google': mocker.patch('pydantic_ai.providers.edenai.google_model_profile', wraps=google_model_profile),
        'mistral': mocker.patch('pydantic_ai.providers.edenai.mistral_model_profile', wraps=mistral_model_profile),
        'cohere': mocker.patch('pydantic_ai.providers.edenai.cohere_model_profile', wraps=cohere_model_profile),
        'meta': mocker.patch('pydantic_ai.providers.edenai.meta_model_profile', wraps=meta_model_profile),
        'amazon': mocker.patch('pydantic_ai.providers.edenai.amazon_model_profile', wraps=amazon_model_profile),
        'deepseek': mocker.patch('pydantic_ai.providers.edenai.deepseek_model_profile', wraps=deepseek_model_profile),
    }

    cases = [
        ('anthropic/test-model', 'anthropic', 'test-model'),
        ('openai/test-model', 'openai', 'test-model'),
        ('google/test-model', 'google', 'test-model'),
        ('mistral/test-model', 'mistral', 'test-model'),
        ('mistralai/test-model', 'mistral', 'test-model'),
        ('cohere/test-model', 'cohere', 'test-model'),
        ('meta/test-model', 'meta', 'test-model'),
        ('meta-llama/test-model', 'meta', 'test-model'),
        ('amazon/test-model', 'amazon', 'test-model'),
        ('bedrock/test-model', 'amazon', 'test-model'),
        ('deepseek/test-model', 'deepseek', 'test-model'),
    ]

    for model_id, expected_vendor, expected_suffix in cases:
        for mock in mocks.values():
            mock.reset_mock()
        profile = provider.model_profile(model_id)
        # Always wrapped in OpenAIModelProfile (the OpenAI-chat-compat shape).
        # We deliberately do not assert on `json_schema_transformer`: when the vendor
        # profile sets one (e.g. GoogleJsonSchemaTransformer), .update() preserves it,
        # matching how LiteLLMProvider, NebiusProvider, and OpenRouterProvider behave.
        assert isinstance(profile, OpenAIModelProfile)
        mocks[expected_vendor].assert_called_once_with(expected_suffix)
