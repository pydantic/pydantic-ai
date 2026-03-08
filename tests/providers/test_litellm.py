import httpx
import pytest
from pytest_mock import MockerFixture

from ..conftest import try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
    from pydantic_ai.providers.litellm import LiteLLMProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='OpenAI client not installed'),
    pytest.mark.anyio,
]


def test_init_with_api_config():
    provider = LiteLLMProvider(api_key='test-key', api_base='https://custom.litellm.com/v1')
    assert provider.base_url == 'https://custom.litellm.com/v1/'
    assert provider.client.api_key == 'test-key'


def test_init_without_api_key():
    provider = LiteLLMProvider()
    assert provider.name == 'litellm'
    assert provider.client.api_key == 'litellm-placeholder'


def test_init_with_openai_client():
    openai_client = AsyncOpenAI(api_key='custom-key', base_url='https://custom.openai.com/v1')
    provider = LiteLLMProvider(openai_client=openai_client)
    assert provider.client == openai_client
    assert provider.base_url == 'https://custom.openai.com/v1/'


def test_model_profile_returns_openai_compatible_profile(mocker: MockerFixture):
    provider = LiteLLMProvider(api_key='test-key')

    # Create a proper mock profile object that can be updated
    from dataclasses import dataclass

    @dataclass
    class MockProfile:
        max_tokens: int = 4096
        supports_streaming: bool = True

    mock_profile = MockProfile()
    mock_openai_profile = mocker.patch('pydantic_ai.providers.litellm.openai_model_profile', return_value=mock_profile)

    profile = provider.model_profile('gpt-3.5-turbo')

    # Verify openai_model_profile was called with the model name
    mock_openai_profile.assert_called_once_with('gpt-3.5-turbo')

    # Verify the returned profile is an OpenAIModelProfile with OpenAIJsonSchemaTransformer
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer


def test_model_profile_with_different_models(mocker: MockerFixture):
    provider = LiteLLMProvider(api_key='test-key')

    # Create mocks for all profile functions
    from dataclasses import dataclass

    @dataclass
    class MockProfile:
        max_tokens: int = 4096
        supports_streaming: bool = True

    # Mock all profile functions
    mock_profiles = {
        'openai': mocker.patch('pydantic_ai.providers.litellm.openai_model_profile', return_value=MockProfile()),
        'anthropic': mocker.patch('pydantic_ai.providers.litellm.anthropic_model_profile', return_value=MockProfile()),
        'google': mocker.patch('pydantic_ai.providers.litellm.google_model_profile', return_value=MockProfile()),
        'meta': mocker.patch('pydantic_ai.providers.litellm.meta_model_profile', return_value=MockProfile()),
        'mistral': mocker.patch('pydantic_ai.providers.litellm.mistral_model_profile', return_value=MockProfile()),
        'cohere': mocker.patch('pydantic_ai.providers.litellm.cohere_model_profile', return_value=MockProfile()),
        'amazon': mocker.patch('pydantic_ai.providers.litellm.amazon_model_profile', return_value=MockProfile()),
        'deepseek': mocker.patch('pydantic_ai.providers.litellm.deepseek_model_profile', return_value=MockProfile()),
        'groq': mocker.patch('pydantic_ai.providers.litellm.groq_model_profile', return_value=MockProfile()),
        'grok': mocker.patch('pydantic_ai.providers.litellm.grok_model_profile', return_value=MockProfile()),
        'moonshotai': mocker.patch(
            'pydantic_ai.providers.litellm.moonshotai_model_profile', return_value=MockProfile()
        ),
        'qwen': mocker.patch('pydantic_ai.providers.litellm.qwen_model_profile', return_value=MockProfile()),
    }

    # Test models without provider prefix (should use openai profile)
    models_without_prefix = ['gpt-4', 'claude-sonnet-4-5', 'gemini-pro', 'llama2-70b']

    for model in models_without_prefix:
        profile = provider.model_profile(model)
        assert isinstance(profile, OpenAIModelProfile)
        assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Verify openai_model_profile was called for each model without prefix
    assert mock_profiles['openai'].call_count == len(models_without_prefix)

    # Reset all call counts
    for mock in mock_profiles.values():
        mock.reset_mock()

    # Test all provider prefixes
    test_cases = [
        ('anthropic/claude-3-haiku', 'anthropic', 'claude-3-haiku'),
        ('openai/gpt-4-turbo', 'openai', 'gpt-4-turbo'),
        ('google/gemini-1.5-pro', 'google', 'gemini-1.5-pro'),
        ('mistralai/mistral-large', 'mistral', 'mistral-large'),
        ('mistral/mistral-7b', 'mistral', 'mistral-7b'),
        ('cohere/command-r', 'cohere', 'command-r'),
        ('amazon/titan-text', 'amazon', 'titan-text'),
        ('bedrock/claude-v2', 'amazon', 'claude-v2'),
        ('meta-llama/llama-3-8b', 'meta', 'llama-3-8b'),
        ('meta/llama-2-70b', 'meta', 'llama-2-70b'),
        ('groq/llama3-70b', 'groq', 'llama3-70b'),
        ('deepseek/deepseek-coder', 'deepseek', 'deepseek-coder'),
        ('moonshotai/moonshot-v1', 'moonshotai', 'moonshot-v1'),
        ('x-ai/grok-beta', 'grok', 'grok-beta'),
        ('qwen/qwen-72b', 'qwen', 'qwen-72b'),
    ]

    for model_name, expected_profile, expected_suffix in test_cases:
        profile = provider.model_profile(model_name)
        assert isinstance(profile, OpenAIModelProfile)
        assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer
        # Verify the correct profile function was called with the correct suffix
        mock_profiles[expected_profile].assert_called_with(expected_suffix)
        mock_profiles[expected_profile].reset_mock()

    # Test unknown provider prefix (should fall back to openai)
    provider.model_profile('unknown-provider/some-model')
    mock_profiles['openai'].assert_called_once_with('unknown-provider/some-model')


async def test_cached_http_client_usage(mocker: MockerFixture):
    # Create a real AsyncClient for the mock
    async with httpx.AsyncClient() as mock_cached_client:
        mock_cached_http_client_func = mocker.patch(
            'pydantic_ai.providers.litellm.cached_async_http_client', return_value=mock_cached_client
        )

        provider = LiteLLMProvider(api_key='test-key')

        # Verify cached_async_http_client was called with 'litellm' provider
        mock_cached_http_client_func.assert_called_once_with(provider='litellm')

        # Verify the client was created
        assert isinstance(provider.client, AsyncOpenAI)


async def test_init_with_http_client_overrides_cached():
    async with httpx.AsyncClient() as custom_client:
        provider = LiteLLMProvider(api_key='test-key', http_client=custom_client)

        # Verify the provider was created successfully with custom client
        assert isinstance(provider.client, AsyncOpenAI)
        assert provider.client.api_key == 'test-key'
