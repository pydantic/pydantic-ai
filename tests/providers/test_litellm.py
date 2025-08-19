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
    assert provider.base_url == 'https://custom.litellm.com/v1'
    assert provider.client.api_key == 'test-key'


def test_init_with_custom_llm_provider():
    provider = LiteLLMProvider(api_key='test-key', custom_llm_provider='anthropic')
    assert provider.name == 'litellm'


def test_init_without_api_key():
    provider = LiteLLMProvider()
    assert provider.name == 'litellm'
    assert provider.base_url == 'https://api.litellm.ai/v1'
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

    mock_openai_profile = mocker.patch('pydantic_ai.providers.litellm.openai_model_profile', return_value={})

    # Test with different model formats that LiteLLM supports
    test_models = [
        'gpt-4',
        'claude-3-sonnet-20240229',
        'gemini-pro',
        'llama2-70b-chat',
        'anthropic/claude-3-haiku-20240307',
    ]

    for model in test_models:
        profile = provider.model_profile(model)
        assert isinstance(profile, OpenAIModelProfile)
        assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Verify openai_model_profile was called for each model
    assert mock_openai_profile.call_count == len(test_models)


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
