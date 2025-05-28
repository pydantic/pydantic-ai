import os

import pytest
from inline_snapshot import snapshot
from pytest_mock import MockerFixture

from pydantic_ai.agent import Agent
from pydantic_ai.profiles import DEFAULT_PROFILE

from ..conftest import try_import

with try_import() as imports_successful:
    from openai import AsyncAzureOpenAI

    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.azure import AzureProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_azure_provider():
    provider = AzureProvider(
        azure_endpoint='https://project-id.openai.azure.com/',
        api_version='2023-03-15-preview',
        api_key='1234567890',
    )
    assert isinstance(provider, AzureProvider)
    assert provider.name == 'azure'
    assert provider.base_url == snapshot('https://project-id.openai.azure.com/openai/')
    assert isinstance(provider.client, AsyncAzureOpenAI)


def test_azure_provider_with_openai_model():
    model = OpenAIModel(
        model_name='gpt-4o',
        provider=AzureProvider(
            azure_endpoint='https://project-id.openai.azure.com/',
            api_version='2023-03-15-preview',
            api_key='1234567890',
        ),
    )
    assert isinstance(model, OpenAIModel)
    assert isinstance(model.client, AsyncAzureOpenAI)


def test_azure_provider_with_azure_openai_client():
    client = AsyncAzureOpenAI(
        api_version='2024-12-01-preview',
        azure_endpoint='https://project-id.openai.azure.com/',
        api_key='1234567890',
    )
    provider = AzureProvider(openai_client=client)
    assert isinstance(provider.client, AsyncAzureOpenAI)


async def test_azure_provider_call(allow_model_requests: None):
    api_key = os.environ.get('AZURE_OPENAI_API_KEY', '1234567890')
    api_version = os.environ.get('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')

    provider = AzureProvider(
        api_key=api_key,
        azure_endpoint='https://pydanticai7521574644.openai.azure.com/',
        api_version=api_version,
    )
    model = OpenAIModel(model_name='gpt-4o', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is **Paris**.')


def test_azure_provider_model_profile(mocker: MockerFixture):
    provider = AzureProvider(
        azure_endpoint='https://project-id.openai.azure.com/',
        api_version='2023-03-15-preview',
        api_key='1234567890',
    )

    ns = 'pydantic_ai.providers.azure'
    meta_model_profile = mocker.patch(f'{ns}.meta_model_profile', return_value=DEFAULT_PROFILE)
    deepseek_model_profile = mocker.patch(f'{ns}.deepseek_model_profile', return_value=DEFAULT_PROFILE)
    mistral_model_profile = mocker.patch(f'{ns}.mistral_model_profile', return_value=DEFAULT_PROFILE)
    cohere_model_profile = mocker.patch(f'{ns}.cohere_model_profile', return_value=DEFAULT_PROFILE)
    grok_model_profile = mocker.patch(f'{ns}.grok_model_profile', return_value=DEFAULT_PROFILE)
    openai_model_profile = mocker.patch(f'{ns}.openai_model_profile', return_value=DEFAULT_PROFILE)

    meta_profile = provider.model_profile('Llama-4-Scout-17B-16E')
    meta_model_profile.assert_called_with('llama-4-scout-17b-16e')
    assert meta_profile == meta_model_profile.return_value

    meta_profile = provider.model_profile('Meta-Llama-3.1-405B-Instruct')
    meta_model_profile.assert_called_with('llama-3.1-405b-instruct')
    assert meta_profile == meta_model_profile.return_value

    deepseek_profile = provider.model_profile('DeepSeek-R1')
    deepseek_model_profile.assert_called_with('deepseek-r1')
    assert deepseek_profile == deepseek_model_profile.return_value

    mistral_profile = provider.model_profile('mistral-medium-2505')
    mistral_model_profile.assert_called_with('mistral-medium-2505')
    assert mistral_profile == mistral_model_profile.return_value

    mistral_profile = provider.model_profile('mistralai-Mixtral-8x22B-Instruct-v0-1')
    mistral_model_profile.assert_called_with('mixtral-8x22b-instruct-v0-1')
    assert mistral_profile == mistral_model_profile.return_value

    cohere_profile = provider.model_profile('cohere-command-a')
    cohere_model_profile.assert_called_with('command-a')
    assert cohere_profile == cohere_model_profile.return_value

    grok_profile = provider.model_profile('grok-3')
    grok_model_profile.assert_called_with('grok-3')
    assert grok_profile == grok_model_profile.return_value

    openai_profile = provider.model_profile('o4-mini')
    openai_model_profile.assert_called_with('o4-mini')
    assert openai_profile == openai_model_profile.return_value

    unknown_profile = provider.model_profile('unknown-model')
    openai_model_profile.assert_called_with('unknown-model')
    assert unknown_profile == openai_model_profile.return_value
