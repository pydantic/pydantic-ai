import os

import pytest
from inline_snapshot import snapshot
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.agent import Agent
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile

from ..conftest import try_import

with try_import() as imports_successful:
    from openai import AsyncAzureOpenAI

    from pydantic_ai.models.openai import OpenAIChatModel
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
    model = OpenAIChatModel(
        model_name='gpt-4o',
        provider=AzureProvider(
            azure_endpoint='https://project-id.openai.azure.com/',
            api_version='2023-03-15-preview',
            api_key='1234567890',
        ),
    )
    assert isinstance(model, OpenAIChatModel)
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
    api_key = os.getenv('AZURE_OPENAI_API_KEY', '1234567890')
    api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')

    provider = AzureProvider(
        api_key=api_key,
        azure_endpoint='https://pydanticai7521574644.openai.azure.com/',
        api_version=api_version,
    )
    model = OpenAIChatModel(model_name='gpt-4o', provider=provider)
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
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    deepseek_model_profile_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    mistral_model_profile_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    cohere_model_profile_mock = mocker.patch(f'{ns}.cohere_model_profile', wraps=cohere_model_profile)
    grok_model_profile_mock = mocker.patch(f'{ns}.grok_model_profile', wraps=grok_model_profile)
    openai_model_profile_mock = mocker.patch(f'{ns}.openai_model_profile', wraps=openai_model_profile)

    meta_profile = provider.model_profile('Llama-4-Scout-17B-16E')
    meta_model_profile_mock.assert_called_with('llama-4-scout-17b-16e')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    meta_profile = provider.model_profile('Meta-Llama-3.1-405B-Instruct')
    meta_model_profile_mock.assert_called_with('llama-3.1-405b-instruct')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    deepseek_profile = provider.model_profile('DeepSeek-R1')
    deepseek_model_profile_mock.assert_called_with('deepseek-r1')
    assert deepseek_profile is not None
    assert deepseek_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    mistral_profile = provider.model_profile('mistral-medium-2505')
    mistral_model_profile_mock.assert_called_with('mistral-medium-2505')
    assert mistral_profile is not None
    assert mistral_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    mistral_profile = provider.model_profile('mistralai-Mixtral-8x22B-Instruct-v0-1')
    mistral_model_profile_mock.assert_called_with('mixtral-8x22b-instruct-v0-1')
    assert mistral_profile is not None
    assert mistral_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    cohere_profile = provider.model_profile('cohere-command-a')
    cohere_model_profile_mock.assert_called_with('command-a')
    assert cohere_profile is not None
    assert cohere_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    grok_profile = provider.model_profile('grok-3')
    grok_model_profile_mock.assert_called_with('grok-3')
    assert grok_profile is not None
    assert grok_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    openai_profile = provider.model_profile('o4-mini')
    openai_model_profile_mock.assert_called_with('o4-mini')
    assert openai_profile is not None
    assert openai_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    unknown_profile = provider.model_profile('unknown-model')
    openai_model_profile_mock.assert_called_with('unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.json_schema_transformer == OpenAIJsonSchemaTransformer
