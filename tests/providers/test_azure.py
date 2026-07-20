import os
from typing import cast

import pytest
from pytest_mock import MockerFixture

from pydantic_ai import BinaryContent, DocumentUrl
from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile
from pydantic_ai.settings import ModelSettings

from .._inline_snapshot import snapshot
from ..conftest import try_import
from ..models.mock_openai import MockOpenAI, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from openai import AsyncAzureOpenAI, AsyncOpenAI
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.openai import OpenAIProvider


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


def test_azure_provider_with_http_client():
    import httpx

    http_client = httpx.AsyncClient()
    provider = AzureProvider(
        azure_endpoint='https://project-id.openai.azure.com/',
        api_key='1234567890',
        api_version='2024-12-01-preview',
        http_client=http_client,
    )
    assert isinstance(provider.client, AsyncAzureOpenAI)
    assert provider._own_http_client is None  # pyright: ignore[reportPrivateUsage]


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
    assert meta_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    meta_profile = provider.model_profile('Meta-Llama-3.1-405B-Instruct')
    meta_model_profile_mock.assert_called_with('llama-3.1-405b-instruct')
    assert meta_profile is not None
    assert meta_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    deepseek_profile = provider.model_profile('DeepSeek-R1')
    deepseek_model_profile_mock.assert_called_with('deepseek-r1')
    assert deepseek_profile is not None
    assert deepseek_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    mistral_profile = provider.model_profile('mistral-medium-2505')
    mistral_model_profile_mock.assert_called_with('mistral-medium-2505')
    assert mistral_profile is not None
    assert mistral_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
    # Azure AI Foundry's Mistral gateway rejects `max_completion_tokens` with 422 (#6593); the
    # override is applied by `AzureProvider.model_profile` itself, scoped to the Mistral prefixes.
    assert mistral_profile.get('openai_chat_supports_max_completion_tokens', True) is False

    mistral_profile = provider.model_profile('mistralai-Mixtral-8x22B-Instruct-v0-1')
    mistral_model_profile_mock.assert_called_with('mixtral-8x22b-instruct-v0-1')
    assert mistral_profile is not None
    assert mistral_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
    assert mistral_profile.get('openai_chat_supports_max_completion_tokens', True) is False

    # `ministral`/`magistral` are Mistral-family model lines (also grouped under Bedrock's `mistral`
    # provider namespace) and must get the same Azure gateway override as `mistral`/`mistralai-`.
    ministral_profile = provider.model_profile('Ministral-3B')
    mistral_model_profile_mock.assert_called_with('ministral-3b')
    assert ministral_profile is not None
    assert ministral_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
    assert ministral_profile.get('openai_chat_supports_max_completion_tokens', True) is False

    magistral_profile = provider.model_profile('magistral-small-2509')
    mistral_model_profile_mock.assert_called_with('magistral-small-2509')
    assert magistral_profile is not None
    assert magistral_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
    assert magistral_profile.get('openai_chat_supports_max_completion_tokens', True) is False

    cohere_profile = provider.model_profile('cohere-command-a')
    cohere_model_profile_mock.assert_called_with('command-a')
    assert cohere_profile is not None
    assert cohere_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
    # The Azure-only override is scoped to the Mistral prefixes and must not leak to sibling gateways.
    assert 'openai_chat_supports_max_completion_tokens' not in cohere_profile

    grok_profile = provider.model_profile('grok-3')
    grok_model_profile_mock.assert_called_with('grok-3')
    assert grok_profile is not None
    assert grok_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    openai_profile = provider.model_profile('o4-mini')
    openai_model_profile_mock.assert_called_with('o4-mini')
    assert openai_profile is not None
    assert openai_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    unknown_profile = provider.model_profile('unknown-model')
    openai_model_profile_mock.assert_called_with('unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer


async def test_azure_document_input_not_supported(allow_model_requests: None):
    provider = AzureProvider(
        azure_endpoint='https://project-id.openai.azure.com/',
        api_version='2023-03-15-preview',
        api_key='1234567890',
    )
    model = OpenAIChatModel(model_name='gpt-4o', provider=provider)
    agent = Agent(model)

    with pytest.raises(
        UserError,
        match=r"Azure's Chat Completions API does not support document input.*OpenAIResponsesModel",
    ):
        await agent.run(
            [
                'Summarize this document',
                BinaryContent(data=b'%PDF-1.4 test', media_type='application/pdf'),
            ]
        )


async def test_azure_document_url_input_not_supported(allow_model_requests: None):
    provider = AzureProvider(
        azure_endpoint='https://project-id.openai.azure.com/',
        api_version='2023-03-15-preview',
        api_key='1234567890',
    )
    model = OpenAIChatModel(model_name='gpt-4o', provider=provider)
    agent = Agent(model)

    with pytest.raises(
        UserError,
        match=r"Azure's Chat Completions API does not support document input.*OpenAIResponsesModel",
    ):
        await agent.run(['Summarize this document', DocumentUrl(url='https://example.com/test.pdf')])


def test_azure_provider_foundry_serverless_endpoint():
    provider = AzureProvider(
        azure_endpoint='https://gpt-oss-120b.eastus2.models.ai.azure.com',
        api_key='test-key-123',
    )
    assert provider.name == 'azure'
    # Serverless model endpoints reject the `api-version` query parameter, so we
    # must use plain AsyncOpenAI rather than AsyncAzureOpenAI.
    assert type(provider.client) is AsyncOpenAI
    assert provider.base_url == 'https://gpt-oss-120b.eastus2.models.ai.azure.com/v1/'


def test_azure_provider_v1_endpoint_rejects_api_version():
    with pytest.raises(UserError, match='`api_version` must not be set'):
        AzureProvider(
            azure_endpoint='https://gpt-oss-120b.eastus2.models.ai.azure.com',
            api_version='2024-12-01-preview',
            api_key='test-key-123',
        )


def test_azure_provider_openai_v1_ga_endpoint():
    # https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle
    provider = AzureProvider(
        azure_endpoint='https://project-id.openai.azure.com/openai/v1/',
        api_key='test-key-123',
    )
    assert type(provider.client) is AsyncOpenAI
    assert provider.base_url == 'https://project-id.openai.azure.com/openai/v1/'


def test_azure_provider_foundry_serverless_with_openai_model():
    model = OpenAIChatModel(
        model_name='gpt-oss-120b',
        provider=AzureProvider(
            azure_endpoint='https://gpt-oss-120b.eastus2.models.ai.azure.com',
            api_key='test-key-123',
        ),
    )
    assert type(model.client) is AsyncOpenAI


async def test_azure_mistral_max_tokens_uses_legacy_field(allow_model_requests: None) -> None:
    """Azure AI Foundry's Mistral gateway rejects `max_completion_tokens` with 422 (#6593).

    Unit test (not VCR): cassette matchers ignore the request body, so a VCR test would stay
    green even if `max_tokens` were still routed to the wrong key. Goes through a real
    `AzureProvider` (rather than a manually-injected `profile=`) so the assertion exercises the
    actual gateway-composition path in `AzureProvider.model_profile`. Mirrors the gateway-specific
    override OpenRouter applies for the same underlying limitation (#5926).
    """
    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIChatModel(
        'mistral-medium-2505', provider=AzureProvider(openai_client=cast(AsyncAzureOpenAI, mock_client))
    )
    agent = Agent(m, model_settings=ModelSettings(max_tokens=256))

    await agent.run('Hello')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['max_tokens'] == 256
    assert 'max_completion_tokens' not in kwargs


async def test_azure_ministral_max_tokens_uses_legacy_field(allow_model_requests: None) -> None:
    """Same as `test_azure_mistral_max_tokens_uses_legacy_field`, but for the exact deployment name
    reported in #6593 (`Ministral-3B`), which doesn't share the `mistral`/`mistralai-` prefix and
    would otherwise fall through to the generic OpenAI profile and keep sending
    `max_completion_tokens`.
    """
    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIChatModel('Ministral-3B', provider=AzureProvider(openai_client=cast(AsyncAzureOpenAI, mock_client)))
    agent = Agent(m, model_settings=ModelSettings(max_tokens=256))

    await agent.run('Hello')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['max_tokens'] == 256
    assert 'max_completion_tokens' not in kwargs


async def test_non_azure_openai_provider_mistral_name_unaffected(allow_model_requests: None) -> None:
    """A bare `OpenAIProvider` doesn't run Mistral family detection, so a Mistral-named model still
    gets the default OpenAI profile and sends `max_completion_tokens`.

    Confirms the Azure-only override doesn't leak into the generic `OpenAIProvider` path, and
    documents that the reporter's exact repro in #6593 (a bare `OpenAIProvider` with an unprefixed
    deployment name) isn't fixed by this change — it needs an explicit `profile=` until
    `OpenAIProvider` family-detection is addressed separately.
    """
    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIChatModel('Ministral-3B', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m, model_settings=ModelSettings(max_tokens=256))

    await agent.run('Hello')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['max_completion_tokens'] == 256
    assert 'max_tokens' not in kwargs
