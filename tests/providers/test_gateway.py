import os
import re
from typing import Any, Literal
from unittest.mock import patch

import httpx
import pytest
from inline_snapshot import snapshot
from inline_snapshot.extra import raises

from pydantic_ai import Agent, UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers import Provider
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.bedrock import BedrockProvider
    from pydantic_ai.providers.gateway import GATEWAY_BASE_URL, gateway_provider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.groq import GroqProvider
    from pydantic_ai.providers.openai import OpenAIProvider


if not imports_successful():
    pytest.skip('Providers not installed', allow_module_level=True)  # pragma: lax no cover

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]


@pytest.mark.parametrize(
    'provider_name, provider_cls, route',
    [
        ('openai', OpenAIProvider, 'openai'),
        ('openai-chat', OpenAIProvider, 'openai'),
        ('openai-responses', OpenAIProvider, 'openai-responses'),
    ],
)
def test_init_with_base_url(
    provider_name: Literal['openai', 'openai-chat', 'openai-responses'], provider_cls: type[Provider[Any]], route: str
):
    provider = gateway_provider(provider_name, base_url='https://example.com/', api_key='foobar')
    assert isinstance(provider, provider_cls)
    assert provider.base_url == f'https://example.com/{route}/'
    assert provider.client.api_key == 'foobar'


def test_init_gateway_without_api_key_raises_error(env: TestEnv):
    env.remove('PYDANTIC_AI_GATEWAY_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `PYDANTIC_AI_GATEWAY_API_KEY` environment variable or pass it via `gateway_provider(..., api_key=...)` to use the Pydantic AI Gateway provider.'
        ),
    ):
        gateway_provider('openai')


async def test_init_with_http_client():
    async with httpx.AsyncClient() as http_client:
        provider = gateway_provider('openai', http_client=http_client, api_key='foobar')
        assert provider.client._client == http_client  # type: ignore


@pytest.fixture
def gateway_api_key():
    return os.getenv('PYDANTIC_AI_GATEWAY_API_KEY', 'test-api-key')


@pytest.fixture(scope='module')
def vcr_config():
    return {
        'ignore_localhost': False,
        # Note: additional header filtering is done inside the serializer
        'filter_headers': ['authorization', 'x-api-key'],
        'decode_compressed_response': True,
    }


@patch.dict(
    os.environ, {'PYDANTIC_AI_GATEWAY_API_KEY': 'test-api-key', 'PYDANTIC_AI_GATEWAY_BASE_URL': GATEWAY_BASE_URL}
)
@pytest.mark.parametrize(
    'provider_name, provider_cls, route',
    [
        ('openai', OpenAIProvider, 'openai'),
        ('openai-chat', OpenAIProvider, 'openai'),
        ('openai-responses', OpenAIProvider, 'openai-responses'),
        ('groq', GroqProvider, 'groq'),
        ('google-vertex', GoogleProvider, 'google-vertex'),
        ('anthropic', AnthropicProvider, 'anthropic'),
        ('bedrock', BedrockProvider, 'bedrock'),
    ],
)
def test_gateway_provider(provider_name: str, provider_cls: type[Provider[Any]], route: str):
    provider = gateway_provider(provider_name)
    assert isinstance(provider, provider_cls)

    # Some providers add a trailing slash, others don't
    assert provider.base_url in (f'{GATEWAY_BASE_URL}/{route}/', f'{GATEWAY_BASE_URL}/{route}')


@patch.dict(os.environ, {'PYDANTIC_AI_GATEWAY_API_KEY': 'test-api-key'})
def test_gateway_provider_unknown():
    with raises(snapshot('UserError: Unknown upstream provider: foo')):
        gateway_provider('foo')


async def test_gateway_provider_with_openai(allow_model_requests: None, gateway_api_key: str):
    provider = gateway_provider('openai', api_key=gateway_api_key, base_url='http://localhost:8787')
    model = OpenAIChatModel('gpt-5', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('Paris.')


async def test_gateway_provider_with_openai_responses(allow_model_requests: None, gateway_api_key: str):
    provider = gateway_provider('openai-responses', api_key=gateway_api_key, base_url='http://localhost:8787')
    model = OpenAIResponsesModel('gpt-5', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('Paris.')


async def test_gateway_provider_with_groq(allow_model_requests: None, gateway_api_key: str):
    provider = gateway_provider('groq', api_key=gateway_api_key, base_url='http://localhost:8787')
    model = GroqModel('llama-3.3-70b-versatile', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.')


async def test_gateway_provider_with_google_vertex(allow_model_requests: None, gateway_api_key: str):
    provider = gateway_provider('google-vertex', api_key=gateway_api_key, base_url='http://localhost:8787')
    model = GoogleModel('gemini-2.5-flash', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is **Paris**.')


async def test_gateway_provider_with_anthropic(allow_model_requests: None, gateway_api_key: str):
    provider = gateway_provider('anthropic', api_key=gateway_api_key, base_url='http://localhost:8787')
    model = AnthropicModel('claude-sonnet-4-5', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.')


async def test_gateway_provider_with_bedrock(allow_model_requests: None, gateway_api_key: str):
    provider = gateway_provider('bedrock', api_key=gateway_api_key, base_url='http://localhost:8787')
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, and it is a major center for culture, commerce, fashion, and international diplomacy. The city is known for its historical landmarks, such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Champs-Élysées, among many other attractions.'
    )


@patch.dict(
    os.environ, {'PYDANTIC_AI_GATEWAY_API_KEY': 'test-api-key', 'PYDANTIC_AI_GATEWAY_BASE_URL': GATEWAY_BASE_URL}
)
async def test_model_provider_argument():
    model = OpenAIChatModel('gpt-5', provider='gateway')
    assert GATEWAY_BASE_URL in model._provider.base_url  # type: ignore[reportPrivateUsage]

    model = OpenAIResponsesModel('gpt-5', provider='gateway')
    assert GATEWAY_BASE_URL in model._provider.base_url  # type: ignore[reportPrivateUsage]

    model = GroqModel('llama-3.3-70b-versatile', provider='gateway')
    assert GATEWAY_BASE_URL in model._provider.base_url  # type: ignore[reportPrivateUsage]

    model = GoogleModel('gemini-1.5-flash', provider='gateway')
    assert GATEWAY_BASE_URL in model._provider.base_url  # type: ignore[reportPrivateUsage]

    model = AnthropicModel('claude-sonnet-4-5', provider='gateway')
    assert GATEWAY_BASE_URL in model._provider.base_url  # type: ignore[reportPrivateUsage]

    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider='gateway')
    assert GATEWAY_BASE_URL in model._provider.base_url  # type: ignore[reportPrivateUsage]


async def test_gateway_provider_routing_group(gateway_api_key: str):
    provider = gateway_provider('openai', route='potato', api_key=gateway_api_key)
    assert provider.client.base_url.path.endswith('/potato/')
