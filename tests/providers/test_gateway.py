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
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers import Provider
    from pydantic_ai.providers.gateway import gateway_provider, infer_model
    from pydantic_ai.providers.openai import OpenAIProvider

if not imports_successful():
    pytest.skip('OpenAI client not installed', allow_module_level=True)  # pragma: lax no cover

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]


@pytest.mark.parametrize(
    'provider_name, provider_cls',
    [('openai', OpenAIProvider), ('openai-chat', OpenAIProvider), ('openai-responses', OpenAIProvider)],
)
def test_init_with_base_url(
    provider_name: Literal['openai', 'openai-chat', 'openai-responses'], provider_cls: type[Provider[Any]]
):
    provider = gateway_provider(provider_name, base_url='https://example.com/', api_key='foobar')
    assert isinstance(provider, provider_cls)
    assert provider.base_url == 'https://example.com/openai/'
    assert provider.client.api_key == 'foobar'


def test_init_gateway_without_api_key_raises_error(env: TestEnv):
    env.remove('PYDANTIC_AI_GATEWAY_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `PYDANTIC_AI_GATEWAY_API_KEY` environment variable or pass it via `gateway_provider(api_key=...)` to use the Pydantic AI Gateway provider.'
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


@patch.dict(os.environ, {'PYDANTIC_AI_GATEWAY_API_KEY': 'test-api-key'})
def test_infer_model():
    model = infer_model('openai/gpt-5')
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == 'gpt-5'

    model = infer_model('openai-chat/gpt-5')
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == 'gpt-5'

    model = infer_model('openai-responses/gpt-5')
    assert isinstance(model, OpenAIResponsesModel)
    assert model.model_name == 'gpt-5'

    model = infer_model('groq/llama-3.3-70b-versatile')
    assert isinstance(model, GroqModel)
    assert model.model_name == 'llama-3.3-70b-versatile'

    model = infer_model('google-vertex/gemini-1.5-flash')
    assert isinstance(model, GoogleModel)
    assert model.model_name == 'gemini-1.5-flash'
    assert model.system == 'google-vertex'

    model = infer_model('anthropic/claude-3-5-sonnet-latest')
    assert isinstance(model, AnthropicModel)
    assert model.model_name == 'claude-3-5-sonnet-latest'
    assert model.system == 'anthropic'

    with raises(snapshot('UserError: The model name "gemini-1.5-flash" is not in the format "provider/model_name".')):
        infer_model('gemini-1.5-flash')

    with raises(snapshot('UserError: Unknown upstream provider: gemini-1.5-flash')):
        infer_model('gemini-1.5-flash/gemini-1.5-flash')


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
    model = GoogleModel('gemini-1.5-flash', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('Paris\n')


async def test_gateway_provider_with_anthropic(allow_model_requests: None, gateway_api_key: str):
    provider = gateway_provider('anthropic', api_key=gateway_api_key, base_url='http://localhost:8787')
    model = AnthropicModel('claude-3-5-sonnet-latest', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.')
