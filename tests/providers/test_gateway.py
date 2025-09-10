import os
import re
from typing import Any, Literal

import httpx
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers import Provider
    from pydantic_ai.providers.gateway import GatewayProvider
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
    provider = GatewayProvider(provider_name, base_url='https://example.com/', api_key='foobar')
    assert isinstance(provider, provider_cls)
    assert provider.base_url == 'https://example.com/openai/'
    assert provider.client.api_key == 'foobar'


def test_init_gateway_without_api_key_raises_error(env: TestEnv):
    env.remove('PYDANTIC_AI_GATEWAY_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `PYDANTIC_AI_GATEWAY_API_KEY` environment variable or pass it via `GatewayProvider(api_key=...)` to use the Pydantic AI Gateway provider.'
        ),
    ):
        GatewayProvider('openai')


async def test_init_with_http_client():
    async with httpx.AsyncClient() as http_client:
        provider = GatewayProvider('openai', http_client=http_client, api_key='foobar')
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


async def test_gateway_provider_with_openai(allow_model_requests: None, gateway_api_key: str):
    provider = GatewayProvider('openai', api_key=gateway_api_key, base_url='http://localhost:8787')
    model = OpenAIChatModel('gpt-5', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('Paris.')
