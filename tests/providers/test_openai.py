import httpx
import pytest

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from openai import OpenAIError

    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='OpenAI client not installed'),
    pytest.mark.anyio,
]


def test_init_with_base_url():
    provider = OpenAIProvider(base_url='https://example.com/v1', api_key='foobar')
    assert provider.base_url == 'https://example.com/v1/'
    assert provider.client.api_key == 'foobar'


def test_init_with_no_api_key_will_still_setup_client():
    provider = OpenAIProvider(base_url='http://localhost:19434/v1')
    assert provider.base_url == 'http://localhost:19434/v1/'


def test_init_with_non_openai_model():
    provider = OpenAIProvider(base_url='https://example.com/v1/')
    assert provider.base_url == 'https://example.com/v1/'


def test_init_of_openai_without_api_key_raises_error(env: TestEnv):
    env.remove('OPENAI_API_KEY')
    with pytest.raises(
        OpenAIError,
        match='^The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable$',
    ):
        OpenAIProvider()


def test_init_of_openai_with_base_url_and_without_api_key(env: TestEnv):
    env.remove('OPENAI_API_KEY')
    provider = OpenAIProvider(base_url='https://example.com/v1')
    assert provider.client.api_key == 'api-key-not-set'


async def test_init_with_http_client():
    async with httpx.AsyncClient() as http_client:
        provider = OpenAIProvider(http_client=http_client, api_key='foobar')
        assert provider.client._client == http_client  # type: ignore
