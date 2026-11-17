import httpx2
import pytest

from pydantic_ai import Agent

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from pydantic_ai.exceptions import UserError
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='OpenAI client not installed'),
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
        UserError,
        match=(
            r'Set the `OPENAI_API_KEY` environment variable or pass it via `OpenAIProvider\(api_key=\.\.\.\)`'
            r" to use the OpenAI provider\. To try Pydantic AI without an API key, use the built-in test model: `Agent\('test'\)`\."
        ),
    ):
        OpenAIProvider()


def test_init_of_openai_with_base_url_and_without_api_key(env: TestEnv):
    env.remove('OPENAI_API_KEY')
    provider = OpenAIProvider(base_url='https://example.com/v1')
    assert provider.client.api_key == 'api-key-not-set'


def test_init_of_openai_with_base_url_env_var_and_without_api_key(env: TestEnv):
    env.remove('OPENAI_API_KEY')
    env.set('OPENAI_BASE_URL', 'https://example.com/v1')
    provider = OpenAIProvider()
    assert provider.client.api_key == 'api-key-not-set'


async def test_async_api_key_callback_refreshes_per_request(allow_model_requests: None):
    """An async `api_key` callable is resolved by the OpenAI SDK for every request.

    Not a VCR test: a live recording cannot show two distinct valid keys.
    """
    token = 'old-token'

    async def api_key() -> str:
        return token

    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                'id': 'chatcmpl-1',
                'object': 'chat.completion',
                'created': 0,
                'model': 'gpt-4o',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
            },
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        agent = Agent(OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=api_key, http_client=http_client)))
        assert (await agent.run('hello')).output == 'ok'
        token = 'refreshed-token'
        assert (await agent.run('again')).output == 'ok'
    assert [request.headers['authorization'] for request in requests] == [
        'Bearer old-token',
        'Bearer refreshed-token',
    ]
