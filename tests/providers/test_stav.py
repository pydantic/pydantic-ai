import re

import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import httpx2
    import openai

    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.stav import StavProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_stav_provider(env: TestEnv):
    env.remove('STAV_BASE_URL')
    provider = StavProvider(api_key='api-key')
    assert provider.name == 'stav'
    assert provider.base_url == 'https://api.stav.ai/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_stav_provider_api_key_from_env(env: TestEnv):
    env.set('STAV_API_KEY', 'env-api-key')
    provider = StavProvider()
    assert provider.client.api_key == 'env-api-key'


def test_stav_provider_need_api_key(env: TestEnv) -> None:
    env.remove('STAV_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `STAV_API_KEY` environment variable or pass it via `StavProvider(api_key=...)` '
            'to use the Stav provider.'
        ),
    ):
        StavProvider()


def test_stav_provider_base_url_from_env(env: TestEnv):
    env.set('STAV_BASE_URL', 'https://staging.stav.ai/v1')
    provider = StavProvider(api_key='api-key')
    assert provider.base_url == 'https://staging.stav.ai/v1'


@pytest.mark.parametrize(
    ('region', 'expected'),
    [
        ('eu', 'eu'),
        ('eea', 'eea'),
        ('no', 'no'),
        (['no', 'se'], 'no,se'),
        (('eu',), 'eu'),
        (None, None),
    ],
)
async def test_stav_region_sent_on_the_wire(
    allow_model_requests: None, region: str | list[str] | tuple[str, ...] | None, expected: str | None
):
    """The region reaches every request as `X-Stav-Runs-In`, and no header goes out without one.

    Not a VCR test: the claim is about a request header, which a mock transport observes directly.
    """
    runs_in: list[str | None] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        runs_in.append(request.headers.get('X-Stav-Runs-In'))
        return httpx2.Response(
            200,
            json={
                'id': 'chatcmpl-1',
                'object': 'chat.completion',
                'created': 0,
                'model': 'mistral-ai/mistral-large-4',
                'choices': [
                    {
                        'index': 0,
                        'message': {'role': 'assistant', 'content': 'Paris'},
                        'finish_reason': 'stop',
                    }
                ],
            },
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        provider = StavProvider(api_key='api-key', region=region, http_client=http_client)
        agent = Agent(OpenAIChatModel('mistral-ai/mistral-large-4', provider=provider))
        await agent.run('What is the capital of France?')

    assert runs_in == [expected]


def test_stav_responses_model_by_name(env: TestEnv):
    env.set('STAV_API_KEY', 'api-key')
    env.remove('STAV_BASE_URL')
    model = OpenAIResponsesModel('openai/gpt-5-mini', provider='stav')
    assert model.system == 'stav'
    assert model.base_url == 'https://api.stav.ai/v1/'


async def test_stav_responses_region_sent_on_the_wire(allow_model_requests: None):
    """A Responses API request goes to Stav's `/v1/responses` with the region header.

    Not a VCR test: the claim is about the request path and a header, which a mock transport observes directly.
    """
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append((request.url.path, request.headers.get('X-Stav-Runs-In')))
        return httpx2.Response(
            200,
            json={
                'id': 'resp-1',
                'object': 'response',
                'created_at': 0,
                'model': 'openai/gpt-5-mini',
                'status': 'completed',
                'output': [
                    {
                        'id': 'msg-1',
                        'type': 'message',
                        'role': 'assistant',
                        'status': 'completed',
                        'content': [{'type': 'output_text', 'text': 'Paris', 'annotations': []}],
                    }
                ],
            },
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        provider = StavProvider(api_key='api-key', region='eea', http_client=http_client)
        agent = Agent(OpenAIResponsesModel('openai/gpt-5-mini', provider=provider))
        await agent.run('What is the capital of France?')

    assert seen == [('/v1/responses', 'eea')]


@pytest.mark.parametrize('region', [[], '', ['no', '']])
def test_stav_provider_region_empty(region: list[str] | str):
    with pytest.raises(UserError, match=re.escape('`region` must be a region or a non-empty list of regions')):
        StavProvider(api_key='api-key', region=region)


def test_stav_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key', base_url='https://api.stav.ai/v1')
    provider = StavProvider(openai_client=openai_client)
    assert provider.client is openai_client
    assert provider.base_url == 'https://api.stav.ai/v1/'


def test_stav_openai_client_with_region() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key', base_url='https://api.stav.ai/v1')
    with pytest.raises(UserError, match=re.escape('Cannot set `region` together with `openai_client`')):
        StavProvider(openai_client=openai_client, region='eu')  # pyright: ignore[reportCallIssue]
