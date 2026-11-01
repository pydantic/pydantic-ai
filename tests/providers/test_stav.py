import re

import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

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
    assert 'X-Stav-Runs-In' not in provider.client.default_headers


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


def test_stav_provider_base_url():
    provider = StavProvider(api_key='api-key', base_url='https://staging.stav.ai/v1')
    assert provider.base_url == 'https://staging.stav.ai/v1'
    assert str(provider.client.base_url) == 'https://staging.stav.ai/v1/'


def test_stav_provider_base_url_from_env(env: TestEnv):
    env.set('STAV_BASE_URL', 'https://staging.stav.ai/v1')
    provider = StavProvider(api_key='api-key')
    assert provider.base_url == 'https://staging.stav.ai/v1'


def test_stav_provider_region():
    provider = StavProvider(api_key='api-key', region='eu')
    assert provider.client.default_headers['X-Stav-Runs-In'] == 'eu'


@pytest.mark.parametrize(('region', 'expected'), [('eu', 'eu'), ('eea', 'eea'), (None, None)])
async def test_stav_region_sent_on_the_wire(allow_model_requests: None, region: str | None, expected: str | None):
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
                'model': 'mistral-ai/mistral-large-2411',
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
        agent = Agent(OpenAIChatModel('mistral-ai/mistral-large-2411', provider=provider))
        result = await agent.run('What is the capital of France?')

    assert result.output == 'Paris'
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
        result = await agent.run('What is the capital of France?')

    assert result.output == 'Paris'
    assert seen == [('/v1/responses', 'eea')]


def test_stav_provider_region_country_code():
    provider = StavProvider(api_key='api-key', region='no')
    assert provider.client.default_headers['X-Stav-Runs-In'] == 'no'


def test_stav_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key', base_url='https://api.stav.ai/v1')
    provider = StavProvider(openai_client=openai_client)
    assert provider.client == openai_client
    assert provider.base_url == 'https://api.stav.ai/v1/'


def test_stav_openai_client_with_region() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key', base_url='https://api.stav.ai/v1')
    with pytest.raises(UserError, match=re.escape('Cannot set `region` together with `openai_client`')):
        StavProvider(openai_client=openai_client, region='eu')  # pyright: ignore[reportCallIssue]


def test_stav_provider_model_profile(mocker: MockerFixture):
    provider = StavProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.stav'

    anthropic_mock = mocker.patch(f'{ns}.anthropic_model_profile', wraps=anthropic_model_profile)
    deepseek_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    google_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    grok_mock = mocker.patch(f'{ns}.grok_model_profile', wraps=grok_model_profile)
    harmony_mock = mocker.patch(f'{ns}.harmony_model_profile', wraps=harmony_model_profile)
    meta_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    mistral_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    moonshotai_mock = mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile)
    openai_mock = mocker.patch(f'{ns}.openai_model_profile', wraps=openai_model_profile)
    qwen_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    zai_mock = mocker.patch(f'{ns}.zai_model_profile', wraps=zai_model_profile)

    profile = provider.model_profile('anthropic/claude-sonnet-4-5')
    anthropic_mock.assert_called_with('claude-sonnet-4-5')
    assert profile is not None

    profile = provider.model_profile('deepseek-ai/deepseek-v4-flash-0731')
    deepseek_mock.assert_called_with('deepseek-v4-flash-0731')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('google/gemma-3-27b-it')
    google_mock.assert_called_with('gemma-3-27b-it')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == GoogleJsonSchemaTransformer

    profile = provider.model_profile('x-ai/grok-4')
    grok_mock.assert_called_with('grok-4')
    profile = provider.model_profile('xai/grok-4-fast')
    grok_mock.assert_called_with('grok-4-fast')
    assert profile is not None

    profile = provider.model_profile('meta-llama/Llama-3.3-70B-Instruct')
    meta_mock.assert_called_with('llama-3.3-70b-instruct')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    profile = provider.model_profile('mistral-ai/mistral-large-2411')
    mistral_mock.assert_called_with('mistral-large-2411')
    profile = provider.model_profile('mistralai/Devstral-Small-2505')
    mistral_mock.assert_called_with('devstral-small-2505')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('moonshotai/Kimi-K2-Instruct')
    moonshotai_mock.assert_called_with('kimi-k2-instruct')
    assert profile is not None

    profile = provider.model_profile('openai/gpt-oss-120b')
    harmony_mock.assert_called_with('gpt-oss-120b')
    assert profile is not None
    assert profile.get('supports_forced_tool_choice', None) is False

    profile = provider.model_profile('openai/gpt-5-mini')
    openai_mock.assert_called_with('gpt-5-mini')
    assert profile is not None

    profile = provider.model_profile('qwen/qwen3-30b-a3b')
    qwen_mock.assert_called_with('qwen3-30b-a3b')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    profile = provider.model_profile('zai-org/glm-4.6')
    zai_mock.assert_called_with('glm-4.6')
    assert profile is not None

    profile = provider.model_profile('unknown-creator/unknown-model')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer


def test_stav_provider_model_name_without_slash():
    profile = StavProvider.model_profile('auto')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
