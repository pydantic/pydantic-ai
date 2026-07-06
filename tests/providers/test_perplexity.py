from __future__ import annotations as _annotations

import os
import re
from datetime import datetime, timezone

import httpx
import pytest

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, UserPromptPart, __version__
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsStr, TestEnv, try_import
from ..models.mock_openai import (
    MockOpenAI,
    completion_message,
)

with try_import() as imports_successful:
    import openai
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.models import infer_model
    from pydantic_ai.models.perplexity import PerplexityModel
    from pydantic_ai.providers.perplexity import PerplexityProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_perplexity_provider():
    provider = PerplexityProvider(api_key='api-key')
    assert provider.name == 'perplexity'
    assert provider.base_url == 'https://api.perplexity.ai'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_perplexity_provider_need_api_key(env: TestEnv) -> None:
    env.remove('PERPLEXITY_API_KEY')
    env.remove('PPLX_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `PERPLEXITY_API_KEY` environment variable or pass it via `PerplexityProvider(api_key=...)`'
            ' to use the Perplexity provider.'
        ),
    ):
        PerplexityProvider()


def test_perplexity_provider_pplx_alias(env: TestEnv) -> None:
    env.remove('PERPLEXITY_API_KEY')
    env.set('PPLX_API_KEY', 'aliased-key')
    provider = PerplexityProvider()
    assert provider.client.api_key == 'aliased-key'


def test_perplexity_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = PerplexityProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_perplexity_provider_sends_attribution_header(allow_model_requests: None) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'id': 'pplx-test',
                'choices': [
                    {
                        'finish_reason': 'stop',
                        'index': 0,
                        'message': {'content': 'Perplexity attribution sent.', 'role': 'assistant'},
                    }
                ],
                'created': 1704067200,
                'model': 'sonar-pro',
                'object': 'chat.completion',
                'usage': {'completion_tokens': 3, 'prompt_tokens': 4, 'total_tokens': 7},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = PerplexityModel('sonar-pro', provider=PerplexityProvider(http_client=http_client, api_key='api-key'))
    agent = Agent(model)

    result = await agent.run('Send attribution header.')

    assert result.output == 'Perplexity attribution sent.'
    assert requests[0].headers['X-Pplx-Integration'] == f'pydantic-ai/{__version__}'


def test_perplexity_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = PerplexityProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_perplexity_provider_set_http_client() -> None:
    new_http_client = httpx.AsyncClient()
    provider = PerplexityProvider(api_key='api-key')
    provider._set_http_client(new_http_client)  # pyright: ignore[reportPrivateUsage]
    assert provider.client._client == new_http_client  # type: ignore[reportPrivateUsage]


def test_perplexity_model_profile() -> None:
    provider = PerplexityProvider(api_key='api-key')
    model = PerplexityModel('sonar-pro', provider=provider)
    profile = model.profile
    assert profile.get('json_schema_transformer') is OpenAIJsonSchemaTransformer
    # Perplexity searches natively and is not configurable per request, so WebSearchTool is not supported.
    assert not profile.get('openai_chat_supports_web_search', False)
    assert WebSearchTool not in profile.get('supported_native_tools', frozenset())


@pytest.mark.parametrize('model_name', ['sonar-reasoning', 'sonar-reasoning-pro', 'sonar-deep-research'])
def test_perplexity_reasoning_model_profile(model_name: str) -> None:
    provider = PerplexityProvider(api_key='api-key')
    model = PerplexityModel(model_name, provider=provider)
    assert model.profile.get('supports_thinking') is True
    assert model.profile.get('ignore_streamed_leading_whitespace') is True


def test_perplexity_non_reasoning_model_profile() -> None:
    provider = PerplexityProvider(api_key='api-key')
    model = PerplexityModel('sonar-pro', provider=provider)
    assert model.profile.get('supports_thinking') is False
    assert model.profile.get('ignore_streamed_leading_whitespace') is False


def test_infer_perplexity_model(env: TestEnv) -> None:
    env.set('PERPLEXITY_API_KEY', 'api-key')
    model = infer_model('perplexity:sonar-pro')
    assert isinstance(model, PerplexityModel)
    assert model.model_name == 'sonar-pro'


@pytest.mark.anyio
async def test_perplexity_web_search_tool_not_supported(allow_model_requests: None) -> None:
    model = PerplexityModel('sonar-pro', provider=PerplexityProvider(api_key='api-key'))
    agent = Agent(model, capabilities=[NativeTool(WebSearchTool())])

    with pytest.raises(UserError, match='WebSearchTool is not supported'):
        await agent.run('Search for Pydantic AI.')


@pytest.mark.anyio
async def test_perplexity_response_without_citations(allow_model_requests: None) -> None:
    # Unit test rather than VCR: exercises the empty-provider-details path (a response with no
    # citations/search_results) directly, so the `or None` collapse stays covered and asserted.
    c = completion_message(ChatCompletionMessage(content='Plain answer.', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = PerplexityModel('sonar-pro', provider=PerplexityProvider(openai_client=mock_client))
    agent = Agent(model)

    result = await agent.run('Hello.')

    assert result.output == 'Plain answer.'
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    provider_details = response.provider_details or {}
    assert 'citations' not in provider_details
    assert 'search_results' not in provider_details


@pytest.mark.anyio
@pytest.mark.vcr
async def test_perplexity_agent_run(allow_model_requests: None, env: TestEnv) -> None:
    env.set(
        'PERPLEXITY_API_KEY',
        os.getenv('PERPLEXITY_API_KEY', os.getenv('PPLX_API_KEY', 'mock-api-key')),
    )
    agent = Agent('perplexity:sonar-pro')

    result = await agent.run('What is Pydantic AI? Answer in one sentence.')

    assert result.output == snapshot(
        'Pydantic AI is a Python agent framework from the Pydantic team for building production-grade applications with generative AI.'
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is Pydantic AI? Answer in one sentence.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Pydantic AI is a Python agent framework from the Pydantic team for building production-grade applications with generative AI.'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=13,
                    output_tokens=21,
                    details={'citation_tokens': 125, 'num_search_queries': 1},
                ),
                model_name='sonar-pro',
                timestamp=IsDatetime(),
                provider_name='perplexity',
                provider_url='https://api.perplexity.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'citations': ['https://ai.pydantic.dev/'],
                    'search_results': [
                        {
                            'title': 'Pydantic AI',
                            'url': 'https://ai.pydantic.dev/',
                            'date': '2026-05-01',
                            'last_updated': '2026-05-01',
                            'snippet': 'Pydantic AI is a Python agent framework designed to make it less painful to build production-grade applications with generative AI.',
                            'source': 'web',
                        }
                    ],
                    'timestamp': datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc),
                },
                provider_response_id='pplx-5250-test',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
