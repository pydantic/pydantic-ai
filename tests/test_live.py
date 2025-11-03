"""Tests of pydantic-ai actually making request to live vendor model APIs.

WARNING: running these tests will make use of the relevant API tokens (and cost money).
"""

import json
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.models import Model

pytestmark = [
    pytest.mark.skipif(os.getenv('PYDANTIC_AI_LIVE_TEST_DANGEROUS') != 'CHARGE-ME!', reason='live tests disabled'),
    pytest.mark.anyio,
]


def openai(http_client: httpx.AsyncClient, _tmp_path: Path) -> Model:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(http_client=http_client))


def gemini(_: httpx.AsyncClient, _tmp_path: Path) -> Model:
    from pydantic_ai.models.google import GoogleModel

    return GoogleModel('gemini-1.5-pro')


def vertexai(http_client: httpx.AsyncClient, tmp_path: Path) -> Model:
    from google.oauth2 import service_account

    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    if service_account_path := os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        project_id = json.loads(Path(service_account_path).read_text())['project_id']
    elif service_account_content := os.getenv('GOOGLE_SERVICE_ACCOUNT_CONTENT'):
        project_id = json.loads(service_account_content)['project_id']
        service_account_path = tmp_path / 'service_account.json'
        service_account_path.write_text(service_account_content)
    else:
        pytest.skip(
            'VertexAI live test requires GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_SERVICE_ACCOUNT_CONTENT to be set'
        )

    credentials = service_account.Credentials.from_service_account_file(  # type: ignore[reportUnknownReturnType]
        service_account_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform'],
    )
    provider = GoogleProvider(credentials=credentials, project=project_id)
    provider.client.aio._api_client._async_httpx_client = http_client  # type: ignore
    return GoogleModel('gemini-2.0-flash', provider=provider)


def groq(http_client: httpx.AsyncClient, _tmp_path: Path) -> Model:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

    return GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(http_client=http_client))


def anthropic(http_client: httpx.AsyncClient, _tmp_path: Path) -> Model:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    return AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(http_client=http_client))


def ollama(http_client: httpx.AsyncClient, _tmp_path: Path) -> Model:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(
        'qwen2:0.5b', provider=OpenAIProvider(base_url='http://localhost:11434/v1/', http_client=http_client)
    )


def mistral(http_client: httpx.AsyncClient, _tmp_path: Path) -> Model:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

    return MistralModel('mistral-small-latest', provider=MistralProvider(http_client=http_client))


def cohere(http_client: httpx.AsyncClient, _tmp_path: Path) -> Model:
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider

    return CohereModel('command-r7b-12-2024', provider=CohereProvider(http_client=http_client))


params = [
    pytest.param(openai, id='openai'),
    pytest.param(gemini, id='gemini', marks=pytest.mark.skip(reason='API seems very flaky')),
    pytest.param(vertexai, id='vertexai'),
    pytest.param(groq, id='groq', marks=pytest.mark.skip(reason='test_structured has started failing')),
    pytest.param(anthropic, id='anthropic'),
    pytest.param(ollama, id='ollama'),
    pytest.param(mistral, id='mistral'),
    pytest.param(cohere, id='cohere', marks=pytest.mark.skip(reason='Might be causing hangs in CI')),
]
GetModel = Callable[[httpx.AsyncClient, Path], Model]


@pytest.fixture
async def http_client(allow_model_requests: None) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=30) as client:
        yield client


@pytest.mark.parametrize('get_model', params)
async def test_text(http_client: httpx.AsyncClient, tmp_path: Path, get_model: GetModel):
    agent = Agent(get_model(http_client, tmp_path), model_settings={'temperature': 0.0}, retries=2)
    result = await agent.run('What is the capital of France?')
    print('Text response:', result.output)
    assert 'paris' in result.output.lower()
    print('Text usage:', result.usage())
    usage = result.usage()
    assert usage.total_tokens is not None and usage.total_tokens > 0


stream_params = [p for p in params if p.id != 'cohere']


@pytest.mark.parametrize('get_model', stream_params)
async def test_stream(http_client: httpx.AsyncClient, tmp_path: Path, get_model: GetModel):
    agent = Agent(get_model(http_client, tmp_path), model_settings={'temperature': 0.0}, retries=2)
    async with agent.run_stream('What is the capital of France?') as result:
        data = await result.get_output()
    print('Stream response:', data)
    assert 'paris' in data.lower()
    print('Stream usage:', result.usage())
    usage = result.usage()
    if get_model.__name__ != 'ollama':
        assert usage.total_tokens is not None and usage.total_tokens > 0


class MyModel(BaseModel):
    city: str


structured_params = [p for p in params if p.id != 'ollama']


@pytest.mark.parametrize('get_model', structured_params)
async def test_structured(http_client: httpx.AsyncClient, tmp_path: Path, get_model: GetModel):
    agent = Agent(get_model(http_client, tmp_path), output_type=MyModel, model_settings={'temperature': 0.0}, retries=2)

    async with agent.iter('What is the capital of the UK?') as run:
        try:
            async for _ in run:
                pass
        except UnexpectedModelBehavior as e:
            raise RuntimeError(run._graph_run.state) from e  # pyright: ignore[reportPrivateUsage])

    result = run.result
    assert result is not None

    print('Structured response:', result.output)
    assert result.output.city.lower() == 'london'
    print('Structured usage:', result.usage())
    usage = result.usage()
    assert usage.total_tokens is not None and usage.total_tokens > 0
