"""Tests of pydantic-ai actually making request to live vendor model APIs.

WARNING: running these tests will make use of the relevant API tokens (and cost money).
"""

import os
from collections.abc import AsyncIterator
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
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIModel('gpt-4o-mini', provider=OpenAIProvider(http_client=http_client))


def gemini(timeout: int, _tmp_path: Path) -> Model:
    from google import genai

    from pydantic_ai.models.gemini import GeminiModel
    from pydantic_ai.providers.google import GoogleProvider

    client = genai.Client(http_options={'timeout': timeout})
    return GeminiModel('gemini-1.5-pro', provider=GoogleProvider(client=client))


def vertexai(timeout: int, _tmp_path: Path) -> Model:
    from google import genai
    from google.oauth2.service_account import Credentials

    from pydantic_ai.models.gemini import GeminiModel
    from pydantic_ai.providers.google import GoogleProvider

    service_account_content = os.environ['GOOGLE_SERVICE_ACCOUNT_CONTENT']
    service_account_path = _tmp_path / 'service_account.json'
    service_account_path.write_text(service_account_content)

    credentials = Credentials.from_service_account_file(  # type: ignore[reportUnknownMemberType]
        service_account_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform'],
    )
    client = genai.Client(credentials=credentials, http_options={'timeout': timeout})
    return GeminiModel('gemini-1.5-flash', provider=GoogleProvider(client=client))


def groq(http_client: httpx.AsyncClient, _tmp_path: Path) -> Model:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

    return GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(http_client=http_client))


def anthropic(http_client: httpx.AsyncClient, _tmp_path: Path) -> Model:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    return AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(http_client=http_client))


def ollama(http_client: httpx.AsyncClient, _tmp_path: Path) -> Model:
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIModel(
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


@pytest.fixture(scope='session')
def timeout() -> int:
    return 30


@pytest.fixture
async def http_client(timeout: int) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        yield client


@pytest.fixture
def live_model(request: pytest.FixtureRequest, http_client: httpx.AsyncClient, timeout: int, tmp_path: Path) -> Model:
    if request.param == 'openai':
        return openai(http_client, tmp_path)
    elif request.param == 'gemini':
        return gemini(timeout, tmp_path)
    elif request.param == 'vertexai':
        return vertexai(timeout, tmp_path)
    elif request.param == 'groq':
        return groq(http_client, tmp_path)
    elif request.param == 'anthropic':
        return anthropic(http_client, tmp_path)
    elif request.param == 'ollama':
        return ollama(http_client, tmp_path)
    elif request.param == 'mistral':
        return mistral(http_client, tmp_path)
    elif request.param == 'cohere':
        return cohere(http_client, tmp_path)
    else:  # pragma: no cover
        raise ValueError(f'Unknown model: {request.param}')


@pytest.mark.parametrize(
    'model', ['openai', 'gemini', 'vertexai', 'groq', 'anthropic', 'ollama', 'mistral', 'cohere'], indirect=True
)
async def test_text(model: Model):
    agent = Agent(model, model_settings={'temperature': 0.0}, retries=2)
    result = await agent.run('What is the capital of France?')
    print('Text response:', result.output)
    assert 'paris' in result.output.lower()
    print('Text usage:', result.usage())
    usage = result.usage()
    assert usage.total_tokens is not None and usage.total_tokens > 0


@pytest.mark.parametrize(
    'model', ['openai', 'gemini', 'vertexai', 'groq', 'anthropic', 'ollama', 'mistral'], indirect=True
)
async def test_stream(model: Model):
    agent = Agent(model, model_settings={'temperature': 0.0}, retries=2)
    async with agent.run_stream('What is the capital of France?') as result:
        data = await result.get_output()
    print('Stream response:', data)
    assert 'paris' in data.lower()
    print('Stream usage:', result.usage())
    usage = result.usage()
    if model.__name__ != 'ollama':
        assert usage.total_tokens is not None and usage.total_tokens > 0


class MyModel(BaseModel):
    city: str


@pytest.mark.parametrize(
    'model', ['openai', 'gemini', 'vertexai', 'groq', 'anthropic', 'mistral', 'cohere'], indirect=True
)
async def test_structured(model: Model):
    agent = Agent(model, output_type=MyModel, model_settings={'temperature': 0.0}, retries=2)

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
