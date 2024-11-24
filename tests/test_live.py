"""Tests of pydantic-ai actually connecting to OpenAI and Gemini models.

WARNING: running these tests will consume your OpenAI and Gemini credits.
"""

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Callable

import httpx
import pytest
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models import Model

pytestmark = [
    pytest.mark.skipif(os.getenv('PYDANTIC_AI_LIVE_TEST_DANGEROUS') != 'CHARGE-ME!', reason='live tests disabled'),
    pytest.mark.anyio,
]


def openai(http_client: httpx.AsyncClient, _tmp_path: Path) -> Model:
    from pydantic_ai.models.openai import OpenAIModel

    return OpenAIModel('gpt-4o-mini', http_client=http_client)


def gemini(http_client: httpx.AsyncClient, _tmp_path: Path) -> Model:
    from pydantic_ai.models.gemini import GeminiModel

    return GeminiModel('gemini-1.5-pro', http_client=http_client)


def vertexai(http_client: httpx.AsyncClient, tmp_path: Path) -> Model:
    from pydantic_ai.models.vertexai import VertexAIModel

    service_account_content = os.environ['GOOGLE_SERVICE_ACCOUNT_CONTENT']
    service_account_path = tmp_path / 'service_account.json'
    service_account_path.write_text(service_account_content)
    return VertexAIModel('gemini-1.5-flash', service_account_file=service_account_path, http_client=http_client)


def groq(http_client: httpx.AsyncClient, _tmp_path: Path) -> Model:
    from pydantic_ai.models.groq import GroqModel

    return GroqModel('llama-3.1-70b-versatile', http_client=http_client)


params = [
    pytest.param(openai, id='openai'),
    pytest.param(gemini, id='gemini'),
    pytest.param(vertexai, id='vertexai'),
    pytest.param(groq, id='groq'),
]
GetModel = Callable[[httpx.AsyncClient, Path], Model]


@pytest.fixture
async def http_client(allow_model_requests: None) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=30) as client:
        yield client


@pytest.mark.parametrize('get_model', params)
async def test_text(http_client: httpx.AsyncClient, tmp_path: Path, get_model: GetModel):
    agent = Agent(get_model(http_client, tmp_path))
    result = await agent.run('What is the capital of France?')
    print('Text response:', result.data)
    assert 'paris' in result.data.lower()
    print('Text cost:', result.cost())
    cost = result.cost()
    assert cost.total_tokens is not None and cost.total_tokens > 0


@pytest.mark.parametrize('get_model', params)
async def test_stream(http_client: httpx.AsyncClient, tmp_path: Path, get_model: GetModel):
    agent = Agent(get_model(http_client, tmp_path))
    async with agent.run_stream('What is the capital of France?') as result:
        data = await result.get_data()
    print('Stream response:', data)
    assert 'paris' in data.lower()
    print('Stream cost:', result.cost())
    cost = result.cost()
    assert cost.total_tokens is not None and cost.total_tokens > 0


class MyModel(BaseModel):
    city: str


@pytest.mark.parametrize('get_model', params)
async def test_structured(http_client: httpx.AsyncClient, tmp_path: Path, get_model: GetModel):
    agent = Agent(get_model(http_client, tmp_path), result_type=MyModel)
    result = await agent.run('What is the capital of the UK?')
    print('Structured response:', result.data)
    assert result.data.city.lower() == 'london'
    print('Structured cost:', result.cost())
    cost = result.cost()
    assert cost.total_tokens is not None and cost.total_tokens > 0
