"""Tests of pydantic-ai actually connecting to OpenAI and Gemini models.

WARNING: running these tests will consume your OpenAI and Gemini credits.
"""

import os
from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel

pytestmark = [
    pytest.mark.skipif(os.getenv('PYDANTIC_AI_LIVE_TEST_DANGEROUS') != 'CHARGE-ME!', reason='live tests disabled'),
    pytest.mark.anyio,
]


@pytest.fixture
async def http_client(allow_model_requests: None) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=30) as client:
        yield client


async def test_openai(http_client: httpx.AsyncClient):
    agent = Agent(OpenAIModel('gpt-3.5-turbo', http_client=http_client))
    result = await agent.run('What is the capital of France?')
    print('OpenAI response:', result.data)
    assert 'paris' in result.data.lower()
    print('OpenAI cost:', result.cost())
    cost = result.cost()
    assert cost.total_tokens is not None and cost.total_tokens > 0


async def test_openai_stream(http_client: httpx.AsyncClient):
    agent = Agent(OpenAIModel('gpt-3.5-turbo', http_client=http_client))
    async with agent.run_stream('What is the capital of France?') as result:
        data = await result.get_data()
    print('OpenAI stream response:', data)
    assert 'paris' in data.lower()
    print('OpenAI stream cost:', result.cost())
    cost = result.cost()
    assert cost.total_tokens is not None and cost.total_tokens > 0


class MyModel(BaseModel):
    city: str


async def test_openai_structured(http_client: httpx.AsyncClient):
    agent = Agent(OpenAIModel('gpt-4o-mini', http_client=http_client), result_type=MyModel)
    result = await agent.run('What is the capital of the UK?')
    print('OpenAI structured response:', result.data)
    assert result.data.city.lower() == 'london'
    print('OpenAI structured cost:', result.cost())
    cost = result.cost()
    assert cost.total_tokens is not None and cost.total_tokens > 0


async def test_gemini(http_client: httpx.AsyncClient):
    agent = Agent(GeminiModel('gemini-1.5-flash', http_client=http_client))
    result = await agent.run('What is the capital of France?')
    print('Gemini response:', result.data)
    assert 'paris' in result.data.lower()
    print('Gemini cost:', result.cost())
    cost = result.cost()
    assert cost.total_tokens is not None and cost.total_tokens > 0


async def test_gemini_stream(http_client: httpx.AsyncClient):
    agent = Agent(GeminiModel('gemini-1.5-pro', http_client=http_client))
    async with agent.run_stream('What is the capital of France?') as result:
        data = await result.get_data()
    print('Gemini stream response:', data)
    assert 'paris' in data.lower()
    print('Gemini stream cost:', result.cost())
    cost = result.cost()
    assert cost.total_tokens is not None and cost.total_tokens > 0


async def test_gemini_structured(http_client: httpx.AsyncClient):
    agent = Agent(GeminiModel('gemini-1.5-pro', http_client=http_client), result_type=MyModel)
    result = await agent.run('What is the capital of the UK?')
    print('Gemini structured response:', result.data)
    assert result.data.city.lower() == 'london'
    print('Gemini structured cost:', result.cost())
    cost = result.cost()
    assert cost.total_tokens is not None and cost.total_tokens > 0
