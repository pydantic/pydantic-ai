from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel

pytestmark = [pytest.mark.anyio, pytest.mark.benchmark]


@pytest.fixture
def blockbuster_enabled() -> bool:
    return False


@pytest.fixture
async def agent() -> Agent[None, str]:
    agent = Agent(TestModel(custom_output_text='ok'))
    await agent.run('hello')
    return agent


async def test_agent_run_without_capabilities(agent: Agent[None, str]) -> None:
    result = await agent.run('hello')
    assert result.output == 'ok'
    assert result.usage.requests == 1


@pytest.fixture(params=[1000, 5000], ids=['1000-fragments', '5000-fragments'])
async def synthetic_history(agent: Agent[None, str], request: pytest.FixtureRequest) -> list[ModelMessage]:
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('Earlier question')])]
    history.extend(ModelResponse(parts=[TextPart('A response fragment')]) for _ in range(request.param))
    await agent.run('hello', message_history=history)
    return history


async def test_agent_run_with_synthetic_history(agent: Agent[None, str], synthetic_history: list[ModelMessage]) -> None:
    result = await agent.run('hello', message_history=synthetic_history)
    assert result.output == 'ok'
    assert len(result.all_messages()[1].parts) == len(synthetic_history) - 1
