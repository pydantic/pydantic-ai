from __future__ import annotations

import pytest

from pydantic_ai import Agent
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
