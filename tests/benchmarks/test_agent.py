from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import CompletedStreamedResponse, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
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


@pytest.fixture(params=[1024, 4096], ids=['1024-chunks', '4096-chunks'])
async def captured_text_stream(request: pytest.FixtureRequest) -> tuple[ModelResponse, list[ModelResponseStreamEvent]]:
    chunk = 'x' * (256 * 1024 // request.param)

    async def stream_text(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        for _ in range(request.param):
            yield chunk

    model = FunctionModel(stream_function=stream_text)
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('hello')])]
    async with model.request_stream(messages, None, ModelRequestParameters()) as stream:
        events = [event async for event in stream]
        return stream.get(), events


async def test_replay_text_stream(captured_text_stream: tuple[ModelResponse, list[ModelResponseStreamEvent]]) -> None:
    response, events = captured_text_stream
    stream = CompletedStreamedResponse(
        response, model_request_parameters=ModelRequestParameters(), replay_events=events
    )
    async for _ in stream:
        pass
    assert stream.get() == response
