"""Smoke tests for the `Fallback` capability.

Not the final suite: see the PR description for the coverage still owed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart
from pydantic_ai.capabilities import Fallback
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError
from pydantic_ai.models.function import AgentInfo, FunctionModel


def failure(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelAPIError(model_name='m', message='boom')


def success(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('hello')])


def rejected(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('nope')])


def reject_nope(response: ModelResponse) -> bool:
    return isinstance(response.parts[0], TextPart) and response.parts[0].content == 'nope'


@pytest.mark.anyio
async def test_agent_model_then_capability_model():
    agent = Agent(FunctionModel(failure), capabilities=[Fallback(FunctionModel(success))])
    result = await agent.run('x')
    assert result.output == 'hello'
    assert result.usage.requests == 1


@pytest.mark.anyio
async def test_no_agent_model_uses_first_candidate():
    agent = Agent(capabilities=[Fallback(FunctionModel(failure), FunctionModel(success))])
    result = await agent.run('x')
    assert result.output == 'hello'


@pytest.mark.anyio
async def test_chain_exhausted():
    agent = Agent(FunctionModel(failure), capabilities=[Fallback(FunctionModel(failure))])
    with pytest.raises(FallbackExceptionGroup) as exc_info:
        await agent.run('x')
    assert len(exc_info.value.exceptions) == 2


@pytest.mark.anyio
async def test_response_rejection():
    agent = Agent(
        FunctionModel(rejected),
        capabilities=[Fallback(FunctionModel(success), fallback_on=reject_nope)],
    )
    result = await agent.run('x')
    assert result.output == 'hello'


@pytest.mark.anyio
async def test_duplicate_model_skipped():
    m = FunctionModel(failure)
    ok = FunctionModel(success)
    agent = Agent(m, capabilities=[Fallback(m, ok)])
    result = await agent.run('x')
    assert result.output == 'hello'


async def failure_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    raise ModelAPIError(model_name='m', message='boom')
    yield ''  # pragma: no cover


async def success_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'hello'


@pytest.mark.anyio
async def test_streaming_open_failure_falls_back():
    agent = Agent(
        FunctionModel(stream_function=failure_stream),
        capabilities=[Fallback(FunctionModel(stream_function=success_stream))],
    )
    async with agent.run_stream('x') as stream:
        assert await stream.get_output() == 'hello'
