"""Smoke tests for the `Fallback` capability.

Not the final suite: see the PR description for the coverage still owed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace

import pytest

from pydantic_ai import Agent, ModelMessage, ModelResponse, RunContext, TextPart
from pydantic_ai.capabilities import AbstractCapability, Fallback, Hooks
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError, RetryModelRequest
from pydantic_ai.models import ModelRequestContext
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


@pytest.mark.anyio
async def test_attempt_survives_a_replaced_request_context():
    """A `prepare_model_request` hook that returns a copy doesn't reset `attempt` for the hooks after it."""
    seen: list[int] = []

    class CopyingHook(AbstractCapability[None]):
        async def prepare_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            return replace(request_context)

    class RecordingHook(AbstractCapability[None]):
        async def prepare_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            seen.append(request_context.attempt)
            return request_context

    agent = Agent(
        FunctionModel(failure),
        deps_type=type(None),
        capabilities=[CopyingHook(), RecordingHook(), Fallback(FunctionModel(success))],
    )
    assert (await agent.run('x')).output == 'hello'
    assert seen == [1, 2]


@pytest.mark.anyio
async def test_hooks_error_chain_passes_retry_model_request_through():
    """A later `Hooks` error callback must not be handed another callback's `RetryModelRequest` as its error."""
    hooks = Hooks[None]()
    later_errors: list[Exception] = []
    fallback_model = FunctionModel(success)

    @hooks.on.model_request_error
    async def retry(ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception) -> ModelResponse:
        if request_context.attempt == 1:
            raise RetryModelRequest(fallback_model)
        raise error  # pragma: no cover

    @hooks.on.model_request_error
    async def later(ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception) -> ModelResponse:
        later_errors.append(error)  # pragma: no cover
        raise error  # pragma: no cover

    agent = Agent(FunctionModel(failure), deps_type=type(None), capabilities=[hooks])
    assert (await agent.run('x')).output == 'hello'
    assert later_errors == []
