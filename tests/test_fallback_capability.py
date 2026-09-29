"""Smoke tests for the `Fallback` capability.

Not the final suite: see the PR description for the coverage still owed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, RunContext, TextPart, UserPromptPart
from pydantic_ai._fallback import continuation_pin
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Fallback, Hooks, SelectModel
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError, RetryModelRequest
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from .model_lifecycle_utils import LifecycleTrackingModel


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


def _suspended_history(pinned_model_id: str) -> list[ModelMessage]:
    """History ending in a response a provider suspended, pinned to the fallback candidate that started it."""
    return [
        ModelRequest(parts=[UserPromptPart('x')]),
        ModelResponse(
            parts=[TextPart('partial')],
            state='suspended',
            provider_response_id='job-1',
            metadata={'__pydantic_ai__': {'fallback_model_id': pinned_model_id}},
        ),
    ]


def _ends_suspended(messages: list[ModelMessage]) -> bool:
    return isinstance(messages[-1], ModelResponse) and messages[-1].state == 'suspended'


@pytest.mark.anyio
async def test_resumed_continuation_goes_to_the_pinned_model():
    """Resuming a suspended response continues it on the candidate that started it, not the chain's first model."""

    def step_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('the continuation must not go to the step model')  # pragma: no cover

    continued: list[bool] = []

    def pinned_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        continued.append(_ends_suspended(messages))
        return ModelResponse(parts=[TextPart('done')])

    pinned = FunctionModel(pinned_model, model_name='pinned')
    agent = Agent(FunctionModel(step_model, model_name='step'), capabilities=[Fallback(pinned)])
    result = await agent.run(message_history=_suspended_history(pinned.model_id))
    assert continued == [True]
    assert result.usage.requests == 1


@pytest.mark.anyio
async def test_failed_pinned_continuation_rewinds_to_the_step_model():
    """A pinned continuation that fails is dropped, and the turn is generated afresh from the chain's start."""
    step_saw: list[bool] = []

    def step_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step_saw.append(_ends_suspended(messages))
        return ModelResponse(parts=[TextPart('fresh')])

    pinned = FunctionModel(failure, model_name='pinned')
    agent = Agent(FunctionModel(step_model, model_name='step'), capabilities=[Fallback(pinned)])
    result = await agent.run(message_history=_suspended_history(pinned.model_id))
    assert result.output == 'fresh'
    assert step_saw == [False]
    assert not any(isinstance(m, ModelResponse) and m.state == 'suspended' for m in result.all_messages())


@pytest.mark.anyio
async def test_streamed_resumed_continuation_goes_to_the_pinned_model():
    async def step_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise AssertionError('the continuation must not go to the step model')  # pragma: no cover
        yield ''  # pragma: no cover

    continued: list[bool] = []

    async def pinned_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        continued.append(_ends_suspended(messages))
        yield 'done'

    pinned = FunctionModel(stream_function=pinned_stream, model_name='pinned')
    agent = Agent(FunctionModel(stream_function=step_stream, model_name='step'), capabilities=[Fallback(pinned)])
    async with agent.run_stream(message_history=_suspended_history(pinned.model_id)) as stream:
        await stream.get_output()
    assert continued == [True]


@pytest.mark.anyio
async def test_streamed_failed_pinned_continuation_rewinds_to_the_step_model():
    step_saw: list[bool] = []

    async def step_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        step_saw.append(_ends_suspended(messages))
        yield 'fresh'

    pinned = FunctionModel(stream_function=failure_stream, model_name='pinned')
    agent = Agent(FunctionModel(stream_function=step_stream, model_name='step'), capabilities=[Fallback(pinned)])
    async with agent.run_stream(message_history=_suspended_history(pinned.model_id)) as stream:
        assert await stream.get_output() == 'fresh'
    assert step_saw == [False]


@pytest.mark.anyio
async def test_suspended_response_is_pinned_to_the_model_that_served_it():
    fallback = Fallback[None](TestModel())
    served = FunctionModel(success, model_name='served')
    ctx = RunContext[None](deps=None, model=served, usage=RunUsage())
    request_context = ModelRequestContext(
        model=served, messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
    )
    suspended = await fallback.after_model_request(
        ctx, request_context=request_context, response=ModelResponse(parts=[], state='suspended')
    )
    assert continuation_pin(suspended) == served.model_id
    complete = await fallback.after_model_request(
        ctx, request_context=request_context, response=ModelResponse(parts=[])
    )
    assert continuation_pin(complete) is None


@pytest.mark.anyio
async def test_select_model_outranks_the_fallback_default():
    """With no agent model, `Fallback`'s first candidate only stands in when nothing else selects a model."""
    selected = FunctionModel(success, model_name='selected')
    agent = Agent(
        capabilities=[
            SelectModel(lambda ctx: selected),
            Fallback(FunctionModel(rejected, model_name='default')),
        ]
    )
    assert (await agent.run('x')).output == 'hello'


@pytest.mark.anyio
async def test_a_model_another_capability_retried_is_not_picked_again():
    calls: list[str] = []

    def failing(name: str) -> FunctionModel:
        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(name)
            raise ModelAPIError(model_name=name, message='boom')

        return FunctionModel(fn, model_name=name)

    retried = failing('retried')

    class RetryOnce(AbstractCapability[None]):
        """Moves the first failed attempt to `retried`, then leaves errors to `Fallback`."""

        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='innermost', wrapped_by=(Fallback,))

        async def on_model_request_error(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            if request_context.attempt == 1:
                raise RetryModelRequest(retried)
            raise error

    agent = Agent(
        failing('step'),
        deps_type=type(None),
        capabilities=[Fallback(retried, FunctionModel(success)), RetryOnce()],
    )
    assert (await agent.run('x')).output == 'hello'
    assert calls == ['step', 'retried']


@pytest.mark.anyio
async def test_prepare_model_request_can_move_to_another_model_before_sending():
    other = FunctionModel(success, model_name='other')

    class Redirect(AbstractCapability[None]):
        async def prepare_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            if request_context.attempt == 1:
                raise RetryModelRequest(other)
            return request_context

    agent = Agent(FunctionModel(rejected), deps_type=type(None), capabilities=[Redirect()])
    result = await agent.run('x')
    assert result.output == 'hello'
    assert result.usage.requests == 1


@pytest.mark.anyio
async def test_fallback_candidates_are_entered_once_and_exited_with_the_run():
    """A candidate is entered when first attempted, once however many steps attempt it, and exited when the run ends."""
    events: list[str] = []
    candidate = LifecycleTrackingModel(events, include_exit_exception=False)
    agent = Agent(FunctionModel(failure), capabilities=[Fallback(candidate)])

    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    result = await agent.run('x')
    assert result.output
    assert events.count('enter') == 1
    assert events[-1] == 'exit'
    assert events.count('request') == len([m for m in result.all_messages() if isinstance(m, ModelResponse)])
