"""The per-attempt model request loop: `prepare_model_request`, `RetryModelRequest` and `ModelRequestContext.attempt`.

The `Fallback` capability built on these is tested in `test_capability_fallback.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace

import pytest

from pydantic_ai import Agent, ModelMessage, ModelResponse, RunContext, TextPart
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Hooks
from pydantic_ai.exceptions import ModelAPIError, ModelRetry, RetryModelRequest, UserError
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytestmark = pytest.mark.anyio


def success(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('hello')])


def failure(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelAPIError(model_name='m', message='boom')


def fails_first(calls: list[int]) -> FunctionModel:
    """A model whose first call fails and whose later calls succeed, recording each call."""

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise ModelAPIError(model_name='flaky', message='boom')
        return ModelResponse(parts=[TextPart('hello')])

    return FunctionModel(fn, model_name='flaky')


class RetrySameModelOnError(AbstractCapability[None]):
    """Re-attempts the same model once when an attempt raises."""

    async def on_model_request_error(
        self, ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        if request_context.attempt == 1:
            raise RetryModelRequest()
        raise error  # pragma: no cover


async def test_prepare_model_request_runs_per_attempt_and_before_model_request_once():
    before: list[int] = []
    prepared: list[tuple[int, str]] = []

    async def record_before(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        before.append(request_context.attempt)
        return request_context

    async def record_prepare(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        prepared.append((request_context.attempt, request_context.model.model_name))
        return request_context

    calls: list[int] = []
    agent = Agent(
        fails_first(calls),
        deps_type=type(None),
        capabilities=[
            Hooks[None](before_model_request=record_before, prepare_model_request=record_prepare),
            RetrySameModelOnError(),
        ],
    )
    result = await agent.run('x')
    assert result.output == 'hello'
    assert before == [1]
    assert prepared == [(1, 'flaky'), (2, 'flaky')]
    assert calls == [1, 2]
    assert result.usage.requests == 1


async def test_prepare_model_request_decorator_registration():
    prepared: list[int] = []
    hooks = Hooks[None]()

    @hooks.on.prepare_model_request(timeout=5)
    async def record(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        prepared.append(request_context.attempt)
        return request_context

    agent = Agent(FunctionModel(success), deps_type=type(None), capabilities=[hooks])
    assert (await agent.run('x')).output == 'hello'
    assert prepared == [1]


async def test_retry_same_model_from_after_model_request():
    responses = iter(['nope', 'hello'])

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(next(responses))])

    class RejectFirst(AbstractCapability[None]):
        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            if request_context.attempt == 1:
                raise RetryModelRequest()
            return response

    agent = Agent(FunctionModel(fn), deps_type=type(None), capabilities=[RejectFirst()])
    result = await agent.run('x')
    assert result.output == 'hello'
    # The rejected response never reaches history, though its usage is still counted.
    assert [m.parts[0] for m in result.all_messages() if isinstance(m, ModelResponse)] == [TextPart('hello')]
    assert result.usage.requests == 1


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

    calls: list[int] = []
    agent = Agent(
        fails_first(calls),
        deps_type=type(None),
        capabilities=[CopyingHook(), RecordingHook(), RetrySameModelOnError()],
    )
    assert (await agent.run('x')).output == 'hello'
    assert seen == [1, 2]


async def test_prepare_model_request_can_move_to_another_model_before_sending():
    other = FunctionModel(success, model_name='other')
    sent: list[str] = []

    def never(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append('never')  # pragma: no cover
        return ModelResponse(parts=[TextPart('nope')])  # pragma: no cover

    class Redirect(AbstractCapability[None]):
        async def prepare_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            if request_context.attempt == 1:
                raise RetryModelRequest(other)
            return request_context

    agent = Agent(FunctionModel(never), deps_type=type(None), capabilities=[Redirect()])
    result = await agent.run('x')
    assert result.output == 'hello'
    assert sent == []
    assert result.usage.requests == 1


async def test_streamed_prepare_model_request_can_move_to_another_model_before_sending():
    opened: list[str] = []

    async def other_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'hello'

    async def never_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        opened.append('never')  # pragma: no cover
        yield 'nope'  # pragma: no cover

    other = FunctionModel(stream_function=other_stream, model_name='other')

    class Redirect(AbstractCapability[None]):
        async def prepare_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            if request_context.attempt == 1:
                raise RetryModelRequest(other)
            return request_context

    agent = Agent(FunctionModel(stream_function=never_stream), deps_type=type(None), capabilities=[Redirect()])
    async with agent.run_stream('x') as stream:
        assert await stream.get_output() == 'hello'
    assert opened == []


async def test_attempt_ceiling():
    class RetryForever(AbstractCapability[None]):
        async def on_model_request_error(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            raise RetryModelRequest()

    agent = Agent(FunctionModel(failure), deps_type=type(None), capabilities=[RetryForever()])
    with pytest.raises(UserError, match='attempted more than the maximum of 100 times'):
        await agent.run('x')


async def test_outer_error_hook_is_not_handed_control_flow():
    """An outer `on_model_request_error` is never called with an inner hook's `RetryModelRequest` or `ModelRetry`."""
    outer_errors: list[Exception] = []
    calls: list[int] = []

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(len(calls) + 1)
        if len(calls) <= 2:
            raise ModelAPIError(model_name='m', message='boom')
        return ModelResponse(parts=[TextPart('hello')])

    class Outer(AbstractCapability[None]):
        async def on_model_request_error(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            outer_errors.append(error)  # pragma: no cover
            raise error  # pragma: no cover

    class Inner(AbstractCapability[None]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='innermost')

        async def on_model_request_error(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            if request_context.attempt == 1:
                raise RetryModelRequest()
            raise ModelRetry('try again')

    agent = Agent(FunctionModel(fn), deps_type=type(None), capabilities=[Outer(), Inner()])
    result = await agent.run('x')
    assert result.output == 'hello'
    assert outer_errors == []
    assert calls == [1, 2, 3]


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


async def failure_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    raise ModelAPIError(model_name='m', message='boom')
    yield ''  # pragma: no cover


async def test_streamed_open_failure_recovered_by_error_hook_is_replayed():
    hooks = Hooks[None]()

    @hooks.on.model_request_error
    async def recover(
        ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        return ModelResponse(parts=[TextPart('recovered')])

    agent = Agent(FunctionModel(stream_function=failure_stream), deps_type=type(None), capabilities=[hooks])
    async with agent.run_stream('x') as stream:
        assert await stream.get_output() == 'recovered'


async def test_streamed_open_failure_raised_as_model_retry_retries_the_step():
    opened: list[int] = []

    async def fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        opened.append(len(opened) + 1)
        if len(opened) == 1:
            raise ModelRetry('try again')
        yield 'hello'

    hooks = Hooks[None]()

    @hooks.on.model_request_error
    async def passthrough(
        ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        raise error  # pragma: no cover

    agent = Agent(FunctionModel(stream_function=fn), deps_type=type(None), capabilities=[hooks])
    async with agent.run_stream('x') as stream:
        assert await stream.get_output() == 'hello'
    assert opened == [1, 2]


async def test_streamed_request_cannot_be_retried_after_it_was_streamed():
    async def fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'hello'

    class RejectAfterStreaming(AbstractCapability[None]):
        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            raise RetryModelRequest()

    agent = Agent(FunctionModel(stream_function=fn), deps_type=type(None), capabilities=[RejectAfterStreaming()])
    with pytest.raises(UserError, match='cannot be raised from `after_model_request` on a streamed request'):
        async with agent.run_stream('x') as stream:
            await stream.get_output()


async def test_streamed_before_model_request_error_is_raised_when_entering_the_stream():
    """Only a failure to *open* the stream is deferred to iteration; a hook error is raised as before."""

    async def fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'hello'  # pragma: no cover

    hooks = Hooks[None]()

    @hooks.on.before_model_request
    async def broken(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        raise ValueError('broken hook')

    @hooks.on.model_request_error
    async def passthrough(
        ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        raise error  # pragma: no cover

    agent = Agent(FunctionModel(stream_function=fn), deps_type=type(None), capabilities=[hooks])
    with pytest.raises(ValueError, match='broken hook'):
        async with agent.run_stream('x'):
            pass  # pragma: no cover


async def test_primed_stream_the_consumer_never_iterates_is_closed():
    closed: list[bool] = []

    async def fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        try:
            yield 'hello'
        finally:
            closed.append(True)

    hooks = Hooks[None]()

    @hooks.on.model_request_error
    async def passthrough(
        ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        raise error  # pragma: no cover

    agent = Agent(FunctionModel(stream_function=fn), deps_type=type(None), capabilities=[hooks])
    async with agent.iter('x') as run:
        node = await run.next(run.next_node)
        assert Agent.is_model_request_node(node)
        async with node.stream(run.ctx):
            pass
    assert closed == [True]
