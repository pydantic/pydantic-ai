"""The per-attempt model request loop: `prepare_model_request`, `RetryModelRequest` and `ModelRequestContext.attempt`.

The `ModelFallback` capability built on these is tested in `test_capability_model_fallback.py`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from pydantic_ai import Agent, ModelMessage, ModelResponse, RunContext, RunUsage, TextPart, capture_run_messages
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    Hooks,
    ModelFallback,
    WrapModelRequestHandler,
    WrapperCapability,
)
from pydantic_ai.exceptions import ModelAPIError, ModelRetry, RetryModelRequest, UserError
from pydantic_ai.messages import AgentStreamEvent, ModelRequest, ModelRequestAttempt, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCalls, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from .continuation_utils import ScriptedContinuationModel, StreamSegment, scripted_response


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
    """An outer `on_model_request_error` is never called with an inner hook's `RetryModelRequest`."""
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
            raise RetryModelRequest()

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


async def test_streamed_failure_before_the_first_event_reaches_the_error_hook():
    """A provider that accepts a streamed request and then fails before any output is handled like an open failure."""
    hooks = Hooks[None]()

    @hooks.on.model_request_error
    async def recover(
        ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        return ModelResponse(parts=[TextPart(f'recovered from {error}')])

    async def fails_before_first_event(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        yield {}  # produces no event, like a provider's `response.created`
        raise ModelAPIError(model_name='m', message='overloaded')

    agent = Agent(FunctionModel(stream_function=fails_before_first_event), deps_type=type(None), capabilities=[hooks])
    async with agent.run_stream('x') as stream:
        assert await stream.get_output() == 'recovered from overloaded'


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
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx):
                    pass
                break
    assert closed == [True]


async def test_a_primed_stream_the_consumer_never_iterates_keeps_what_it_produced():
    """The response committed for a primed stream the consumer left without iterating has what priming received.

    Priming opened the stream and read its first event, so the provider billed the request and the response
    has its id, as when the consumer iterates.
    """
    hooks = Hooks[None]()

    @hooks.on.model_request_error
    async def passthrough(
        ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        raise error  # pragma: no cover

    model = ScriptedContinuationModel(
        segments=[
            StreamSegment(
                texts=['hello'], state='complete', provider_response_id='resp-1', input_tokens=7, output_tokens=3
            )
        ]
    )
    agent = Agent(model, deps_type=type(None), capabilities=[hooks])
    usage = RunUsage()
    with capture_run_messages() as messages:
        async with agent.iter('x', usage=usage) as run:
            async for node in run:  # pragma: no branch
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx):
                        pass
                    break
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert (response.provider_response_id, response.usage) == ('resp-1', RequestUsage(input_tokens=7, output_tokens=3))
    assert usage == RunUsage(input_tokens=7, output_tokens=3, requests=1)


@pytest.mark.parametrize('iterate', [False, True])
async def test_a_resumed_primed_stream_the_consumer_never_iterates_keeps_the_resumed_response(iterate: bool):
    """A primed stream resuming a suspended response from history keeps that response's parts when left un-iterated.

    And counts only what the continuation billed beyond it, which the run that suspended it already counted: the same
    as when the consumer iterates the stream.
    """
    hooks = Hooks[None]()

    @hooks.on.model_request_error
    async def passthrough(
        ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        raise error  # pragma: no cover

    model = ScriptedContinuationModel(
        segments=[
            StreamSegment(
                texts=['is 42.'], state='complete', provider_response_id='job-2', input_tokens=3, output_tokens=4
            )
        ]
    )
    agent = Agent(model, deps_type=type(None), capabilities=[hooks])
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('go')]),
        scripted_response(
            texts=['The answer '], state='suspended', provider_response_id='job-1', input_tokens=5, output_tokens=2
        ),
    ]
    usage = RunUsage()
    with capture_run_messages() as messages:
        async with agent.iter(message_history=history, usage=usage) as run:
            async for node in run:  # pragma: no branch
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        if iterate:
                            async for _ in stream:
                                pass
                    break
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    # The continuation merged onto the response it resumes: unread, priming has only seen its part start.
    assert (
        [part.content for part in response.parts if isinstance(part, TextPart)],
        response.provider_response_id,
        response.state,
        response.usage,
    ) == (
        ['The answer ', 'is 42.' if iterate else ''],
        'job-2',
        'complete' if iterate else 'incomplete',
        RequestUsage(input_tokens=8, output_tokens=6),
    )
    # The resumed response's usage was counted by the run that suspended it.
    assert usage == RunUsage(input_tokens=3, output_tokens=4, requests=1)


async def test_open_failure_is_raised_when_the_consumer_never_iterates_the_stream():
    hooks = Hooks[None]()

    @hooks.on.model_request_error
    async def passthrough(
        ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        raise error

    agent = Agent(FunctionModel(stream_function=failure_stream), deps_type=type(None), capabilities=[hooks])
    with pytest.raises(ModelAPIError, match='boom'):
        async with agent.iter('x') as run:
            # The failed stream ends the run.
            async for node in run:  # pragma: no branch
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx):
                        pass


class _HandlesSomeErrors(AbstractCapability[None]):
    """Recovers from a failed request only for the model it handles, like an always-present provider capability."""

    def __init__(self, handled_model: str) -> None:
        self.handled_model = handled_model
        self.recovered: list[str] = []

    def _handles_model_request_error(self, ctx: RunContext[None], request_context: ModelRequestContext) -> bool:
        return request_context.model.model_name == self.handled_model

    async def on_model_request_error(
        self, ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        self.recovered.append(request_context.model.model_name)
        return ModelResponse(parts=[TextPart('recovered')])


def _error_capability(capability: _HandlesSomeErrors, kind: str) -> AbstractCapability[None]:
    return WrapperCapability(capability) if kind == 'wrapped' else capability


@pytest.mark.parametrize('kind', ['direct', 'wrapped'])
@pytest.mark.parametrize('handles', [True, False])
async def test_a_stream_is_only_primed_for_a_request_an_error_hook_handles(kind: str, handles: bool):
    """A stream is opened before it's handed over only when a capability handles a failure of that request."""
    opened: list[str] = []

    async def fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        opened.append('opened')
        yield 'hello'

    capability = _HandlesSomeErrors('streamer' if handles else 'other')
    agent = Agent(
        FunctionModel(stream_function=fn, model_name='streamer'),
        deps_type=type(None),
        capabilities=[_error_capability(capability, kind)],
    )
    async with agent.iter('x') as run:
        async for node in run:  # pragma: no branch
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx):
                    pass
                break
    # Primed, it was opened before the consumer could iterate it; lazily, it never was.
    assert opened == (['opened'] if handles else [])


@pytest.mark.parametrize('kind', ['direct', 'wrapped'])
@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('handles', [True, False])
async def test_an_error_hook_is_only_called_for_a_request_it_handles(kind: str, streaming: bool, handles: bool):
    """`on_model_request_error` is only called for a request its capability says it handles."""

    def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelAPIError(model_name='failing', message='boom')

    capability = _HandlesSomeErrors('failing' if handles else 'other')
    agent = Agent(
        FunctionModel(fail, stream_function=failure_stream, model_name='failing'),
        deps_type=type(None),
        capabilities=[_error_capability(capability, kind)],
    )
    if handles:
        if streaming:
            async with agent.run_stream('x') as result:
                assert await result.get_output() == 'recovered'
        else:
            assert (await agent.run('x')).output == 'recovered'
        assert capability.recovered == ['failing']
    else:

        async def consume(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass

        with pytest.raises(ModelAPIError, match='boom'):
            # An event stream handler makes the run stream its request.
            await agent.run('x', event_stream_handler=consume if streaming else None)
        assert capability.recovered == []


@pytest.mark.parametrize('streaming', [False, True])
async def test_only_the_capabilities_that_handle_a_request_error_are_called(streaming: bool):
    """Of several error hooks, the one whose capability doesn't handle the request is skipped for the next one."""

    def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelAPIError(model_name='failing', message='boom')

    handling = _HandlesSomeErrors('failing')
    other = _HandlesSomeErrors('other')
    agent = Agent(
        FunctionModel(fail, stream_function=failure_stream, model_name='failing'),
        deps_type=type(None),
        # The innermost capability's hook is asked first.
        capabilities=[handling, other],
    )
    if streaming:
        async with agent.run_stream('x') as result:
            assert await result.get_output() == 'recovered'
    else:
        assert (await agent.run('x')).output == 'recovered'
    assert (handling.recovered, other.recovered) == (['failing'], [])


async def test_a_wrapper_with_its_own_error_hook_handles_every_request_error():
    """A wrapper that overrides `on_model_request_error` itself answers for every request, whatever it wraps."""

    class RecoveringWrapper(WrapperCapability[None]):
        async def on_model_request_error(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart('recovered by the wrapper')])

    agent = Agent(
        FunctionModel(stream_function=failure_stream, model_name='failing'),
        deps_type=type(None),
        capabilities=[RecoveringWrapper(_HandlesSomeErrors('other'))],
    )
    async with agent.run_stream('x') as result:
        assert await result.get_output() == 'recovered by the wrapper'


async def test_hooks_handle_every_request_error():
    """A `Hooks` error handler answers for every request, as it did before requests could be told apart."""
    hooks = Hooks[None]()

    @hooks.on.model_request_error
    async def recover(
        ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        return ModelResponse(parts=[TextPart('recovered')])

    agent = Agent(
        FunctionModel(stream_function=failure_stream, model_name='any'), deps_type=type(None), capabilities=[hooks]
    )
    async with agent.run_stream('x') as result:
        assert await result.get_output() == 'recovered'


class RetryFromBeforeModelRequest(AbstractCapability[None]):
    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        raise RetryModelRequest()


class RetryFromWrapModelRequest(AbstractCapability[None]):
    async def wrap_model_request(
        self, ctx: RunContext[None], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        raise RetryModelRequest()


@pytest.mark.parametrize('capability', [RetryFromBeforeModelRequest(), RetryFromWrapModelRequest()])
@pytest.mark.parametrize('streaming', [False, True])
async def test_retry_model_request_before_any_attempt_is_a_user_error(
    capability: AbstractCapability[None], streaming: bool
):
    """`before_model_request` and `wrap_model_request` run once per step, before there is an attempt to retry."""
    agent = Agent(FunctionModel(success), deps_type=type(None), capabilities=[capability])
    with pytest.raises(UserError, match='can only be raised from `prepare_model_request`'):
        if streaming:
            async with agent.run_stream('x') as stream:
                await stream.get_output()  # pragma: no cover
        else:
            await agent.run('x')


class RetryThenRecover(AbstractCapability[None]):
    """Moves on from the first failed attempt, then recovers the second with a response of its own."""

    async def on_model_request_error(
        self, ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        if request_context.attempt == 1:
            raise RetryModelRequest()
        return ModelResponse(parts=[TextPart('recovered')])


def _attempt_outcomes(response: object) -> list[tuple[str, str | None]]:
    assert isinstance(response, ModelResponse)
    return [(attempt.outcome, attempt.error) for attempt in response.failed_attempts or []]


async def test_a_recovered_response_keeps_the_attempts_that_failed_before_it():
    agent = Agent(FunctionModel(failure), deps_type=type(None), capabilities=[RetryThenRecover()])
    result = await agent.run('x')
    assert result.output == 'recovered'
    assert _attempt_outcomes(result.all_messages()[-1]) == [('error', 'ModelAPIError: boom')]
    # The recovered response was never billed, and the failed attempt reported no usage.
    assert (result.usage.requests, result.usage.input_tokens, result.usage.output_tokens) == (1, 0, 0)


async def test_a_recovered_stream_keeps_the_attempts_that_failed_before_it():
    agent = Agent(
        FunctionModel(stream_function=failure_stream), deps_type=type(None), capabilities=[RetryThenRecover()]
    )
    async with agent.run_stream('x') as stream:
        assert await stream.get_output() == 'recovered'
    assert _attempt_outcomes(stream.all_messages()[-1]) == [('error', 'ModelAPIError: boom')]


async def test_in_place_parameter_edits_do_not_reach_the_next_attempt():
    """Each attempt starts from what `before_model_request` produced, even if an attempt edited its lists in place."""
    seen: list[tuple[int, list[str], bool]] = []

    class EditInPlaceThenSwitch(AbstractCapability[None]):
        async def prepare_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            parameters = request_context.model_request_parameters
            settings = request_context.model_settings or {}
            seen.append(
                (
                    request_context.attempt,
                    [tool.name for tool in parameters.function_tools],
                    'x-attempt' in settings.get('extra_headers', {}),
                )
            )
            if request_context.attempt == 1:
                parameters.function_tools.clear()
                parameters.output_tools.clear()
                parameters.revealed_tool_names.add('leaked')
                settings.setdefault('extra_headers', {})['x-attempt'] = '1'
                raise RetryModelRequest(FunctionModel(success, model_name='other'))
            return request_context

    agent = Agent(
        FunctionModel(failure),
        deps_type=type(None),
        model_settings={'extra_headers': {'x-base': 'yes'}},
        capabilities=[EditInPlaceThenSwitch()],
    )

    @agent.tool_plain
    def lookup() -> str:
        return 'found'  # pragma: no cover

    result = await agent.run('x')
    assert result.output == 'hello'
    assert seen == [(1, ['lookup'], False), (2, ['lookup'], False)]


async def test_a_recovered_response_does_not_count_a_rejected_attempts_usage_again():
    """The attempts a recovered response carries describe usage committed when they failed."""
    calls: list[int] = []

    def billed_then_failing(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            return ModelResponse(parts=[TextPart('nope')], usage=RequestUsage(input_tokens=100, output_tokens=10))
        raise ModelAPIError(model_name='m', message='boom')

    class RejectThenRecover(AbstractCapability[None]):
        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            if response.text == 'nope':
                raise RetryModelRequest()
            return response

        async def on_model_request_error(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart('recovered')])

    agent = Agent(FunctionModel(billed_then_failing), deps_type=type(None), capabilities=[RejectThenRecover()])
    result = await agent.run('x')
    assert result.output == 'recovered'
    assert _attempt_outcomes(result.all_messages()[-1]) == [('rejected', None)]
    assert (result.usage.requests, result.usage.input_tokens, result.usage.output_tokens) == (1, 100, 10)


async def test_nested_settings_edited_in_place_do_not_reach_the_next_attempt():
    """Containers nested in a setting are copied for each attempt too, such as a provider's `extra_body`."""
    seen: list[str] = []

    class EditNestedInPlaceThenRetry(AbstractCapability[None]):
        async def prepare_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            extra_body = cast(dict[str, Any], (request_context.model_settings or {}).get('extra_body'))
            seen.append(repr(extra_body))
            extra_body['nested']['x'] = 'edited'
            extra_body['tags'].add('edited')
            if request_context.attempt == 1:
                raise RetryModelRequest(FunctionModel(success, model_name='other'))
            return request_context

    agent = Agent(
        FunctionModel(failure),
        deps_type=type(None),
        model_settings={'extra_body': {'nested': {'x': 'original'}, 'tags': {'original'}}},
        capabilities=[EditNestedInPlaceThenRetry()],
    )
    assert (await agent.run('x')).output == 'hello'
    original = "{'nested': {'x': 'original'}, 'tags': {'original'}}"
    assert seen == [original, original]


async def test_tool_schema_edits_in_place_do_not_reach_the_next_attempt():
    """A tool definition a `prepare_model_request` hook edits in place, schema included, is fresh for the next attempt."""
    seen: list[tuple[int, str | None, list[str]]] = []

    class EditToolInPlaceThenSwitch(AbstractCapability[None]):
        async def prepare_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            tool = request_context.model_request_parameters.function_tools[0]
            properties = cast(dict[str, Any], tool.parameters_json_schema['properties'])
            seen.append((request_context.attempt, tool.description, sorted(properties)))
            if request_context.attempt == 1:
                tool.description = 'edited'
                properties['injected'] = {'type': 'string'}
                raise RetryModelRequest(FunctionModel(success, model_name='other'))
            return request_context

    agent = Agent(FunctionModel(failure), deps_type=type(None), capabilities=[EditToolInPlaceThenSwitch()])

    @agent.tool_plain
    def lookup(query: str) -> str:
        """Look something up."""
        return query  # pragma: no cover

    assert (await agent.run('x')).output == 'hello'
    assert seen == [(1, 'Look something up.', ['query']), (2, 'Look something up.', ['query'])]


async def test_failed_attempts_a_streamed_model_carries_are_counted():
    """A stream a model hands over with failed attempts of its own, billed by that model, counts their usage.

    Only the attempts the agent's own attempt loop moved on from were counted already.
    """
    attempt = ModelRequestAttempt(
        model_name='inner',
        provider_name='inner',
        outcome='rejected',
        timestamp=datetime.now(UTC),
        duration=timedelta(seconds=1),
        usage=RequestUsage(input_tokens=100, output_tokens=200),
    )

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'hello'

    class AttemptCarryingModel(FunctionModel):
        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any] | None = None,
        ) -> AsyncGenerator[StreamedResponse]:
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed_response:
                streamed_response.failed_attempts = [attempt]
                yield streamed_response

    async def consume(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in events:
            pass

    model = AttemptCarryingModel(stream_function=stream_fn)
    for capabilities in ([], [RetrySameModelOnError()]):
        agent = Agent(model, deps_type=type(None), capabilities=capabilities)
        result = await agent.run('x', event_stream_handler=consume)
        assert (result.usage.input_tokens, result.usage.output_tokens) == (150, 201)


async def test_a_streamed_fallback_model_under_a_plain_agent_keeps_its_attempts_and_usage():
    """A `FallbackModel` streaming under an agent without `ModelFallback` keeps its failed attempts and the answer's usage."""

    async def failing_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise ModelAPIError(model_name='primary', message='boom')
        yield ''  # pragma: no cover

    async def answer_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'hello'

    model = FallbackModel(
        FunctionModel(stream_function=failing_stream, model_name='primary'),
        FunctionModel(stream_function=answer_stream, model_name='answer'),
    )
    agent = Agent(model, deps_type=type(None))
    async with agent.run_stream('x') as result:
        assert await result.get_output() == 'hello'
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert [(attempt.model_name, attempt.outcome) for attempt in response.failed_attempts or []] == [
        ('primary', 'error')
    ]
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.requests) == (
        response.usage.input_tokens,
        response.usage.output_tokens,
        1,
    )


@pytest.mark.parametrize('error_capability', ['on_model_request_error', 'ModelFallback'])
async def test_a_wrapper_that_answers_itself_stops_the_stream_being_primed(error_capability: str):
    """A `wrap_model_request` that cancels its handler while the stream opens, and answers itself, ends the step.

    A hook that can act on a failure to open makes the stream open before it's handed over; that waits on the
    provider's first event, and must give up once nothing waits for the stream any more.
    """
    opening = asyncio.Event()

    async def never_answers(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        opening.set()
        await asyncio.Event().wait()
        yield ''  # pragma: no cover

    class AnswerWhileOpening(AbstractCapability[Any]):
        async def wrap_model_request(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
        ) -> ModelResponse:
            sending = asyncio.ensure_future(handler(request_context))
            await opening.wait()
            sending.cancel()
            with suppress(asyncio.CancelledError):
                await sending
            return ModelResponse(parts=[TextPart('blocked')])

    hooks = Hooks[Any]()

    @hooks.on.model_request_error
    async def reraise(ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception) -> ModelResponse:
        # Only its presence matters: the stream never fails.
        raise error  # pragma: no cover

    agent = Agent(
        FunctionModel(stream_function=never_answers),
        capabilities=[
            AnswerWhileOpening(),
            hooks if error_capability == 'on_model_request_error' else ModelFallback(FunctionModel(failure)),
        ],
    )

    async def run() -> str:
        async with agent.run_stream('x') as result:
            return await result.get_output()

    assert await asyncio.wait_for(run(), timeout=10) == 'blocked'


class MarkHistoryPerAttempt(AbstractCapability[None]):
    """Records each attempt on the latest request through `ctx.messages`, then asks for a second attempt."""

    async def prepare_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        index = max(i for i, message in enumerate(ctx.messages) if isinstance(message, ModelRequest))
        request = ctx.messages[index]
        assert isinstance(request, ModelRequest)
        marks = (request.metadata or {}).get('attempts', [])
        ctx.messages[index] = replace(request, metadata={'attempts': [*marks, request_context.attempt]})
        if request_context.attempt == 1:
            raise RetryModelRequest()
        return request_context


@pytest.mark.parametrize('mode', ['run', 'run_stream', 'iter'])
async def test_history_changes_in_prepare_model_request_persist_across_attempts(mode: str):
    """Changes to `ctx.messages` in `prepare_model_request` reach the run's history and survive a further attempt.

    Changes to `request_context` are for one attempt; the run's history is not rolled back between attempts.
    """

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'hello'

    agent = Agent(
        FunctionModel(success, stream_function=stream),
        deps_type=type(None),
        capabilities=[MarkHistoryPerAttempt()],
    )
    if mode == 'run':
        messages = (await agent.run('x')).all_messages()
    elif mode == 'run_stream':
        async with agent.run_stream('x') as streamed:
            assert await streamed.get_output() == 'hello'
        messages = streamed.all_messages()
    else:
        async with agent.iter('x') as run:
            async for _ in run:
                pass
        assert run.result is not None
        messages = run.result.all_messages()
    request = messages[0]
    assert isinstance(request, ModelRequest)
    assert request.metadata == {'attempts': [1, 2]}
