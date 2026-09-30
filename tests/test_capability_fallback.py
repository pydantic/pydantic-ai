"""The `Fallback` capability.

The attempt loop it drives (`prepare_model_request`, `RetryModelRequest`) is tested in
`test_model_request_attempts.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, RunContext, TextPart, UserPromptPart
from pydantic_ai._fallback import FALLBACK_CAPABILITY_PIN_KEY, continuation_pin
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Fallback, SelectModel, WrapperCapability
from pydantic_ai.exceptions import (
    FallbackExceptionGroup,
    ModelAPIError,
    ModelRetry,
    RetryModelRequest,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

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


def _suspended_history(pinned_model_id: str) -> list[ModelMessage]:
    """History ending in a response a provider suspended, pinned to the fallback candidate that started it."""
    return [
        ModelRequest(parts=[UserPromptPart('x')]),
        ModelResponse(
            parts=[TextPart('partial')],
            state='suspended',
            provider_response_id='job-1',
            metadata={'__pydantic_ai__': {'fallback_candidate': pinned_model_id}},
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
    assert continuation_pin(suspended, key=FALLBACK_CAPABILITY_PIN_KEY) == served.model_id
    complete = await fallback.after_model_request(
        ctx, request_context=request_context, response=ModelResponse(parts=[])
    )
    assert continuation_pin(complete, key=FALLBACK_CAPABILITY_PIN_KEY) is None


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


@pytest.mark.anyio
async def test_a_pin_to_an_unknown_model_is_refused():
    agent = Agent(FunctionModel(success, model_name='step'), capabilities=[Fallback(FunctionModel(success))])
    with pytest.raises(UserError, match="started by 'function:gone'"):
        await agent.run(message_history=_suspended_history('function:gone'))


@pytest.mark.anyio
async def test_a_fallback_model_candidate_keeps_its_own_pin():
    """`Fallback`'s pin sits beside `FallbackModel`'s, so a `FallbackModel` candidate still resumes on its inner model."""
    fallback = Fallback[None](TestModel())
    inner = FunctionModel(success, model_name='inner')
    ctx = RunContext[None](deps=None, model=inner, usage=RunUsage())
    request_context = ModelRequestContext(
        model=inner, messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
    )
    response = ModelResponse(
        parts=[], state='suspended', metadata={'__pydantic_ai__': {'fallback_model_id': 'function:inner-of-inner'}}
    )
    stamped = await fallback.after_model_request(ctx, request_context=request_context, response=response)
    assert stamped.metadata == {
        '__pydantic_ai__': {'fallback_model_id': 'function:inner-of-inner', 'fallback_candidate': inner.model_id}
    }


def test_fallback_requires_a_model():
    with pytest.raises(UserError, match='requires at least one model'):
        Fallback()


def test_fallback_is_not_spec_serializable():
    assert Fallback.get_serialization_name() is None


@pytest.mark.anyio
async def test_an_error_fallback_on_does_not_match_propagates():
    later: list[str] = []

    def potato(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ValueError('not a model API error')

    def later_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        later.append('called')  # pragma: no cover
        return ModelResponse(parts=[TextPart('hello')])  # pragma: no cover

    agent = Agent(FunctionModel(potato), capabilities=[Fallback(FunctionModel(later_model))])
    with pytest.raises(ValueError, match='not a model API error'):
        await agent.run('x')
    assert later == []


@pytest.mark.anyio
async def test_async_predicates():
    async def on_exception(exc: Exception) -> bool:
        return isinstance(exc, ModelAPIError)

    async def on_response(response: ModelResponse) -> bool:
        return reject_nope(response)

    agent = Agent(
        FunctionModel(failure),
        capabilities=[
            Fallback(FunctionModel(rejected), FunctionModel(success), fallback_on=[on_exception, on_response])
        ],
    )
    assert (await agent.run('x')).output == 'hello'


@pytest.mark.anyio
async def test_exhausted_chain_reports_every_failure_and_rejection():
    agent = Agent(
        FunctionModel(failure),
        capabilities=[Fallback(FunctionModel(rejected), fallback_on=[ModelAPIError, reject_nope])],
    )
    with pytest.raises(FallbackExceptionGroup) as exc_info:
        await agent.run('x')
    assert [type(e).__name__ for e in exc_info.value.exceptions] == ['ModelAPIError', 'ResponseRejected']


@pytest.mark.anyio
async def test_a_wrapped_step_model_counts_as_attempted():
    """A candidate that the step's model wraps was already attempted, so it isn't tried again."""
    calls: list[str] = []

    def inner_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('inner')
        raise ModelAPIError(model_name='inner', message='boom')

    inner = FunctionModel(inner_fn)
    agent = Agent(WrapperModel(inner), capabilities=[Fallback(inner, FunctionModel(success))])
    assert (await agent.run('x')).output == 'hello'
    assert calls == ['inner']


@pytest.mark.anyio
async def test_an_unpinned_suspended_response_resumes_on_the_step_model():
    continued: list[bool] = []

    def step_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        continued.append(_ends_suspended(messages))
        return ModelResponse(parts=[TextPart('done')])

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('x')]),
        ModelResponse(parts=[TextPart('partial')], state='suspended', provider_response_id='job-1'),
    ]
    agent = Agent(FunctionModel(step_model), capabilities=[Fallback(FunctionModel(success))])
    await agent.run(message_history=history)
    assert continued == [True]


@pytest.mark.anyio
async def test_a_failed_pinned_continuation_cancels_its_job():
    cancelled: list[str | None] = []

    class CancellingModel(FunctionModel):
        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            cancelled.append(response.provider_response_id)

    pinned = CancellingModel(failure, model_name='pinned')
    agent = Agent(FunctionModel(success, model_name='step'), capabilities=[Fallback(pinned)])
    assert (await agent.run(message_history=_suspended_history(pinned.model_id))).output == 'hello'
    assert cancelled == ['job-1']


@pytest.mark.anyio
async def test_every_candidate_prepares_its_own_messages():
    """Each attempt runs the candidate's own `prepare_messages`, on history not prepared for another model."""
    prepared_by: list[tuple[str, int]] = []

    class MarkingModel(FunctionModel):
        def prepare_messages(
            self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters | None = None
        ) -> list[ModelMessage]:
            markers = sum(
                1 for m in messages for p in m.parts if isinstance(p, UserPromptPart) and p.content == 'marker'
            )
            prepared_by.append((self.model_name, markers))
            last = messages[-1]
            assert isinstance(last, ModelRequest)
            return [*messages[:-1], replace(last, parts=[*last.parts, UserPromptPart('marker')])]

    agent = Agent(
        MarkingModel(failure, model_name='first'),
        capabilities=[Fallback(MarkingModel(success, model_name='second'))],
    )
    assert (await agent.run('x')).output == 'hello'
    assert prepared_by == [('first', 0), ('second', 0)]


@pytest.mark.anyio
async def test_outer_capabilities_only_see_the_accepted_response():
    inner_seen: list[str] = []
    outer_seen: list[str] = []

    def text(response: ModelResponse) -> str:
        part = response.parts[0]
        assert isinstance(part, TextPart)
        return part.content

    class Outer(AbstractCapability[None]):
        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            outer_seen.append(text(response))
            return response

    class InsideFallback(AbstractCapability[None]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='innermost', wrapped_by=(Fallback,))

        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            inner_seen.append(text(response))
            return response

    agent = Agent(
        FunctionModel(rejected),
        deps_type=type(None),
        capabilities=[Outer(), Fallback(FunctionModel(success), fallback_on=reject_nope), InsideFallback()],
    )
    assert (await agent.run('x')).output == 'hello'
    assert inner_seen == ['nope', 'hello']
    assert outer_seen == ['hello']


def _rejected_with_usage(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('nope')], usage=RequestUsage(input_tokens=10, output_tokens=3))


def _success_with_usage(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('hello')], usage=RequestUsage(input_tokens=1, output_tokens=1))


@pytest.mark.anyio
async def test_usage_counts_one_request_and_every_attempts_tokens():
    agent = Agent(
        FunctionModel(_rejected_with_usage),
        capabilities=[
            Fallback(FunctionModel(_rejected_with_usage), FunctionModel(_success_with_usage), fallback_on=reject_nope)
        ],
    )
    result = await agent.run('x')
    assert result.output == 'hello'
    assert (result.usage.requests, result.usage.input_tokens, result.usage.output_tokens) == (1, 21, 7)


@pytest.mark.anyio
async def test_token_limits_apply_to_rejected_attempts():
    agent = Agent(
        FunctionModel(_rejected_with_usage),
        capabilities=[
            Fallback(FunctionModel(_rejected_with_usage), FunctionModel(_success_with_usage), fallback_on=reject_nope)
        ],
    )
    with pytest.raises(UsageLimitExceeded, match='input_tokens_limit of 15'):
        await agent.run('x', usage_limits=UsageLimits(input_tokens_limit=15))


@pytest.mark.anyio
async def test_tokens_are_counted_for_each_candidate():
    counted: list[str] = []

    class CountingModel(FunctionModel):
        async def count_tokens(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> RequestUsage:
            counted.append(self.model_name)
            return RequestUsage(input_tokens=1)

    agent = Agent(
        CountingModel(failure, model_name='first'),
        capabilities=[Fallback(CountingModel(success, model_name='second'))],
    )
    result = await agent.run('x', usage_limits=UsageLimits(count_tokens_before_request=True))
    assert result.output == 'hello'
    assert counted == ['first', 'second']


@pytest.mark.anyio
async def test_a_model_retry_after_a_rejected_attempt_keeps_only_the_retried_response():
    retried: list[bool] = []
    second_responses = iter(['retry me', 'hello'])

    def second(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(next(second_responses))])

    class RetryOnce(AbstractCapability[None]):
        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            part = response.parts[0]
            if isinstance(part, TextPart) and part.content == 'retry me':
                retried.append(True)
                raise ModelRetry('again')
            return response

    agent = Agent(
        FunctionModel(rejected),
        deps_type=type(None),
        capabilities=[RetryOnce(), Fallback(FunctionModel(second), fallback_on=reject_nope)],
    )
    result = await agent.run('x')
    assert result.output == 'hello'
    assert retried == [True]
    texts = [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, TextPart)
    ]
    assert texts == ['retry me', 'hello']


@pytest.mark.anyio
async def test_a_run_model_is_attempted_before_the_chain():
    calls: list[str] = []

    def failing(name: str) -> FunctionModel:
        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(name)
            raise ModelAPIError(model_name=name, message='boom')

        return FunctionModel(fn, model_name=name)

    agent = Agent(capabilities=[Fallback(failing('first'), FunctionModel(success))])
    assert (await agent.run('x', model=failing('explicit'))).output == 'hello'
    assert calls == ['explicit', 'first']


@pytest.mark.anyio
async def test_a_run_level_model_selection_outranks_the_fallback_default():
    selected = FunctionModel(success, model_name='selected')
    agent = Agent(capabilities=[Fallback(FunctionModel(rejected, model_name='default'))])
    result = await agent.run('x', capabilities=[SelectModel(lambda ctx: selected)])
    assert result.output == 'hello'


@pytest.mark.anyio
async def test_a_streamed_resume_waits_out_the_continuation_delay_before_opening():
    events: list[str] = []

    async def pinned_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        events.append(f'open, continuing={_ends_suspended(messages)}')
        yield 'done'

    async def record_sleep(delay: float) -> None:
        events.append(f'sleep {delay}')

    class PollingModel(FunctionModel):
        def continuation_delay(self, response: ModelResponse) -> float | None:
            return 0.001

    step = PollingModel(stream_function=pinned_stream, model_name='step')
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('x')]),
        ModelResponse(parts=[TextPart('partial')], state='suspended', provider_response_id='job-1'),
    ]
    agent = Agent(step, capabilities=[Fallback(FunctionModel(success))])
    with Agent.using_sleep(record_sleep):
        async with agent.run_stream(message_history=history) as stream:
            await stream.get_output()
    assert events == ['sleep 0.001', 'open, continuing=True']


@pytest.mark.anyio
async def test_a_wrapped_fallback_default_is_still_outranked():
    selected = FunctionModel(success, model_name='selected')
    agent = Agent(
        capabilities=[
            SelectModel(lambda ctx: selected),
            WrapperCapability(wrapped=Fallback(FunctionModel(rejected, model_name='default'))),
        ]
    )
    assert (await agent.run('x')).output == 'hello'
