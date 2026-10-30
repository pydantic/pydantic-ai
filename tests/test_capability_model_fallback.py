"""The `ModelFallback` capability.

The attempt loop it drives (`prepare_model_request`, `RetryModelRequest`) is tested in
`test_model_request_attempts.py`.

These are offline tests with `FunctionModel` rather than recorded requests to providers: which model is attempted
next, and what each attempt is sent and records, doesn't depend on the provider, and scripted models can fail, reject
or suspend on cue. A few read internals no request shows, such as a model's provider name, the post-compaction window
tool dispatch uses, and a run's cancellation.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast, get_args

import anyio
import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    AgentRunResult,
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeOutput,
    PendingModelResponse,
    RunContext,
    TextPart,
    ToolCallPart,
    UserPromptPart,
    capture_run_messages,
    exceptions,
    messages as _messages,
)
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import (
    AbstractCapability,
    BatchModelRequests,
    CapabilityOrdering,
    Instrumentation,
    ModelFallback,
    NativeTool,
    ResolveModelId,
    SelectModel,
    WrapModelRequestHandler,
    WrapperCapability,
)
from pydantic_ai.capabilities.model_fallback import _SpecFallbackError  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.exceptions import (
    ContentFilterError,
    FallbackExceptionGroup,
    ModelAPIError,
    ModelRateLimitError,
    ModelRetry,
    RetryModelRequest,
    RunCancelled,
    RunTimedOut,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import CompactionPart, ModelRequestAttempt, ToolReturnPart
from pydantic_ai.models import (
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    ModelResolutionContext,
    StreamedResponse,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCalls, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from ._inline_snapshot import snapshot
from .batch_utils import FAST_BATCH_SETTINGS, FailingBatchTestModel, SlowBatchTestModel
from .conftest import try_import
from .continuation_utils import ScriptedContinuationModel, StreamSegment
from .model_lifecycle_utils import LifecycleTrackingModel

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire


def failure(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelAPIError(model_name='m', message='boom')


def success(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('hello')])


def rejected(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('nope')])


def reject_nope(response: ModelResponse) -> bool:
    return isinstance(response.parts[0], TextPart) and response.parts[0].content == 'nope'


async def test_agent_model_then_capability_model():
    agent = Agent(FunctionModel(failure), capabilities=[ModelFallback(FunctionModel(success))])
    result = await agent.run('x')
    assert result.output == 'hello'
    assert result.usage.requests == 1


async def test_no_agent_model_uses_first_candidate():
    agent = Agent(capabilities=[ModelFallback(FunctionModel(failure), FunctionModel(success))])
    result = await agent.run('x')
    assert result.output == 'hello'


async def test_chain_exhausted():
    agent = Agent(FunctionModel(failure), capabilities=[ModelFallback(FunctionModel(failure))])
    with pytest.raises(FallbackExceptionGroup) as exc_info:
        await agent.run('x')
    assert len(exc_info.value.exceptions) == 2


async def test_response_rejection():
    agent = Agent(
        FunctionModel(rejected),
        capabilities=[ModelFallback(FunctionModel(success), fallback_on=reject_nope)],
    )
    result = await agent.run('x')
    assert result.output == 'hello'


async def test_duplicate_model_skipped():
    m = FunctionModel(failure)
    ok = FunctionModel(success)
    agent = Agent(m, capabilities=[ModelFallback(m, ok)])
    result = await agent.run('x')
    assert result.output == 'hello'


async def failure_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    raise ModelAPIError(model_name='m', message='boom')
    yield ''  # pragma: no cover


async def success_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'hello'


async def test_streaming_open_failure_falls_back():
    agent = Agent(
        FunctionModel(stream_function=failure_stream),
        capabilities=[ModelFallback(FunctionModel(stream_function=success_stream))],
    )
    async with agent.run_stream('x') as stream:
        assert await stream.get_output() == 'hello'


def _suspended_history(pinned_model_id: str) -> list[ModelMessage]:
    """History ending in a response a provider suspended, started by the model `pinned_model_id` identifies."""
    return [
        ModelRequest(parts=[UserPromptPart('x')]),
        ModelResponse(
            parts=[TextPart('partial')],
            model_id=pinned_model_id,
            state='suspended',
            provider_response_id='job-1',
        ),
    ]


def _ends_suspended(messages: list[ModelMessage]) -> bool:
    return isinstance(messages[-1], ModelResponse) and messages[-1].state == 'suspended'


async def test_resumed_continuation_goes_to_the_pinned_model():
    """Resuming a suspended response continues it on the candidate that started it, not the chain's first model."""

    def step_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('the continuation must not go to the step model')  # pragma: no cover

    continued: list[bool] = []

    def pinned_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        continued.append(_ends_suspended(messages))
        return ModelResponse(parts=[TextPart('done')])

    pinned = FunctionModel(pinned_model, model_name='pinned')
    agent = Agent(FunctionModel(step_model, model_name='step'), capabilities=[ModelFallback(pinned)])
    result = await agent.run(message_history=_suspended_history(pinned.model_id))
    assert continued == [True]
    assert result.usage.requests == 1


async def test_failed_pinned_continuation_moves_on_to_the_next_model():
    """A pinned continuation that fails is dropped, and the turn is generated afresh by the model after the pinned one.

    Not by the chain's first model: when checking on a batch fails, that would submit a new batch to the model the
    chain had already moved on from, so the run would never reach the fallback.
    """

    def step_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('the chain must not start over at the step model')  # pragma: no cover

    next_saw: list[bool] = []

    def next_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        next_saw.append(_ends_suspended(messages))
        return ModelResponse(parts=[TextPart('fresh')])

    pinned = FunctionModel(failure, model_name='pinned')
    agent = Agent(
        FunctionModel(step_model, model_name='step'),
        capabilities=[ModelFallback(pinned, FunctionModel(next_model, model_name='next'))],
    )
    result = await agent.run(message_history=_suspended_history(pinned.model_id))
    assert result.output == 'fresh'
    assert next_saw == [False]
    assert not any(isinstance(m, ModelResponse) and m.state == 'suspended' for m in result.all_messages())


async def test_failed_pinned_continuation_on_the_last_model_exhausts_the_chain():
    """The models before the pinned one aren't attempted when its continuation fails, so a last pinned model ends the chain."""

    def step_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('the chain must not start over at the step model')  # pragma: no cover

    pinned = FunctionModel(failure, model_name='pinned')
    agent = Agent(FunctionModel(step_model, model_name='step'), capabilities=[ModelFallback(pinned)])
    with pytest.raises(FallbackExceptionGroup) as exc_info:
        await agent.run(message_history=_suspended_history(pinned.model_id))
    assert [attempt.model_name for attempt in exc_info.value.attempts] == ['pinned']


async def test_failed_continuation_pinned_to_the_run_model_moves_on_past_it():
    """When the run's model already is the one the continuation is pinned to, a failure still moves on past it.

    Not back to the models before it in the chain, which it would otherwise start over from.
    """
    calls: list[str] = []

    def candidate(name: str, *, fails: bool = False) -> FunctionModel:
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(name)
            if fails:
                raise ModelAPIError(model_name=name, message='failed continuation')
            return ModelResponse(parts=[TextPart(name)])

        return FunctionModel(respond, model_name=name)

    a, b, c = candidate('a'), candidate('b', fails=True), candidate('c')
    agent = Agent(a, capabilities=[ModelFallback(a, b, c)])
    result = await agent.run(model=b, message_history=_suspended_history(b.model_id))
    assert calls == ['b', 'c']
    assert result.output == 'c'


async def test_streamed_resumed_continuation_goes_to_the_pinned_model():
    async def step_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise AssertionError('the continuation must not go to the step model')  # pragma: no cover
        yield ''  # pragma: no cover

    continued: list[bool] = []

    async def pinned_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        continued.append(_ends_suspended(messages))
        yield 'done'

    pinned = FunctionModel(stream_function=pinned_stream, model_name='pinned')
    agent = Agent(FunctionModel(stream_function=step_stream, model_name='step'), capabilities=[ModelFallback(pinned)])
    async with agent.run_stream(message_history=_suspended_history(pinned.model_id)) as stream:
        await stream.get_output()
    assert continued == [True]


async def test_streamed_failed_pinned_continuation_moves_on_to_the_next_model():
    async def step_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise AssertionError('the chain must not start over at the step model')  # pragma: no cover
        yield ''  # pragma: no cover

    next_saw: list[bool] = []

    async def next_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        next_saw.append(_ends_suspended(messages))
        yield 'fresh'

    pinned = FunctionModel(stream_function=failure_stream, model_name='pinned')
    agent = Agent(
        FunctionModel(stream_function=step_stream, model_name='step'),
        capabilities=[ModelFallback(pinned, FunctionModel(stream_function=next_stream, model_name='next'))],
    )
    async with agent.run_stream(message_history=_suspended_history(pinned.model_id)) as stream:
        assert await stream.get_output() == 'fresh'
    assert next_saw == [False]


class _SilentlySuspendsThenFails(ScriptedContinuationModel):
    """Streams a billed, suspended first segment without any output, then fails to open its continuation."""

    def __init__(self) -> None:
        super().__init__(
            segments=[
                StreamSegment(
                    texts=[], state='suspended', provider_response_id='job-1', input_tokens=100, output_tokens=10
                )
            ]
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        if self.request_stream_calls:
            raise ModelAPIError(model_name='scripted', message='continuation failed')
        async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as stream:
            yield stream


async def test_a_streamed_continuation_failing_before_any_output_falls_back():
    """A stream whose first segment is suspended without output hasn't streamed anything when its continuation fails.

    So the failure still falls back, and what the provider billed for the suspended segment is counted.
    """
    primary = _SilentlySuspendsThenFails()
    agent = Agent(primary, capabilities=[ModelFallback(TestModel())])
    async with agent.run_stream('x') as result:
        assert ''.join([text async for text in result.stream_text(delta=True, debounce_by=None)]) == (
            'success (no tool calls)'
        )
    assert result.usage == RunUsage(input_tokens=151, output_tokens=14, requests=1)
    assert [(attempt.model_name, attempt.outcome) for attempt in result.response.failed_attempts or []] == [
        ('scripted', 'error')
    ]
    assert [response.provider_response_id for response in primary.cancelled] == ['job-1']


async def test_a_streamed_continuation_failing_before_any_output_counts_its_usage_once():
    """When nothing falls back from that failure, the billed suspended segment is still counted, and only once."""
    agent = Agent(
        _SilentlySuspendsThenFails(), capabilities=[ModelFallback(TestModel(), fallback_on=lambda error: False)]
    )
    usage = RunUsage()
    with pytest.raises(ModelAPIError, match='continuation failed'):
        # The stream fails to open, so it's never handed over.
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(agent.run_stream('x', usage=usage))
    assert usage == RunUsage(input_tokens=100, output_tokens=10)


class _SilentlySuspendsThenHangs(_SilentlySuspendsThenFails):
    """Like `_SilentlySuspendsThenFails`, but its continuation hangs until the run is stopped."""

    def __init__(self, continuing: asyncio.Event) -> None:
        super().__init__()
        self.continuing = continuing

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        if self.request_stream_calls:
            self.continuing.set()
            await asyncio.Event().wait()
        async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as stream:
            yield stream


@pytest.mark.parametrize('billed', [False, True], ids=['before-any-segment', 'after-a-silent-segment'])
async def test_a_stream_stopped_while_it_opens_is_recorded_as_interrupted(billed: bool):
    """A run stopped while `ModelFallback` waits for a stream's first event records the step as interrupted.

    As it would had the consumer stopped the stream after it opened, so the stopped request isn't left unanswered in
    history, where the next prompt would be merged into it, and what the provider billed is counted once.
    """
    opening = asyncio.Event()

    async def hang(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        opening.set()
        await asyncio.Event().wait()
        yield ''  # pragma: no cover

    model: Model = _SilentlySuspendsThenHangs(opening) if billed else FunctionModel(stream_function=hang)
    agent = Agent(model, capabilities=[ModelFallback(TestModel())])
    contexts: list[RunContext[object]] = []

    @agent.instructions
    def capture_context(ctx: RunContext[object]) -> None:
        contexts.append(ctx)

    async def stop_once_opening() -> None:
        await opening.wait()
        contexts[-1].cancel()

    async def consume(ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
        # Consuming the run's events streams its requests. The run stops before a stream reaches it.
        [event async for event in events]  # pragma: no cover

    usage = RunUsage()
    stopper = asyncio.create_task(stop_once_opening())
    with pytest.raises(RunCancelled) as exc_info:
        await agent.run('x', usage=usage, event_stream_handler=consume)
    await stopper
    assert [(type(message).__name__, getattr(message, 'state', None)) for message in exc_info.value.all_messages()] == [
        ('ModelRequest', 'complete'),
        ('ModelResponse', 'interrupted'),
    ]
    assert usage == (RunUsage(input_tokens=100, output_tokens=10) if billed else RunUsage())


async def test_a_task_cancelled_while_its_stream_is_primed_closes_what_priming_opened():
    """Cancelling the task that consumes the run while a stream is being primed closes the segments it opened.

    Here priming has opened a silent, suspended first segment and is waiting on its continuation's first event, so
    the cancellation reaches priming from outside, not through priming's own scope.
    """
    exited: list[int] = []

    class TracksSegments(_SilentlySuspendsThenHangs):
        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any] | None = None,
        ) -> AsyncGenerator[StreamedResponse]:
            segment = self.request_stream_calls
            try:
                async with super().request_stream(
                    messages, model_settings, model_request_parameters, run_context
                ) as stream:
                    yield stream
            finally:
                exited.append(segment)

    continuing = asyncio.Event()
    agent = Agent(TracksSegments(continuing), capabilities=[ModelFallback(TestModel())])

    async def consume(ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
        # Consuming the run's events streams its requests. The run stops before a stream reaches it.
        [event async for event in events]  # pragma: no cover

    run = asyncio.create_task(agent.run('x', event_stream_handler=consume))
    await continuing.wait()
    run.cancel()
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await run
    # The suspended first segment, and the continuation that was opening.
    assert sorted(exited) == [0, 1]
    cancelled = RunCancelled.from_cancellation(exc_info.value)
    assert cancelled is not None
    assert [(type(message).__name__, getattr(message, 'state', None)) for message in cancelled.all_messages()] == [
        ('ModelRequest', 'complete'),
        ('ModelResponse', 'interrupted'),
    ]


async def test_suspended_response_is_pinned_to_the_model_that_served_it():
    """Walking away from a suspended stream leaves it in history pinned to the candidate that started the job."""
    served = ScriptedContinuationModel(
        segments=[
            StreamSegment(
                texts=['partial '], state='suspended', provider_response_id='job-1', input_tokens=5, output_tokens=2
            )
        ]
    )
    agent = Agent(
        FunctionModel(stream_function=failure_stream, model_name='step'), capabilities=[ModelFallback(served)]
    )
    with capture_run_messages() as messages:
        async with agent.run_stream('x') as result:
            async for _ in result.stream_text(delta=True, debounce_by=None):  # pragma: no branch
                break
    suspended = messages[-1]
    assert isinstance(suspended, ModelResponse)
    assert suspended.state == 'suspended'
    assert suspended.model_id == 'scripted:scripted'
    assert suspended.metadata is None


async def test_complete_response_is_not_pinned():
    agent = Agent(FunctionModel(failure, model_name='step'), capabilities=[ModelFallback(FunctionModel(success))])
    result = await agent.run('x')
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.metadata is None


async def test_select_model_outranks_the_fallback_default():
    """With no agent model, `ModelFallback`'s first candidate only stands in when nothing else selects a model."""
    selected = FunctionModel(success, model_name='selected')
    agent = Agent(
        capabilities=[
            SelectModel(lambda ctx: selected),
            ModelFallback(FunctionModel(rejected, model_name='default')),
        ]
    )
    assert (await agent.run('x')).output == 'hello'


async def test_a_model_another_capability_retried_is_not_picked_again():
    calls: list[str] = []

    def failing(name: str) -> FunctionModel:
        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(name)
            raise ModelAPIError(model_name=name, message='boom')

        return FunctionModel(fn, model_name=name)

    retried = failing('retried')

    class RetryOnce(AbstractCapability[None]):
        """Moves the first failed attempt to `retried`, then leaves errors to `ModelFallback`."""

        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='innermost', wrapped_by=(ModelFallback,))

        async def on_model_request_error(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            if request_context.attempt == 1:
                raise RetryModelRequest(retried)
            raise error

    agent = Agent(
        failing('step'),
        deps_type=type(None),
        capabilities=[ModelFallback(retried, FunctionModel(success)), RetryOnce()],
    )
    assert (await agent.run('x')).output == 'hello'
    assert calls == ['step', 'retried']


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('backup', [True, False])
async def test_a_candidate_resolving_to_an_attempted_model_is_skipped(backup: bool, stream: bool):
    """A candidate given as a string that resolves to a model already attempted, like the step's own, isn't sent again."""
    calls: list[str] = []

    def primary_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('primary')
        raise ModelAPIError(model_name='primary', message='boom')

    async def primary_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        calls.append('primary')
        raise ModelAPIError(model_name='primary', message='boom')
        yield ''  # pragma: no cover

    def backup_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('backup')
        return ModelResponse(parts=[TextPart('ok')])

    async def backup_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        calls.append('backup')
        yield 'ok'

    primary = FunctionModel(primary_fn, stream_function=primary_stream, model_name='primary')

    class Alias(AbstractCapability[None]):
        async def resolve_model_id(self, ctx: ModelResolutionContext[None], *, model_id: str) -> Model | None:
            return primary if model_id == 'my-primary' else None

    candidates: list[Model | str] = ['my-primary']
    if backup:
        candidates.append(FunctionModel(backup_fn, stream_function=backup_stream, model_name='backup'))
    agent = Agent(primary, deps_type=type(None), capabilities=[Alias(), ModelFallback(*candidates)])

    async def consume(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
        # Consuming the run's events streams its requests.
        [event async for event in events]

    handler = consume if stream else None
    if backup:
        assert (await agent.run('x', event_stream_handler=handler)).output == 'ok'
        assert calls == ['primary', 'backup']
    else:
        with pytest.raises(FallbackExceptionGroup) as exc_info:
            await agent.run('x', event_stream_handler=handler)
        assert calls == ['primary']
        assert [attempt.model_name for attempt in exc_info.value.attempts] == ['primary']


async def test_fallback_candidates_are_entered_once_and_exited_with_the_run():
    """A candidate is entered when first attempted, once however many steps attempt it, and exited when the run ends."""
    events: list[str] = []
    candidate = LifecycleTrackingModel(events, include_exit_exception=False)
    agent = Agent(FunctionModel(failure), capabilities=[ModelFallback(candidate)])

    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    result = await agent.run('x')
    assert result.output
    assert events.count('enter') == 1
    assert events[-1] == 'exit'
    assert events.count('request') == len([m for m in result.all_messages() if isinstance(m, ModelResponse)])


async def test_a_response_started_outside_the_chain_resumes_on_the_step_model():
    """A suspended response a model outside the chain started is continued by the step's model, as without fallback."""
    continued: list[bool] = []

    def step_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        continued.append(_ends_suspended(messages))
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(step_model, model_name='step'), capabilities=[ModelFallback(FunctionModel(success))])
    await agent.run(message_history=_suspended_history('function:gone'))
    assert continued == [True]


async def test_a_fallback_model_candidate_keeps_its_own_pin():
    """`ModelFallback` leaves the metadata alone, so a `FallbackModel` candidate still resumes on its inner model."""
    fallback = ModelFallback[None](TestModel())
    inner = FunctionModel(success, model_name='inner')
    ctx = RunContext[None](deps=None, model=inner, usage=RunUsage())
    request_context = ModelRequestContext(
        model=inner, messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
    )
    response = ModelResponse(
        parts=[], state='suspended', metadata={'__pydantic_ai__': {'fallback_model_id': 'function:inner-of-inner'}}
    )
    accepted = await fallback.after_model_request(ctx, request_context=request_context, response=response)
    assert accepted.metadata == {'__pydantic_ai__': {'fallback_model_id': 'function:inner-of-inner'}}


def test_fallback_requires_a_model():
    with pytest.raises(UserError, match='requires at least one model'):
        ModelFallback()


async def test_an_error_fallback_on_does_not_match_propagates():
    later: list[str] = []

    def potato(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ValueError('not a model API error')

    def later_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        later.append('called')  # pragma: no cover
        return ModelResponse(parts=[TextPart('hello')])  # pragma: no cover

    agent = Agent(FunctionModel(potato), capabilities=[ModelFallback(FunctionModel(later_model))])
    with pytest.raises(ValueError, match='not a model API error'):
        await agent.run('x')
    assert later == []


async def test_async_predicates():
    async def on_exception(exc: Exception) -> bool:
        return isinstance(exc, ModelAPIError)

    async def on_response(response: ModelResponse) -> bool:
        return reject_nope(response)

    agent = Agent(
        FunctionModel(failure),
        capabilities=[
            ModelFallback(FunctionModel(rejected), FunctionModel(success), fallback_on=[on_exception, on_response])
        ],
    )
    assert (await agent.run('x')).output == 'hello'


async def test_exhausted_chain_reports_every_failure_and_rejection():
    agent = Agent(
        FunctionModel(failure),
        capabilities=[ModelFallback(FunctionModel(rejected), fallback_on=[ModelAPIError, reject_nope])],
    )
    with pytest.raises(FallbackExceptionGroup) as exc_info:
        await agent.run('x')
    assert [type(e).__name__ for e in exc_info.value.exceptions] == ['ModelAPIError', 'ResponseRejected']


async def test_a_wrapped_step_model_counts_as_attempted():
    """A candidate that the step's model wraps was already attempted, so it isn't tried again."""
    calls: list[str] = []

    def inner_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('inner')
        raise ModelAPIError(model_name='inner', message='boom')

    inner = FunctionModel(inner_fn)
    agent = Agent(WrapperModel(inner), capabilities=[ModelFallback(inner, FunctionModel(success))])
    assert (await agent.run('x')).output == 'hello'
    assert calls == ['inner']


async def test_an_unpinned_suspended_response_resumes_on_the_step_model():
    continued: list[bool] = []

    def step_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        continued.append(_ends_suspended(messages))
        return ModelResponse(parts=[TextPart('done')])

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('x')]),
        ModelResponse(parts=[TextPart('partial')], state='suspended', provider_response_id='job-1'),
    ]
    agent = Agent(FunctionModel(step_model), capabilities=[ModelFallback(FunctionModel(success))])
    await agent.run(message_history=history)
    assert continued == [True]


async def test_a_rejected_resumed_continuation_describes_only_what_this_run_billed():
    """A resumed continuation that's rejected is listed with the usage billed beyond the suspended response it continued.

    The run that suspended the response already counted its usage.
    """

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('bad')], usage=RequestUsage(input_tokens=5, output_tokens=1))

    def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('good')], usage=RequestUsage(input_tokens=1, output_tokens=1))

    def reject_primary(response: ModelResponse) -> bool:
        return response.model_name == 'primary'

    primary_model = FunctionModel(primary, model_name='primary')
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('x')]),
        ModelResponse(
            parts=[TextPart('old')],
            model_name='primary',
            state='suspended',
            model_id=primary_model.model_id,
            usage=RequestUsage(input_tokens=100, output_tokens=10),
        ),
    ]
    agent = Agent(
        primary_model,
        capabilities=[ModelFallback(FunctionModel(backup, model_name='backup'), fallback_on=reject_primary)],
    )
    result = await agent.run(message_history=history)
    assert result.output == 'good'
    assert result.usage.input_tokens == 6
    assert [(attempt.model_name, attempt.usage) for attempt in result.response.failed_attempts or []] == [
        ('primary', RequestUsage(input_tokens=5, output_tokens=1))
    ]


async def test_a_failed_pinned_continuation_cancels_its_job():
    cancelled: list[str | None] = []

    class CancellingModel(FunctionModel):
        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            cancelled.append(response.provider_response_id)

    pinned = CancellingModel(failure, model_name='pinned')
    agent = Agent(
        FunctionModel(success, model_name='step'),
        capabilities=[ModelFallback(pinned, FunctionModel(success, model_name='next'))],
    )
    assert (await agent.run(message_history=_suspended_history(pinned.model_id))).output == 'hello'
    assert cancelled == ['job-1']


async def test_output_schema_and_native_tool_edits_stay_with_their_attempt():
    """A `prepare_model_request` hook's in-place edit of the output schema or a native tool is undone for the next
    attempt, and doesn't reach the agent's own definitions, which the next run starts from."""

    class Answer(BaseModel):
        value: str

    seen: list[tuple[int, str | None, list[str]]] = []

    class EditInPlace(AbstractCapability[None]):
        async def prepare_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            parameters = request_context.model_request_parameters
            assert parameters.output_object is not None
            value_schema = parameters.output_object.json_schema['properties']['value']
            web_search = parameters.native_tools[0]
            assert isinstance(web_search, WebSearchTool)
            seen.append(
                (request_context.attempt, value_schema.get('description'), list(web_search.allowed_domains or []))
            )
            if request_context.attempt == 1:
                value_schema['description'] = 'first attempt only'
                web_search.allowed_domains = web_search.allowed_domains or []
                web_search.allowed_domains.append('first.example')
            return request_context

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('{"value": "ok"}')])

    agent = Agent(
        FunctionModel(failure, model_name='primary'),
        deps_type=type(None),
        output_type=NativeOutput(Answer),
        capabilities=[
            NativeTool(WebSearchTool(allowed_domains=['example.com'])),
            EditInPlace(),
            ModelFallback(FunctionModel(answer, model_name='backup')),
        ],
    )
    await agent.run('x')
    await agent.run('x')
    assert seen == [
        (1, None, ['example.com']),
        (2, None, ['example.com']),
        (1, None, ['example.com']),
        (2, None, ['example.com']),
    ]


def _provider_model(name: str, system: str, seen: list[tuple[str, list[str]]], *, fail: bool) -> FunctionModel:
    """A model of the given provider, supporting web search on Anthropic, that records the native tools it's sent."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append((name, [type(tool).__name__ for tool in info.model_request_parameters.native_tools]))
        if fail:
            raise ModelAPIError(model_name=name, message='boom')
        return ModelResponse(parts=[TextPart('hello')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        respond(messages, info)
        yield 'hello'

    model = FunctionModel(
        respond,
        stream_function=stream,
        model_name=name,
        profile=ModelProfile(supported_native_tools=frozenset({WebSearchTool})) if system == 'anthropic' else None,
    )
    model._system = system  # pyright: ignore[reportPrivateUsage]
    return model


def _anthropic_web_search(evaluated: list[str]) -> Callable[[RunContext[None]], WebSearchTool | None]:
    """A native tool factory that only gives an Anthropic model web search."""

    def factory(ctx: RunContext[None]) -> WebSearchTool | None:
        evaluated.append(ctx.model.system)
        return WebSearchTool() if ctx.model.system == 'anthropic' else None

    return factory


_NO_TOOLS: list[str] = []


@pytest.mark.parametrize(
    ('primary_system', 'backup_system', 'sent'),
    [
        # What the factory gave the step's model isn't sent to a model it doesn't support, or means nothing to.
        ('anthropic', 'openai', [('primary', ['WebSearchTool']), ('backup', _NO_TOOLS)]),
        # And a model the factory gives a tool gets it, though the step's model got none.
        ('openai', 'anthropic', [('primary', _NO_TOOLS), ('backup', ['WebSearchTool'])]),
        # A model of the same provider gets what the step's model got.
        ('anthropic', 'anthropic', [('primary', ['WebSearchTool']), ('backup', ['WebSearchTool'])]),
    ],
)
@pytest.mark.parametrize('streaming', [False, True])
async def test_native_tool_factories_resolve_for_each_model_attempted(
    primary_system: str, backup_system: str, sent: list[tuple[str, list[str]]], streaming: bool
):
    """A native tool factory sees the model serving each attempt as `ctx.model`, as each model would on its own."""
    seen: list[tuple[str, list[str]]] = []
    evaluated: list[str] = []
    agent = Agent(
        _provider_model('primary', primary_system, seen, fail=True),
        deps_type=type(None),
        capabilities=[
            ModelFallback(_provider_model('backup', backup_system, seen, fail=False)),
            NativeTool(_anthropic_web_search(evaluated)),
        ],
    )
    if streaming:
        async with agent.run_stream('x') as result:
            assert await result.get_output() == 'hello'
    else:
        assert (await agent.run('x')).output == 'hello'
    assert seen == sent
    assert evaluated == [primary_system, backup_system]


async def test_a_native_tool_a_hook_removed_stays_removed_for_other_models():
    """A native tool `before_model_request` removed isn't brought back by resolving its factory for another model."""
    seen: list[tuple[str, list[str]]] = []
    evaluated: list[str] = []

    class RemoveNativeTools(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            request_context.model_request_parameters = replace(
                request_context.model_request_parameters, native_tools=[]
            )
            return request_context

    agent = Agent(
        _provider_model('primary', 'anthropic', seen, fail=True),
        deps_type=type(None),
        capabilities=[
            ModelFallback(_provider_model('backup', 'anthropic', seen, fail=False)),
            NativeTool(_anthropic_web_search(evaluated)),
            RemoveNativeTools(),
        ],
    )
    assert (await agent.run('x')).output == 'hello'
    assert seen == [('primary', []), ('backup', [])]
    assert evaluated == ['anthropic']


class _TrimmingFunctionModel(FunctionModel):
    """A model that drops the history before a compaction part it accepts, as the OpenAI Responses API does."""

    _honors_compaction_boundaries = True


@pytest.mark.parametrize('fallback', ['capability', 'model'])
async def test_tool_calls_are_judged_by_the_history_the_answering_model_was_sent(
    fallback: str, monkeypatch: pytest.MonkeyPatch
):
    """Which earlier tools a call may rely on is decided by the model that answered, not the step's model.

    The answering model accepts a compaction from a compatible system, and only saw the history after it, so
    the call's evidence starts there. `FallbackModel` asks the model that produced the response, and so does
    `ModelFallback`.
    """

    def respond(name: str, system: str, *, fail: bool) -> FunctionModel:
        def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if fail:
                raise ModelAPIError(model_name=name, message='boom')
            if isinstance(messages[-1].parts[0], ToolReturnPart):
                return ModelResponse(parts=[TextPart('done')], provider_name=system)
            return ModelResponse(parts=[ToolCallPart('tool', {})], provider_name=system)

        model = _TrimmingFunctionModel(call_tool, model_name=name)
        model._system = system  # pyright: ignore[reportPrivateUsage]
        return model

    windows: list[list[ModelMessage]] = []
    window_for_response = _messages._post_compaction_window_for_response  # pyright: ignore[reportPrivateUsage]

    def record_window(
        messages: Sequence[ModelMessage], serving_response: ModelResponse, accepts: Any = None
    ) -> list[ModelMessage]:
        windows.append(window := window_for_response(messages, serving_response, accepts))
        return window

    monkeypatch.setattr(_messages, '_post_compaction_window_for_response', record_window)

    primary, backup = respond('primary', 'anthropic', fail=True), respond('backup', 'openai', fail=False)
    agent = (
        Agent(primary, capabilities=[ModelFallback(backup)])
        if fallback == 'capability'
        else Agent(FallbackModel(primary, backup))
    )

    @agent.tool_plain
    def tool() -> str:
        return 'ok'

    compaction = CompactionPart(
        content=None, id='cmp_1', provider_name='openai-codex', provider_details={'encrypted_content': 'encrypted'}
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('before the compaction')]),
        ModelResponse(parts=[compaction], provider_name='openai-codex', canonical_system='openai'),
    ]
    result = await agent.run('after the compaction', message_history=history)
    assert result.output == 'done'
    assert windows[0][0].parts == [compaction]


async def test_a_usage_limit_exceeded_by_a_suspended_response_is_not_fallen_back_from():
    """A token limit the run's own check finds exceeded ends the run, even when `fallback_on` matches every error."""
    calls: list[str] = []

    def expensive(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('expensive')
        return ModelResponse(
            parts=[TextPart('partial')], state='suspended', usage=RequestUsage(input_tokens=100, output_tokens=1)
        )

    def cheap(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('cheap')  # pragma: no cover
        return ModelResponse(parts=[TextPart('ok')])  # pragma: no cover

    agent = Agent(
        FunctionModel(expensive, model_name='expensive'),
        capabilities=[ModelFallback(FunctionModel(cheap, model_name='cheap'), fallback_on=(Exception,))],
    )
    with pytest.raises(UsageLimitExceeded):
        await agent.run('x', usage_limits=UsageLimits(input_tokens_limit=5))
    assert calls == ['expensive']


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
        capabilities=[ModelFallback(MarkingModel(success, model_name='second'))],
    )
    assert (await agent.run('x')).output == 'hello'
    assert prepared_by == [('first', 0), ('second', 0)]


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
            return CapabilityOrdering(position='innermost', wrapped_by=(ModelFallback,))

        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            inner_seen.append(text(response))
            return response

    agent = Agent(
        FunctionModel(rejected),
        deps_type=type(None),
        capabilities=[Outer(), ModelFallback(FunctionModel(success), fallback_on=reject_nope), InsideFallback()],
    )
    assert (await agent.run('x')).output == 'hello'
    assert inner_seen == ['nope', 'hello']
    assert outer_seen == ['hello']


def _rejected_with_usage(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('nope')], usage=RequestUsage(input_tokens=10, output_tokens=3))


def _success_with_usage(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('hello')], usage=RequestUsage(input_tokens=1, output_tokens=1))


async def test_usage_counts_one_request_and_every_attempts_tokens():
    agent = Agent(
        FunctionModel(_rejected_with_usage),
        capabilities=[
            ModelFallback(
                FunctionModel(_rejected_with_usage), FunctionModel(_success_with_usage), fallback_on=reject_nope
            )
        ],
    )
    result = await agent.run('x')
    assert result.output == 'hello'
    assert (result.usage.requests, result.usage.input_tokens, result.usage.output_tokens) == (1, 21, 7)


async def test_token_limits_apply_to_rejected_attempts():
    agent = Agent(
        FunctionModel(_rejected_with_usage),
        capabilities=[
            ModelFallback(
                FunctionModel(_rejected_with_usage), FunctionModel(_success_with_usage), fallback_on=reject_nope
            )
        ],
    )
    with pytest.raises(UsageLimitExceeded, match='input_tokens_limit of 15'):
        await agent.run('x', usage_limits=UsageLimits(input_tokens_limit=15))


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
        capabilities=[ModelFallback(CountingModel(success, model_name='second'))],
    )
    result = await agent.run('x', usage_limits=UsageLimits(count_tokens_before_request=True))
    assert result.output == 'hello'
    assert counted == ['first', 'second']


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
        capabilities=[RetryOnce(), ModelFallback(FunctionModel(second), fallback_on=reject_nope)],
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


async def test_a_run_model_is_attempted_before_the_chain():
    calls: list[str] = []

    def failing(name: str) -> FunctionModel:
        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(name)
            raise ModelAPIError(model_name=name, message='boom')

        return FunctionModel(fn, model_name=name)

    agent = Agent(capabilities=[ModelFallback(failing('first'), FunctionModel(success))])
    assert (await agent.run('x', model=failing('explicit'))).output == 'hello'
    assert calls == ['explicit', 'first']


async def test_a_run_level_model_selection_outranks_the_fallback_default():
    selected = FunctionModel(success, model_name='selected')
    agent = Agent(capabilities=[ModelFallback(FunctionModel(rejected, model_name='default'))])
    result = await agent.run('x', capabilities=[SelectModel(lambda ctx: selected)])
    assert result.output == 'hello'


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
    agent = Agent(step, capabilities=[ModelFallback(FunctionModel(success))])
    with Agent.using_sleep(record_sleep):
        async with agent.run_stream(message_history=history) as stream:
            await stream.get_output()
    assert events == ['sleep 0.001', 'open, continuing=True']


async def test_a_streamed_resume_times_out_while_waiting_out_the_continuation_delay(monkeypatch: pytest.MonkeyPatch):
    """The `request_timeout` deadline covers the continuation delay waited out before the resumed segment opens."""

    class SlowPollingModel(ScriptedContinuationModel):
        def continuation_delay(self, response: ModelResponse) -> float | None:
            return 60

    loop = asyncio.get_running_loop()
    real_time = loop.time
    offset = 0.0

    async def wait_past_the_deadline(delay: float) -> None:
        # Move the event loop's clock past the deadline rather than wait for it.
        nonlocal offset
        offset += 120
        await anyio.sleep_forever()

    monkeypatch.setattr(loop, 'time', lambda: real_time() + offset)
    served = SlowPollingModel()
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('x')]),
        ModelResponse(parts=[TextPart('partial')], state='suspended', provider_response_id='job-1'),
    ]
    agent = Agent(
        served,
        model_settings={'request_timeout': 90},
        capabilities=[ModelFallback(FunctionModel(stream_function=success_stream, model_name='fallback'))],
    )
    with Agent.using_sleep(wait_past_the_deadline):
        async with agent.run_stream(message_history=history) as stream:
            assert await stream.get_output() == 'hello'
    response = stream.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert [(a.model_name, (a.error or '').split(':')[0]) for a in response.failed_attempts or []] == [
        ('scripted', 'ModelRequestTimeout')
    ]
    assert [response.provider_response_id for response in served.cancelled] == ['job-1']


async def test_a_wrapped_fallback_default_is_still_outranked():
    selected = FunctionModel(success, model_name='selected')
    agent = Agent(
        capabilities=[
            SelectModel(lambda ctx: selected),
            WrapperCapability(wrapped=ModelFallback(FunctionModel(rejected, model_name='default'))),
        ]
    )
    assert (await agent.run('x')).output == 'hello'


async def test_response_handlers_are_not_consulted_for_a_streamed_response():
    async def nope_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'nope'

    agent = Agent(
        FunctionModel(stream_function=nope_stream),
        capabilities=[ModelFallback(FunctionModel(stream_function=success_stream), fallback_on=reject_nope)],
    )
    with pytest.warns(
        UserWarning, match='`ModelFallback` response handlers in `fallback_on` are not applied to streamed'
    ):
        async with agent.run_stream('x') as stream:
            assert await stream.get_output() == 'nope'


async def failure_before_first_event_stream(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:
    """A provider that accepts the streamed request, then reports an overload before any output."""
    # A chunk that produces no event, like a provider's `response.created`, so the stream opens fine.
    yield {}
    raise ModelAPIError(model_name='m', message='overloaded')


async def test_stream_error_before_first_event_falls_back():
    """An error a stream raises before its first event falls back, as one while opening it does."""
    first = FunctionModel(stream_function=failure_before_first_event_stream, model_name='failure')
    agent = Agent(first, capabilities=[ModelFallback(FunctionModel(stream_function=success_stream))])
    async with agent.run_stream('x') as stream:
        assert await stream.get_output() == 'hello'
    response = stream.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert _described(response.failed_attempts) == snapshot(
        [('failure', 'function', 'error', 'ModelAPIError: overloaded', None)]
    )


async def test_stream_error_after_first_event_propagates():
    """Once a stream has produced an event, a later error propagates rather than falling back."""

    async def fails_after_first_event(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'partial'
        raise ModelAPIError(model_name='m', message='lost')

    fallback_calls: list[bool] = []

    async def fallback_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        fallback_calls.append(True)  # pragma: no cover
        yield 'hello'  # pragma: no cover

    agent = Agent(
        FunctionModel(stream_function=fails_after_first_event),
        capabilities=[ModelFallback(FunctionModel(stream_function=fallback_stream))],
    )
    with pytest.raises(ModelAPIError, match='lost'):
        async with agent.run_stream('x') as stream:
            await stream.get_output()
    assert fallback_calls == []


# Telemetry and usage, compared with the same chain under `FallbackModel`.


def _billed_rejection(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart('nope')], usage=RequestUsage(input_tokens=100, output_tokens=10, cost=Decimal('0.001'))
    )


def _billed_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart('hello')], usage=RequestUsage(input_tokens=20, output_tokens=2, cost=Decimal('0.002'))
    )


_FALLBACK_ON = (ModelAPIError, reject_nope)


def _chain(*functions: Any) -> list[Model]:
    return [FunctionModel(function, model_name=function.__name__.strip('_')) for function in functions]


def _capability_agent(*functions: Any, instrument: bool = False) -> Agent[Any, str]:
    first, *rest = _chain(*functions)
    capabilities: list[AbstractCapability[Any]] = [ModelFallback(*rest, fallback_on=_FALLBACK_ON)]
    if instrument:
        capabilities.append(Instrumentation(settings=InstrumentationSettings()))
    return Agent(first, capabilities=capabilities)


def _fallback_model_agent(*functions: Any, instrument: bool = False) -> Agent[Any, str]:
    capabilities: list[AbstractCapability[Any]] = []
    if instrument:
        capabilities.append(Instrumentation(settings=InstrumentationSettings()))
    return Agent(FallbackModel(*_chain(*functions), fallback_on=_FALLBACK_ON), capabilities=capabilities)


def _described(attempts: list[ModelRequestAttempt] | Any) -> list[tuple[Any, ...]]:
    """The attempts without their timing, which differs between runs."""
    return [(a.model_name, a.provider_name, a.outcome, a.error, a.usage) for a in attempts]


async def test_usage_and_attempts_match_fallback_model():
    """A rejected response is counted in `RunUsage` and recorded as an attempt exactly as under `FallbackModel`.

    Its tokens and cost count once, at the provider boundary, the step counts as one request, and the
    attempt records on the answer describe that usage without adding it again.
    """
    chain = (failure, _billed_rejection, _billed_answer)
    capability = await _capability_agent(*chain).run('x')
    model = await _fallback_model_agent(*chain).run('x')

    assert (
        capability.usage
        == model.usage
        == snapshot(RunUsage(input_tokens=120, output_tokens=12, cost=Decimal('0.003'), requests=1))
    )
    answer = capability.all_messages()[-1]
    assert isinstance(answer, ModelResponse)
    assert answer.usage == RequestUsage(input_tokens=20, output_tokens=2, cost=Decimal('0.002'))
    model_answer = model.all_messages()[-1]
    assert isinstance(model_answer, ModelResponse)
    assert (
        _described(answer.failed_attempts)
        == _described(model_answer.failed_attempts)
        == snapshot(
            [
                ('failure', 'function', 'error', 'ModelAPIError: boom', None),
                (
                    'billed_rejection',
                    'function',
                    'rejected',
                    None,
                    RequestUsage(input_tokens=100, output_tokens=10, cost=Decimal('0.001')),
                ),
            ]
        )
    )


async def test_exhausted_chain_lists_its_attempts_like_fallback_model():
    chain = (failure, _billed_rejection, _billed_rejection)
    groups: list[FallbackExceptionGroup] = []
    usages: list[RunUsage] = []
    for agent in (_capability_agent(*chain), _fallback_model_agent(*chain)):
        usage = RunUsage()
        with pytest.raises(FallbackExceptionGroup) as exc_info:
            await agent.run('x', usage=usage)
        groups.append(exc_info.value)
        usages.append(usage)

    assert usages[0] == usages[1] == snapshot(RunUsage(input_tokens=200, output_tokens=20, cost=Decimal('0.002')))
    assert (
        _described(groups[0].attempts)
        == _described(groups[1].attempts)
        == snapshot(
            [
                ('failure', 'function', 'error', 'ModelAPIError: boom', None),
                (
                    'billed_rejection',
                    'function',
                    'rejected',
                    None,
                    RequestUsage(input_tokens=100, output_tokens=10, cost=Decimal('0.001')),
                ),
                (
                    'billed_rejection',
                    'function',
                    'rejected',
                    None,
                    RequestUsage(input_tokens=100, output_tokens=10, cost=Decimal('0.001')),
                ),
            ]
        )
    )


async def test_exhausted_chain_raises_the_limit_its_attempts_exceeded():
    """As under `FallbackModel`, the limit the rejected responses exceeded is raised, caused by the group."""
    agent = _capability_agent(_billed_rejection, _billed_rejection)
    with pytest.raises(UsageLimitExceeded, match='input_tokens_limit of 150') as exc_info:
        await agent.run('x', usage_limits=UsageLimits(input_tokens_limit=150))
    assert isinstance(exc_info.value.__cause__, FallbackExceptionGroup)


async def test_a_fallback_model_candidates_own_group_keeps_its_attempts():
    """A `FallbackModel` candidate's group the capability doesn't fall back on lists only the attempts it made."""
    inner = FallbackModel(*_chain(failure, failure))
    agent = Agent(inner, capabilities=[ModelFallback(FunctionModel(success))])
    with pytest.raises(FallbackExceptionGroup) as exc_info:
        await agent.run('x')
    assert _described(exc_info.value.attempts) == snapshot(
        [
            ('failure', 'function', 'error', 'ModelAPIError: boom', None),
            ('failure', 'function', 'error', 'ModelAPIError: boom', None),
        ]
    )


async def test_a_rejected_recovery_is_recorded_as_the_error_it_recovered():
    """A response an error hook made up was never billed, so rejecting it records the attempt's error."""

    @dataclass
    class RecoverWithNope(AbstractCapability[Any]):
        async def on_model_request_error(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart('nope')], usage=RequestUsage(input_tokens=1000))

    first, answer = _chain(failure, _billed_answer)
    agent = Agent(first, capabilities=[RecoverWithNope(), ModelFallback(answer, fallback_on=reject_nope)])
    result = await agent.run('x')
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert _described(response.failed_attempts) == snapshot(
        [('failure', 'function', 'error', 'ModelAPIError: boom', None)]
    )
    assert result.usage == snapshot(RunUsage(input_tokens=20, output_tokens=2, cost=Decimal('0.002'), requests=1))


async def test_a_stream_records_the_attempts_that_failed_to_open():
    first = FunctionModel(stream_function=failure_stream, model_name='failure')
    agent = Agent(first, capabilities=[ModelFallback(FunctionModel(stream_function=success_stream))])
    async with agent.run_stream('x') as stream:
        assert await stream.get_output() == 'hello'
    response = stream.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert _described(response.failed_attempts) == snapshot(
        [('failure', 'function', 'error', 'ModelAPIError: boom', None)]
    )


_SPAN_ATTRIBUTES = (
    'gen_ai.request.model',
    'gen_ai.response.model',
    'pydantic_ai.model_request.attempt',
    'gen_ai.usage.input_tokens',
    'gen_ai.usage.output_tokens',
    'operation.cost',
)


def _model_spans(capfire: CaptureLogfire) -> list[dict[str, Any]]:
    """The spans under the agent run, with their status and the attributes describing the model and usage."""
    statuses = {span.context.span_id: span.status for span in capfire.exporter.exported_spans if span.context}
    spans = capfire.exporter.exported_spans_as_dict()
    names = {span['context']['span_id']: span['name'] for span in spans}
    tree = [
        {
            'name': span['name'],
            'parent': names.get(span['parent']['span_id']) if span['parent'] else None,
            'status': statuses[span['context']['span_id']].status_code.name,
            'attributes': {key: span['attributes'][key] for key in _SPAN_ATTRIBUTES if key in span['attributes']},
        }
        for span in spans
        if span['name'] != 'invoke_agent agent'
    ]
    capfire.exporter.clear()
    return tree


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_spans_match_fallback_model(capfire: CaptureLogfire):
    """Each failed attempt gets an ERROR span under `chat`, which is named after and reports the model that answered."""
    chain = (failure, _billed_rejection, _billed_answer)
    await _capability_agent(*chain, instrument=True).run('x')
    capability = _model_spans(capfire)
    await _fallback_model_agent(*chain, instrument=True).run('x')
    assert capability == _model_spans(capfire)
    assert capability == snapshot(
        [
            {
                'name': 'model request attempt failure',
                'parent': 'chat billed_answer',
                'status': 'ERROR',
                'attributes': {
                    'gen_ai.request.model': 'failure',
                    'gen_ai.response.model': 'failure',
                    'pydantic_ai.model_request.attempt': 0,
                },
            },
            {
                'name': 'model request attempt billed_rejection',
                'parent': 'chat billed_answer',
                'status': 'ERROR',
                'attributes': {
                    'gen_ai.request.model': 'billed_rejection',
                    'gen_ai.response.model': 'billed_rejection',
                    'pydantic_ai.model_request.attempt': 1,
                    'gen_ai.usage.input_tokens': 100,
                    'gen_ai.usage.output_tokens': 10,
                    'operation.cost': 0.001,
                },
            },
            {
                'name': 'chat billed_answer',
                'parent': 'invoke_agent agent',
                'status': 'UNSET',
                'attributes': {
                    'gen_ai.request.model': 'billed_answer',
                    'gen_ai.response.model': 'billed_answer',
                    'gen_ai.usage.input_tokens': 20,
                    'gen_ai.usage.output_tokens': 2,
                    'operation.cost': 0.002,
                },
            },
        ]
    )


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_rejected_attempt_span_status_says_what_rejected_it(capfire: CaptureLogfire):
    """A `FallbackModel` attempt span says its `fallback_on` rejected the response, which is all that can there.

    In the agent's attempt loop any capability can reject one, so its span only says the response was rejected.
    """

    def rejected_statuses() -> list[str | None]:
        statuses = [
            span.status.description
            for span in capfire.exporter.exported_spans
            # Logfire also exports a pending span when each span starts, before its status is set.
            if span.name == 'model request attempt billed_rejection'
            and (span.attributes or {}).get('logfire.span_type') != 'pending_span'
        ]
        capfire.exporter.clear()
        return statuses

    chain = (_billed_rejection, _billed_answer)
    await _capability_agent(*chain, instrument=True).run('x')
    assert rejected_statuses() == ['Response rejected']
    await _fallback_model_agent(*chain, instrument=True).run('x')
    assert rejected_statuses() == ['Response rejected by `fallback_on`']


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_exhausted_chain_spans_match_fallback_model(capfire: CaptureLogfire):
    """When every attempt fails, the `chat` span reports the error and no response; each attempt has its own span."""
    chain = (failure, _billed_rejection)
    with pytest.raises(FallbackExceptionGroup):
        await _capability_agent(*chain, instrument=True).run('x')
    # No model answered, so the `chat` span is named after the last one attempted, where a
    # `FallbackModel`'s keeps its own name.
    assert _model_spans(capfire) == snapshot(
        [
            {
                'name': 'model request attempt failure',
                'parent': 'chat billed_rejection',
                'status': 'ERROR',
                'attributes': {
                    'gen_ai.request.model': 'failure',
                    'gen_ai.response.model': 'failure',
                    'pydantic_ai.model_request.attempt': 0,
                },
            },
            {
                'name': 'model request attempt billed_rejection',
                'parent': 'chat billed_rejection',
                'status': 'ERROR',
                'attributes': {
                    'gen_ai.request.model': 'billed_rejection',
                    'gen_ai.response.model': 'billed_rejection',
                    'pydantic_ai.model_request.attempt': 1,
                    'gen_ai.usage.input_tokens': 100,
                    'gen_ai.usage.output_tokens': 10,
                    'operation.cost': 0.001,
                },
            },
            {
                'name': 'chat billed_rejection',
                'parent': 'invoke_agent agent',
                'status': 'ERROR',
                'attributes': {'gen_ai.request.model': 'billed_rejection', 'gen_ai.response.model': 'billed_rejection'},
            },
        ]
    )
    with pytest.raises(FallbackExceptionGroup):
        await _fallback_model_agent(*chain, instrument=True).run('x')
    assert _model_spans(capfire) == snapshot(
        [
            {
                'name': 'model request attempt failure',
                'parent': 'chat fallback:failure,billed_rejection',
                'status': 'ERROR',
                'attributes': {
                    'gen_ai.request.model': 'failure',
                    'gen_ai.response.model': 'failure',
                    'pydantic_ai.model_request.attempt': 0,
                },
            },
            {
                'name': 'model request attempt billed_rejection',
                'parent': 'chat fallback:failure,billed_rejection',
                'status': 'ERROR',
                'attributes': {
                    'gen_ai.request.model': 'billed_rejection',
                    'gen_ai.response.model': 'billed_rejection',
                    'pydantic_ai.model_request.attempt': 1,
                    'gen_ai.usage.input_tokens': 100,
                    'gen_ai.usage.output_tokens': 10,
                    'operation.cost': 0.001,
                },
            },
            {
                'name': 'chat fallback:failure,billed_rejection',
                'parent': 'invoke_agent agent',
                'status': 'ERROR',
                'attributes': {
                    'gen_ai.request.model': 'fallback:failure,billed_rejection',
                    'gen_ai.response.model': 'fallback:failure,billed_rejection',
                },
            },
        ]
    )


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_stream_spans_match_fallback_model(capfire: CaptureLogfire):
    first = FunctionModel(stream_function=failure_stream, model_name='failure')
    answer = FunctionModel(stream_function=success_stream, model_name='answer')
    instrumentation = Instrumentation(settings=InstrumentationSettings())
    trees: list[list[dict[str, Any]]] = []
    for agent in (
        Agent(first, capabilities=[ModelFallback(answer), instrumentation]),
        Agent(FallbackModel(first, answer), capabilities=[instrumentation]),
    ):
        async with agent.run_stream('x') as stream:
            await stream.get_output()
        trees.append(_model_spans(capfire))
    assert (
        trees[0]
        == trees[1]
        == snapshot(
            [
                {
                    'name': 'model request attempt failure',
                    'parent': 'chat answer',
                    'status': 'ERROR',
                    'attributes': {
                        'gen_ai.request.model': 'failure',
                        'gen_ai.response.model': 'failure',
                        'pydantic_ai.model_request.attempt': 0,
                    },
                },
                {
                    'name': 'chat answer',
                    'parent': 'invoke_agent agent',
                    'status': 'UNSET',
                    'attributes': {
                        'gen_ai.request.model': 'answer',
                        'gen_ai.response.model': 'answer',
                        'gen_ai.usage.input_tokens': 50,
                        'gen_ai.usage.output_tokens': 1,
                    },
                },
            ]
        )
    )


LONG_TIMEOUT = 60
"""A run timeout no test waits out: a test that needs the run to time out expires its deadline with `_expire`."""


def _expire(ctx: RunContext[object]) -> None:
    """Expire the run's deadline now, as its timer would, so a test doesn't race a real timer against the run's progress."""
    cancellation = ctx._cancellation  # pyright: ignore[reportPrivateUsage]
    assert cancellation is not None
    cancellation.expire()


def _capture_contexts(agent: Agent[object, Any]) -> list[RunContext[object]]:
    """The run contexts the agent's requests are made with, to expire the run from."""
    contexts: list[RunContext[object]] = []

    @agent.instructions
    def capture_context(ctx: RunContext[object]) -> None:
        contexts.append(ctx)

    return contexts


async def test_a_run_timeout_is_not_fallen_back_from():
    """A run that times out while a candidate is answering ends with `RunTimedOut`; the next candidate isn't tried."""
    fallback_calls: list[bool] = []

    async def hangs(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        _expire(contexts[-1])
        await asyncio.Event().wait()
        raise AssertionError('unreachable')  # pragma: no cover

    def fallback(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        fallback_calls.append(True)  # pragma: no cover
        return ModelResponse(parts=[TextPart('hello')])  # pragma: no cover

    agent = Agent(FunctionModel(hangs), capabilities=[ModelFallback(FunctionModel(fallback), fallback_on=(Exception,))])
    contexts = _capture_contexts(agent)
    with pytest.raises(RunTimedOut):
        await agent.run('x', timeout=LONG_TIMEOUT)
    assert fallback_calls == []


def _suspends_then_hangs(started: asyncio.Event, on_hang: Callable[[], None] = lambda: None) -> FunctionModel:
    """A model whose first request is billed and suspended, and whose continuation hangs until the run stops.

    `on_hang` is called as the continuation starts hanging, to stop the run.
    """
    calls = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[TextPart('partial')], state='suspended', usage=RequestUsage(input_tokens=10, output_tokens=2)
            )
        started.set()
        on_hang()
        await asyncio.Event().wait()
        raise AssertionError('unreachable')  # pragma: no cover

    return FunctionModel(respond, model_name='suspends')


async def test_a_cancelled_continuation_keeps_the_usage_already_billed():
    """A run cancelled while continuing a suspended response still counts what the provider billed before it."""
    started = asyncio.Event()
    agent = Agent(_suspends_then_hangs(started), capabilities=[ModelFallback(TestModel())])
    usage = RunUsage()
    with pytest.raises(RunCancelled) as exc_info:
        async with agent.iter('x', usage=usage) as run:

            async def cancel() -> None:
                await started.wait()
                run.cancel()

            canceller = asyncio.create_task(cancel())
            try:
                async for _ in run:
                    pass
            finally:
                await canceller
    assert usage == RunUsage(input_tokens=10, output_tokens=2)
    assert exc_info.value.usage == RunUsage(input_tokens=10, output_tokens=2)


async def test_a_timed_out_continuation_keeps_the_usage_already_billed():
    """A run that times out while continuing a suspended response still counts what the provider billed before it."""
    agent = Agent(
        _suspends_then_hangs(asyncio.Event(), lambda: _expire(contexts[-1])), capabilities=[ModelFallback(TestModel())]
    )
    contexts = _capture_contexts(agent)
    usage = RunUsage()
    with pytest.raises(RunTimedOut):
        await agent.run('x', usage=usage, timeout=LONG_TIMEOUT)
    assert usage == RunUsage(input_tokens=10, output_tokens=2)


async def test_each_candidate_gets_its_own_max_retries():
    """`max_retries` set on a candidate applies to that candidate only, as under `FallbackModel`."""
    seen: list[tuple[str, int | None]] = []

    def respond(name: str, fail: bool) -> FunctionModel:
        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append((name, (info.model_settings or {}).get('max_retries')))
            if fail:
                raise ModelAPIError(model_name=name, message='boom')
            return ModelResponse(parts=[TextPart('hello')])

        return FunctionModel(fn, model_name=name, settings={'max_retries': 0} if fail else None)

    agent = Agent(respond('first', fail=True), capabilities=[ModelFallback(respond('last', fail=False))])
    assert (await agent.run('x')).output == 'hello'
    assert seen == [('first', 0), ('last', None)]

    # Set on the agent, it applies to every candidate and takes precedence over their own.
    seen.clear()
    assert (await agent.run('x', model_settings={'max_retries': 1})).output == 'hello'
    assert seen == [('first', 1), ('last', 1)]


async def test_settings_a_hook_sets_reach_every_candidate():
    """A setting a `before_model_request` hook changes or removes applies to every candidate.

    A removed setting falls back to each candidate's own.
    """
    seen: list[tuple[str, int | None, float | None]] = []

    def respond(name: str, fail: bool, settings: ModelSettings) -> FunctionModel:
        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            model_settings = info.model_settings or {}
            seen.append((name, model_settings.get('max_retries'), model_settings.get('temperature')))
            if fail:
                raise ModelAPIError(model_name=name, message='boom')
            return ModelResponse(parts=[TextPart('hello')])

        return FunctionModel(fn, model_name=name, settings=settings)

    @dataclass
    class SetRetriesAndDropTemperature(AbstractCapability[Any]):
        in_place: bool

        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            if self.in_place:
                assert request_context.model_settings is not None
                request_context.model_settings['max_retries'] = 3
                del request_context.model_settings['temperature']
                return request_context
            settings = dict(request_context.model_settings or {})
            settings['max_retries'] = 3
            del settings['temperature']
            return replace(request_context, model_settings=cast(ModelSettings, settings))

    for in_place in (False, True):
        seen.clear()
        agent = Agent(
            respond('first', True, {'max_retries': 2, 'temperature': 0.5}),
            model_settings={'temperature': 0.1},
            capabilities=[
                SetRetriesAndDropTemperature(in_place),
                ModelFallback(respond('last', False, {'max_retries': 5, 'temperature': 0.9})),
            ],
        )
        assert (await agent.run('x')).output == 'hello'
        # Dropping the agent's `temperature` leaves each candidate's own, which it merges in itself.
        assert seen == [('first', 3, 0.5), ('last', 3, 0.9)]


async def test_callable_model_settings_resolve_once_per_step():
    """A callable `model_settings` layer is resolved once per step, even when an attempt moves to another model."""
    calls: list[int] = []

    def settings(ctx: RunContext[Any]) -> ModelSettings:
        calls.append(ctx.run_step)
        return {'temperature': 0.2}

    temperatures: list[float | None] = []

    def respond(fail: bool) -> FunctionModel:
        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            temperatures.append((info.model_settings or {}).get('temperature'))
            if fail:
                raise ModelAPIError(model_name='m', message='boom')
            return ModelResponse(parts=[TextPart('hello')])

        return FunctionModel(fn)

    agent = Agent(respond(True), model_settings=settings, capabilities=[ModelFallback(respond(False))])
    assert (await agent.run('x')).output == 'hello'
    assert calls == [1]
    assert temperatures == [0.2, 0.2]


async def test_a_streamed_request_a_wrapper_answers_without_calling_the_model():
    """A `wrap_model_request` that answers without calling its handler still short-circuits a streamed request."""

    @dataclass
    class AnswerFromCache(AbstractCapability[Any]):
        async def wrap_model_request(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart('cached')])

    agent = Agent(
        FunctionModel(stream_function=failure_stream),
        capabilities=[AnswerFromCache(), ModelFallback(FunctionModel(stream_function=success_stream))],
    )
    async with agent.run_stream('x') as stream:
        assert await stream.get_output() == 'cached'


async def test_cancelling_a_primed_stream_before_iterating_cancels_its_job():
    """A stream `ModelFallback` primed is in flight, so cancelling it before iteration starts cancels its server-side job."""
    served = ScriptedContinuationModel(
        segments=[
            StreamSegment(
                texts=['partial '], state='suspended', provider_response_id='job-1', input_tokens=1, output_tokens=1
            )
        ]
    )
    agent = Agent(served, capabilities=[ModelFallback(FunctionModel(success))])
    async with agent.iter('x') as run:
        async for node in run:  # pragma: no branch
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    await stream.cancel()
                break
    assert [response.provider_response_id for response in served.cancelled] == ['job-1']


async def test_a_failed_continuation_delay_before_priming_cancels_the_resumed_job():
    """If waiting out a resumed job's continuation delay fails, the job is cancelled, as when its segment fails to open."""

    class DelayedModel(ScriptedContinuationModel):
        def continuation_delay(self, response: ModelResponse) -> float | None:
            return 0.001

    async def failing_sleep(delay: float) -> None:
        raise RuntimeError('sleep failed')

    served = DelayedModel()
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('x')]),
        ModelResponse(parts=[TextPart('partial')], state='suspended', provider_response_id='job-1'),
    ]
    agent = Agent(served, capabilities=[ModelFallback(FunctionModel(success))])
    with Agent.using_sleep(failing_sleep), pytest.raises(RuntimeError, match='sleep failed'):
        async with agent.run_stream(message_history=history) as stream:
            await stream.get_output()  # pragma: no cover
    assert [response.provider_response_id for response in served.cancelled] == ['job-1']


async def test_the_fallback_model_is_batched_after_the_primary_batch_fails():
    """With `BatchModelRequests`, the attempt `ModelFallback` moves to also goes into a batch, on the fallback model."""
    primary = FailingBatchTestModel(model_name='primary')
    fallback = SlowBatchTestModel(pending_checks=1, model_name='fallback')
    agent = Agent(primary, capabilities=[ModelFallback(fallback), BatchModelRequests(**FAST_BATCH_SETTINGS)])

    result = await agent.run('Hello')

    assert result.output == 'success (no tool calls)'
    assert (len(primary.created), len(fallback.created)) == (1, 1)
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.model_name == 'fallback'
    assert [(attempt.model_name, attempt.outcome) for attempt in response.failed_attempts or []] == [
        ('primary', 'error')
    ]
    assert result.usage.requests == 1


async def test_a_failed_batch_moves_on_down_the_chain_across_resumed_runs():
    """A run that ends on a pending batch and is resumed from its history moves on from a failed batch result.

    Each run either submits a batch or checks on the one its history ends in. When a check fails, the next model in the
    chain gets a batch, rather than the first model, which would accept a new batch and never let the chain move on.
    """
    primary = FailingBatchTestModel(model_name='primary')
    second = FailingBatchTestModel(model_name='second')
    third = SlowBatchTestModel(model_name='third')

    def batched_agent() -> Agent[None, str | PendingModelResponse]:
        # A new agent for every run, as a run resumed in another process would have.
        return Agent(
            primary,
            output_type=[str, PendingModelResponse],
            capabilities=[ModelFallback(second, third), BatchModelRequests(**FAST_BATCH_SETTINGS)],
        )

    runs = [await batched_agent().run('Hello')]
    # Bounded, as a chain that started over after a failed check would cycle through new batches forever.
    while isinstance(runs[-1].output, PendingModelResponse) and len(runs) < 5:
        runs.append(await batched_agent().run(message_history=runs[-1].all_messages()))

    assert runs[-1].output == 'success (no tool calls)'
    # The model each pending run ended waiting on, as its response records.
    assert [run.output.response.model_id for run in runs if isinstance(run.output, PendingModelResponse)] == [
        'test:primary',
        'test:second',
        'test:third',
    ]
    # Every model got one batch: none was submitted to again after its result failed.
    assert (len(primary.created), len(second.created), len(third.created)) == (1, 1, 1)


async def test_a_rejected_pending_response_has_its_job_cancelled():
    """A response a handler rejects while the provider is still working on it has its job cancelled, as under
    `FallbackModel`, rather than left running while the request moves on."""
    cancelled: list[str | None] = []

    class BackgroundModel(FunctionModel):
        """Answers with a job the provider is still working on, which the run can end on and check later."""

        def continuation_delay(self, response: ModelResponse) -> float | None:
            return 60.0

        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            cancelled.append(response.provider_response_id)

    def pending(job_id: str) -> Callable[[list[ModelMessage], AgentInfo], ModelResponse]:
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[], state='suspended', provider_response_id=job_id)

        return respond

    judged: list[str] = []

    def reject_the_first(response: ModelResponse) -> bool:
        judged.append(response.state)
        return len(judged) == 1

    primary = BackgroundModel(pending('primary-job'), model_name='primary')
    agent = Agent(
        primary,
        output_type=[str, PendingModelResponse],
        capabilities=[
            ModelFallback(BackgroundModel(pending('backup-job'), model_name='backup'), fallback_on=reject_the_first)
        ],
    )
    result = await agent.run('Hello')

    assert isinstance(result.output, PendingModelResponse)
    assert judged == ['suspended', 'suspended']
    # Only the rejected job: the accepted one is left for a later run to check on.
    assert cancelled == ['primary-job']
    assert [(attempt.model_name, attempt.outcome) for attempt in result.response.failed_attempts or []] == [
        ('primary', 'rejected')
    ]


async def test_a_rejected_pending_batch_request_is_withdrawn_once():
    """A pending batch request a handler rejects is withdrawn once: its batch, which nothing else waits on, is cancelled
    a single time, and the run goes on with the fallback's batch request."""
    primary = SlowBatchTestModel(pending_checks=100, model_name='primary')
    backup = SlowBatchTestModel(model_name='backup')
    judged: list[str] = []

    def reject_the_first(response: ModelResponse) -> bool:
        judged.append(response.state)
        return len(judged) == 1

    def batched_agent() -> Agent[None, str | PendingModelResponse]:
        return Agent(
            primary,
            output_type=[str, PendingModelResponse],
            capabilities=[
                ModelFallback(backup, fallback_on=reject_the_first),
                BatchModelRequests(**FAST_BATCH_SETTINGS),
            ],
        )

    pending = await batched_agent().run('Hello')
    assert isinstance(pending.output, PendingModelResponse)
    assert judged == ['suspended', 'suspended']
    assert primary.canceled == [batch.id for batch in primary.created]
    assert (len(primary.created), len(backup.created), backup.canceled) == (1, 1, [])
    assert pending.output.response.model_id == 'test:backup'

    # The fallback's batch request was left for a later run, which collects it.
    result = await batched_agent().run(message_history=pending.all_messages())
    assert result.output == 'success (no tool calls)'
    assert (len(primary.created), len(backup.created), primary.canceled, backup.canceled) == (
        1,
        1,
        [primary.created[0].id],
        [],
    )


def _candidate(name: str, calls: list[str], *, fails: bool = False) -> FunctionModel:
    """A model that logs its calls, calls the `noop` tool when there is one to answer a prompt, and otherwise answers."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(name)
        if fails:
            raise ModelAPIError(model_name=name, message='down')
        if info.function_tools and any(isinstance(part, UserPromptPart) for part in messages[-1].parts):
            return ModelResponse(parts=[ToolCallPart('noop')])
        return ModelResponse(parts=[TextPart(name)])

    return FunctionModel(respond, model_name=name)


def _model_ids(messages: list[ModelMessage]) -> list[str | None]:
    """The model each response in `messages` records it came from."""
    return [message.model_id for message in messages if isinstance(message, ModelResponse)]


def _failed_attempt(model_name: str) -> ModelRequestAttempt:
    """An attempt at `model_name` that failed before the response it's recorded on, which was fallen back to."""
    return ModelRequestAttempt(
        model_name=model_name, outcome='error', timestamp=datetime(2026, 1, 1, tzinfo=UTC), duration=timedelta()
    )


def _sticky_history(model_id: str) -> list[ModelMessage]:
    """A finished exchange whose answer came from the model `model_id` identifies, after the primary failed."""
    return [
        ModelRequest(parts=[UserPromptPart('x')]),
        ModelResponse(parts=[TextPart('earlier')], model_id=model_id, failed_attempts=[_failed_attempt('primary')]),
    ]


def _with_noop(agent: Agent[None, str]) -> Agent[None, str]:
    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    return agent


async def test_sticky_stays_on_the_fallback_across_steps():
    """After falling back, later steps of the run go straight to the candidate that answered."""
    calls: list[str] = []
    agent = _with_noop(
        Agent(
            _candidate('primary', calls, fails=True),
            capabilities=[ModelFallback(_candidate('backup', calls), sticky=True)],
        )
    )
    result = await agent.run('x')
    assert result.output == 'backup'
    assert calls == ['primary', 'backup', 'backup']
    assert _model_ids(result.all_messages()) == ['function:backup', 'function:backup']
    # Going straight to the sticky candidate isn't a failed attempt.
    last = result.all_messages()[-1]
    assert isinstance(last, ModelResponse)
    assert last.failed_attempts is None


async def test_sticky_stays_on_the_fallback_across_runs():
    """A later run continuing the conversation's history starts from the candidate that last answered it."""
    calls: list[str] = []
    backup = _candidate('backup', calls)

    def sticky_agent() -> Agent[None, str]:
        # A new agent for every run, as a run continuing the conversation in another process would have.
        return Agent(_candidate('primary', calls, fails=True), capabilities=[ModelFallback(backup, sticky=True)])

    first = await sticky_agent().run('x')
    second = await sticky_agent().run('y', message_history=first.all_messages())
    assert second.output == 'backup'
    assert calls == ['primary', 'backup', 'backup']
    assert _model_ids(second.new_messages()) == ['function:backup']


async def test_without_sticky_every_step_starts_from_the_step_model():
    calls: list[str] = []
    agent = _with_noop(
        Agent(_candidate('primary', calls, fails=True), capabilities=[ModelFallback(_candidate('backup', calls))])
    )
    result = await agent.run('x')
    # Each answer records the candidate it came from, but without `sticky` the next step doesn't go to it.
    assert calls == ['primary', 'backup', 'primary', 'backup']
    assert _model_ids(result.all_messages()) == ['function:backup', 'function:backup']


async def test_a_conversation_the_step_model_answered_stays_on_it():
    calls: list[str] = []
    agent = Agent(_candidate('primary', calls), capabilities=[ModelFallback(_candidate('backup', calls), sticky=True)])
    first = await agent.run('x')
    assert _model_ids(first.all_messages()) == ['function:primary']
    await agent.run('y', message_history=first.all_messages())
    assert calls == ['primary', 'primary']


async def test_sticky_with_a_default_model_given_by_name():
    """The agent's model given as a string counts as its default, however a capability resolves it."""
    calls: list[str] = []
    primary = _candidate('primary', calls, fails=True)

    class Alias(AbstractCapability[None]):
        async def resolve_model_id(self, ctx: ModelResolutionContext[None], *, model_id: str) -> Model | None:
            return primary if model_id == 'my-primary' else None

    agent = _with_noop(
        Agent(
            'my-primary',
            deps_type=type(None),
            capabilities=[Alias(), ModelFallback(_candidate('backup', calls), sticky=True)],
        )
    )
    assert (await agent.run('x')).output == 'backup'
    assert calls == ['primary', 'backup', 'backup']


async def test_sticky_with_an_inferred_default_model():
    """A default model the agent infers from its name when it's constructed still counts as the default."""
    calls: list[str] = []
    backup = _candidate('backup', calls)
    agent = Agent('test', capabilities=[ModelFallback(backup, sticky=True)])
    result = await agent.run('y', message_history=_sticky_history(backup.model_id))
    assert result.output == 'backup'
    assert calls == ['backup']


async def test_sticky_follows_a_reassigned_default_model():
    """The default is the agent's model when the run starts, not when the capability was attached."""
    calls: list[str] = []
    backup = _candidate('backup', calls)
    agent = Agent(_candidate('old', calls), capabilities=[ModelFallback(backup, sticky=True)])
    agent.model = _candidate('new', calls)
    result = await agent.run('y', message_history=_sticky_history(backup.model_id))
    assert result.output == 'backup'
    assert calls == ['backup']


async def test_sticky_with_no_agent_model_replaces_the_first_candidate():
    calls: list[str] = []
    agent = _with_noop(
        Agent(
            capabilities=[
                ModelFallback(_candidate('first', calls, fails=True), _candidate('second', calls), sticky=True)
            ]
        )
    )
    assert (await agent.run('x')).output == 'second'
    assert calls == ['first', 'second', 'second']


@dataclass
class _StickyFailureCase:
    failing: tuple[str, ...]
    calls: list[str]
    answered_by: str


@pytest.mark.parametrize(
    'case',
    [
        pytest.param(_StickyFailureCase(('b',), ['b', 'c'], 'function:c'), id='moves-on-after-the-sticky-model'),
        pytest.param(
            _StickyFailureCase(('b', 'c'), ['b', 'c', 'primary'], 'function:primary'),
            id='comes-back-around-to-the-step-model',
        ),
        pytest.param(
            _StickyFailureCase(('b', 'c', 'primary'), ['b', 'c', 'primary', 'a'], 'function:a'),
            id='then-the-models-before-it',
        ),
    ],
)
async def test_a_failing_sticky_model_moves_on_down_the_rotated_chain(case: _StickyFailureCase):
    """The chain goes on after the sticky model, then comes back around to the step's model and the ones before it.

    An answer from the step's own model takes the conversation back to starting from it.
    """
    calls: list[str] = []
    a, b, c = (_candidate(name, calls, fails=name in case.failing) for name in 'abc')
    agent = Agent(
        _candidate('primary', calls, fails='primary' in case.failing),
        capabilities=[ModelFallback(a, b, c, sticky=True)],
    )
    result = await agent.run('y', message_history=_sticky_history(b.model_id))
    assert calls == case.calls
    assert _model_ids(result.new_messages()) == [case.answered_by]


async def test_a_sticky_chain_that_fails_everywhere_attempts_each_model_once():
    calls: list[str] = []
    a, b = _candidate('a', calls, fails=True), _candidate('b', calls, fails=True)
    agent = Agent(_candidate('primary', calls, fails=True), capabilities=[ModelFallback(a, b, a, sticky=True)])
    with pytest.raises(FallbackExceptionGroup) as exc_info:
        await agent.run('y', message_history=_sticky_history(b.model_id))
    assert [attempt.model_name for attempt in exc_info.value.attempts] == ['b', 'a', 'primary']


async def test_an_explicit_run_model_outranks_sticky():
    calls: list[str] = []
    agent = Agent(_candidate('primary', calls), capabilities=[ModelFallback(_candidate('backup', calls), sticky=True)])
    result = await agent.run(
        'y', model=_candidate('explicit', calls), message_history=_sticky_history('function:backup')
    )
    assert result.output == 'explicit'
    assert calls == ['explicit']


async def test_select_model_outranks_sticky():
    calls: list[str] = []
    selected = _candidate('selected', calls)
    agent = Agent(
        _candidate('primary', calls),
        capabilities=[SelectModel(lambda ctx: selected), ModelFallback(_candidate('backup', calls), sticky=True)],
    )
    await agent.run('y', message_history=_sticky_history('function:backup'))
    assert calls == ['selected']


@pytest.mark.parametrize('model_id', ['function:gone', 'function:primary'])
async def test_an_answer_from_a_model_that_is_not_another_candidate_starts_the_step_normally(model_id: str):
    """An answer from a model that's no longer a candidate, or from the step's own model, starts the step normally."""
    calls: list[str] = []
    primary = _candidate('primary', calls)
    agent = Agent(primary, capabilities=[ModelFallback(primary, _candidate('backup', calls), sticky=True)])
    assert (await agent.run('y', message_history=_sticky_history(model_id))).output == 'primary'
    assert calls == ['primary']


async def test_a_fallback_from_an_explicit_run_model_is_sticky_for_later_runs():
    """The answer records the candidate it came from, so a later run on the default model picks up from there."""
    calls: list[str] = []
    agent = Agent(_candidate('primary', calls), capabilities=[ModelFallback(_candidate('backup', calls), sticky=True)])
    first = await agent.run('x', model=_candidate('explicit', calls, fails=True))
    await agent.run('y', message_history=first.all_messages())
    assert calls == ['explicit', 'backup', 'backup']


async def test_sticky_resumes_a_suspended_response_on_its_pinned_model_and_stays_there():
    """A suspended response is still continued by the model that started it, which the conversation then sticks to."""
    calls: list[str] = []
    pinned = _candidate('pinned', calls)
    agent = Agent(
        _candidate('step', calls), capabilities=[ModelFallback(_candidate('other', calls), pinned, sticky=True)]
    )
    # The job was started on the candidate after the step's model failed, as its attempts record.
    history = _suspended_history(pinned.model_id)
    suspended = history[-1]
    assert isinstance(suspended, ModelResponse)
    suspended.failed_attempts = [_failed_attempt('step')]
    first = await agent.run(message_history=history)
    assert _model_ids(first.new_messages()) == ['function:pinned']
    await agent.run('y', message_history=first.all_messages())
    assert calls == ['pinned', 'pinned']


async def test_sticky_leaves_a_conversation_whose_suspended_response_the_step_model_started_on_it():
    """A suspended response the step's model started is continued there, and the conversation goes on there."""
    calls: list[str] = []
    step = _candidate('step', calls)
    agent = Agent(step, capabilities=[ModelFallback(_candidate('backup', calls), sticky=True)])
    first = await agent.run(message_history=_suspended_history(step.model_id))
    await agent.run('y', message_history=first.all_messages())
    assert calls == ['step', 'step']


async def test_a_fallback_model_that_runs_out_keeps_the_attempts_before_it():
    """A `FallbackModel` among `ModelFallback`'s models that exhausts its own chain lists the step's earlier attempts first."""
    agent = Agent(
        FunctionModel(failure, model_name='primary'),
        capabilities=[ModelFallback(FallbackModel(FunctionModel(failure, model_name='nested')))],
    )
    with pytest.raises(FallbackExceptionGroup) as exc_info:
        await agent.run('x')
    assert [attempt.model_name for attempt in exc_info.value.attempts] == ['primary', 'nested']


async def test_the_answer_that_crosses_a_token_limit_after_a_fallback_is_kept_as_interrupted():
    """A response a candidate answers with, which takes the run past a token limit, is kept in history as interrupted.

    Its attempts are kept on it, and the rejected attempt's tokens count towards the limit too.
    """

    def rejected(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('nope')], usage=RequestUsage(input_tokens=40))

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('hello')], usage=RequestUsage(input_tokens=40))

    agent = Agent(
        FunctionModel(rejected, model_name='primary'),
        capabilities=[ModelFallback(FunctionModel(answer, model_name='backup'), fallback_on=reject_nope)],
    )
    with capture_run_messages() as messages:
        with pytest.raises(UsageLimitExceeded):
            await agent.run('x', usage_limits=UsageLimits(input_tokens_limit=50))
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert (response.state, response.text, response.model_name) == ('interrupted', 'hello', 'backup')
    assert [(attempt.model_name, attempt.outcome) for attempt in response.failed_attempts or []] == [
        ('primary', 'rejected')
    ]


async def test_sticky_follows_an_answer_another_capability_retried_on_the_same_candidate():
    """A candidate's answer is followed even when another capability had it retried on that candidate."""
    calls: list[str] = []

    class RetryFirstAnswer(AbstractCapability[None]):
        """Rejects the first response it sees, retrying the same model."""

        rejected: bool = False

        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            if not self.rejected:
                self.rejected = True
                raise RetryModelRequest()
            return response

    agent = Agent(
        _candidate('primary', calls, fails=True),
        deps_type=type(None),
        capabilities=[RetryFirstAnswer(), ModelFallback(_candidate('backup', calls), sticky=True)],
    )
    first = await agent.run('x')
    assert _model_ids(first.new_messages()) == ['function:backup']
    await agent.run('y', message_history=first.all_messages())
    assert calls == ['primary', 'backup', 'backup', 'backup']


async def test_an_answer_from_a_model_chosen_for_its_run_does_not_stick():
    """`run(model=...)` picks a candidate for that run only: an answer no attempt failed before doesn't make it sticky."""
    calls: list[str] = []
    backup = _candidate('backup', calls)
    agent = Agent(_candidate('primary', calls), capabilities=[ModelFallback(backup, sticky=True)])
    first = await agent.run('x', model=backup)
    assert _model_ids(first.all_messages()) == ['function:backup']
    await agent.run('y', message_history=first.all_messages())
    assert calls == ['backup', 'primary']


async def test_a_conversation_stays_on_the_model_it_fell_back_to_until_the_primary_answers():
    """Sticky answers after the fallback keep the conversation there; once the primary answers again, it starts there."""
    calls: list[str] = []
    backup_fails = False

    def backup_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('backup')
        if backup_fails:
            raise ModelAPIError(model_name='backup', message='down')
        return ModelResponse(parts=[TextPart('backup')])

    agent = Agent(
        _candidate('primary', calls),
        capabilities=[ModelFallback(FunctionModel(backup_fn, model_name='backup'), sticky=True)],
    )
    history = _sticky_history('function:backup')
    for _ in range(2):
        history = (await agent.run('y', message_history=history)).all_messages()
    backup_fails = True
    history = (await agent.run('y', message_history=history)).all_messages()
    await agent.run('y', message_history=history)
    assert calls == ['backup', 'backup', 'backup', 'primary', 'primary']


async def test_a_later_answer_from_another_model_ends_the_sticky_streak():
    """Only the latest answers count: one from the primary after a fallback, like a run's own choice, ends the streak."""
    calls: list[str] = []
    agent = Agent(_candidate('primary', calls), capabilities=[ModelFallback(_candidate('backup', calls), sticky=True)])
    history = [
        *_sticky_history('function:backup'),
        ModelRequest(parts=[UserPromptPart('y')]),
        ModelResponse(parts=[TextPart('from primary')], model_id='function:primary'),
    ]
    await agent.run('z', message_history=history)
    assert calls == ['primary']


# --- Agent specs ---


def rate_limited(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelRateLimitError(model_name='m', message='slow down')


def spec_models() -> ResolveModelId[Any]:
    """Resolves the aliases a spec names to `FunctionModel`s, which a spec can't name directly."""
    models = {
        'failing': FunctionModel(failure, model_name='failing'),
        'rate-limited': FunctionModel(rate_limited, model_name='rate-limited'),
        'working': FunctionModel(success, model_name='working'),
    }
    return ResolveModelId(lambda ctx, model_id: models.get(model_id))


def attempted_models(result: AgentRunResult[Any]) -> list[str | None]:
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    return [attempt.model_name for attempt in response.failed_attempts or []] + [response.model_name]


@pytest.mark.parametrize(
    'spec',
    [
        pytest.param({'ModelFallback': ['failing', 'working']}, id='list'),
        pytest.param({'ModelFallback': {'models': ['failing', 'working']}}, id='mapping'),
        pytest.param(
            {'ModelFallback': {'models': ['failing', 'working'], 'fallback_on': ['ModelAPIError']}}, id='fallback_on'
        ),
    ],
)
async def test_spec_without_model_attempts_each_alias_in_turn(spec: dict[str, Any]):
    """A spec without `model:` has its first candidate supply the model, as in Python, with aliases resolved lazily."""
    agent = Agent.from_spec({'capabilities': [spec]}, capabilities=[spec_models()])
    result = await agent.run('x')
    assert result.output == 'hello'
    assert attempted_models(result) == ['failing', 'working']


async def test_spec_model_is_attempted_before_the_fallback():
    agent = Agent.from_spec(
        {'model': 'failing', 'capabilities': [{'ModelFallback': 'working'}]}, capabilities=[spec_models()]
    )
    result = await agent.run('x')
    assert attempted_models(result) == ['failing', 'working']

    # A run-level model still outranks the spec's, and falls back the same way.
    result = await agent.run('x', model='rate-limited')
    assert attempted_models(result) == ['rate-limited', 'working']


@pytest.mark.parametrize('fallback_on', ['ModelRateLimitError', ['ModelUnavailableError', 'ModelRateLimitError']])
async def test_spec_fallback_on_names_only_fall_back_on_those_errors(fallback_on: str | list[str]):
    agent = Agent.from_spec(
        {'capabilities': [{'ModelFallback': {'models': ['working'], 'fallback_on': fallback_on}}]},
        capabilities=[spec_models()],
    )
    result = await agent.run('x', model='rate-limited')
    assert attempted_models(result) == ['rate-limited', 'working']

    with pytest.raises(ModelAPIError, match='boom'):
        await agent.run('x', model='failing')


def test_spec_fallback_on_names_every_public_model_api_error():
    """Every public `ModelAPIError` class, and `ContentFilterError`, can be named in a spec, and nothing else can."""
    public_errors = {
        name
        for name in exceptions.__all__
        if isinstance(cls := getattr(exceptions, name), type) and issubclass(cls, ModelAPIError)
    }
    assert set(get_args(_SpecFallbackError)) == public_errors | {'ContentFilterError'}


@pytest.mark.parametrize(
    'spec, message',
    [
        pytest.param(
            {'ModelFallback': {'models': ['working'], 'fallback_on': ['ModelRateLimitError', 'UserError']}},
            '`ModelFallback` `fallback_on` in an agent spec takes names of `ModelAPIError` classes and '
            "`ContentFilterError`, got ['UserError']. Valid names: ModelAPIError, ModelHTTPError, ",
            id='not-a-model-api-error',
        ),
        pytest.param(
            {'ModelFallback': {'models': ['working'], 'fallback_on': 'TimeoutError'}},
            "got ['TimeoutError']",
            id='builtin',
        ),
        pytest.param(
            {'ModelFallback': {'models': ['working'], 'fallback_on': [{'ModelRateLimitError': None}]}},
            "got [{'ModelRateLimitError': None}]",
            id='not-a-name',
        ),
        pytest.param(
            {'ModelFallback': {'models': ['working'], 'fallback_on': []}},
            '`ModelFallback` created with an empty `fallback_on`',
            id='empty-fallback-on',
        ),
        pytest.param({'ModelFallback': []}, '`ModelFallback` requires at least one model', id='no-models'),
        pytest.param(
            {'ModelFallback': {'models': {'working': 'test'}}},
            "`ModelFallback` `models` in an agent spec must be a name or a list of names, got {'working': 'test'}.",
            id='models-mapping',
        ),
        pytest.param(
            {'ModelFallback': {'models': ['working'], 'fallback_on': {'ModelRateLimitError': False}}},
            '`ModelFallback` `fallback_on` in an agent spec must be a name or a list of names, '
            "got {'ModelRateLimitError': False}.",
            id='fallback-on-mapping',
        ),
        pytest.param(
            {'ModelFallback': ['working', {'name': 'other'}]},
            "`ModelFallback` in an agent spec takes model names, got [{'name': 'other'}].",
            id='not-a-model-name',
        ),
    ],
)
def test_spec_rejects_invalid_arguments(spec: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=re.escape(message)):
        Agent.from_spec({'model': 'test', 'capabilities': [spec]})


@pytest.mark.parametrize('suffix', ['yaml', 'json'])
@pytest.mark.parametrize(
    'capability',
    [
        pytest.param({'ModelFallback': ['failing', 'working']}, id='list'),
        pytest.param(
            {'ModelFallback': {'models': ['failing', 'working'], 'fallback_on': ['ModelAPIError']}}, id='mapping'
        ),
    ],
)
def test_spec_file_round_trip(tmp_path: Path, suffix: str, capability: dict[str, Any]):
    path = tmp_path / f'agent.{suffix}'
    AgentSpec.model_validate({'capabilities': [capability]}).to_file(path)

    loaded = AgentSpec.from_file(path)
    assert loaded.model_dump(include={'capabilities'}, context={'use_short_form': True}) == {
        'capabilities': [capability]
    }

    agent = Agent.from_file(path, capabilities=[spec_models()])
    result = agent.run_sync('x')
    assert attempted_models(result) == ['failing', 'working']


async def test_run_spec_adds_the_fallback():
    agent = Agent('failing', capabilities=[spec_models()])
    with pytest.raises(ModelAPIError, match='boom'):
        await agent.run('x')

    result = await agent.run('x', spec={'capabilities': [{'ModelFallback': ['working']}]})
    assert attempted_models(result) == ['failing', 'working']


async def test_override_spec_sets_the_fallback():
    agent = Agent(FunctionModel(failure, model_name='failing'))
    with agent.override(spec={'capabilities': [{'ModelFallback': ['test']}]}):
        result = await agent.run('x')
    assert result.output == 'success (no tool calls)'
    assert attempted_models(result) == ['failing', 'test']


# --- Refusals ---


def _refusal(name: str, calls: list[str], text: str | None = None) -> FunctionModel:
    """A model whose content filter ends every response, as a provider's safety refusal does."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(name)
        return ModelResponse(
            parts=[TextPart(text)] if text else [],
            finish_reason='content_filter',
            provider_details={'refusal': 'I cannot help with that.'},
            usage=RequestUsage(input_tokens=10, output_tokens=2),
        )

    return FunctionModel(respond, model_name=name)


def _answering(name: str, calls: list[str]) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(name)
        return ModelResponse(parts=[TextPart(f'{name} answered')])

    return FunctionModel(respond, model_name=name)


@pytest.mark.parametrize('text', [None, 'I cannot help with that.'], ids=['empty', 'with-text'])
async def test_falls_back_when_a_model_refuses_by_default(text: str | None):
    """A response the content filter ended falls back to the next model, whether or not it has partial text."""
    calls: list[str] = []
    agent = Agent(_refusal('primary', calls, text), capabilities=[ModelFallback(_answering('backup', calls))])
    result = await agent.run('x')
    assert result.output == 'backup answered'
    assert calls == ['primary', 'backup']
    assert [(a.model_name, a.outcome) for a in result.response.failed_attempts or []] == [('primary', 'rejected')]
    # The refused response was billed.
    assert result.usage.input_tokens == 10 + 51


async def test_a_refusal_ends_the_run_when_opted_out():
    """With `fallback_on=(ModelAPIError,)`, a refusal is accepted, so the run ends on `ContentFilterError` as before."""
    calls: list[str] = []
    agent = Agent(
        _refusal('primary', calls),
        capabilities=[ModelFallback(_answering('backup', calls), fallback_on=(ModelAPIError,))],
    )
    with pytest.raises(ContentFilterError, match='Content filter triggered'):
        await agent.run('x')
    assert calls == ['primary']


async def test_a_refusal_with_text_is_the_output_when_opted_out():
    """With `fallback_on=(ModelAPIError,)`, a refusal with text is the run's output, as without `ModelFallback`."""
    calls: list[str] = []
    agent = Agent(
        _refusal('primary', calls, 'I cannot help with that.'),
        capabilities=[ModelFallback(_answering('backup', calls), fallback_on=(ModelAPIError,))],
    )
    assert (await agent.run('x')).output == 'I cannot help with that.'
    assert calls == ['primary']


async def test_only_content_filter_error_names_refusals():
    """A supertype of `ContentFilterError`, like `UnexpectedModelBehavior` or `Exception`, doesn't name refusals."""
    calls: list[str] = []
    agent = Agent(
        _refusal('primary', calls),
        capabilities=[ModelFallback(_answering('backup', calls), fallback_on=(Exception,))],
    )
    with pytest.raises(ContentFilterError):
        await agent.run('x')
    assert calls == ['primary']


async def test_every_model_refusing_raises_the_group():
    calls: list[str] = []
    agent = Agent(_refusal('primary', calls), capabilities=[ModelFallback(_refusal('backup', calls))])
    with pytest.raises(FallbackExceptionGroup) as exc_info:
        await agent.run('x')
    assert [(a.model_name, a.outcome) for a in exc_info.value.attempts] == [
        ('primary', 'rejected'),
        ('backup', 'rejected'),
    ]


async def test_fallback_model_does_not_fall_back_on_refusals_by_default():
    """`FallbackModel` keeps its `(ModelAPIError,)` default, so a refusal ends the run as before."""
    calls: list[str] = []
    with pytest.raises(ContentFilterError):
        await Agent(FallbackModel(_refusal('primary', calls), _answering('backup', calls))).run('x')
    assert calls == ['primary']


@pytest.mark.parametrize('text', [None, 'I cannot help with that.'], ids=['empty', 'with-text'])
async def test_fallback_model_falls_back_on_refusals_when_named(text: str | None):
    """`FallbackModel` falls back on a refusal, empty or with text, when `fallback_on` names `ContentFilterError`."""
    calls: list[str] = []
    model = FallbackModel(
        _refusal('primary', calls, text), _answering('backup', calls), fallback_on=(ModelAPIError, ContentFilterError)
    )
    result = await Agent(model).run('x')
    assert result.output == 'backup answered'
    assert calls == ['primary', 'backup']


async def test_spec_falls_back_on_refusals_by_default_and_can_opt_out():
    calls: list[str] = []
    models = {'primary': _refusal('primary', calls), 'backup': _answering('backup', calls)}
    resolve = ResolveModelId(lambda ctx, model_id: models.get(model_id))

    agent = Agent.from_spec({'capabilities': [{'ModelFallback': ['backup']}]}, model='primary', capabilities=[resolve])
    assert (await agent.run('x')).output == 'backup answered'

    calls.clear()
    agent = Agent.from_spec(
        {'capabilities': [{'ModelFallback': {'models': ['backup'], 'fallback_on': ['ModelAPIError']}}]},
        model='primary',
        capabilities=[resolve],
    )
    with pytest.raises(ContentFilterError):
        await agent.run('x')
    assert calls == ['primary']

    calls.clear()
    agent = Agent.from_spec(
        {'capabilities': [{'ModelFallback': {'models': ['backup'], 'fallback_on': 'ContentFilterError'}}]},
        model='primary',
        capabilities=[resolve],
    )
    assert (await agent.run('x')).output == 'backup answered'


async def test_sticky_stays_on_the_model_that_answered_a_refusal():
    """A refusal that fell back makes the conversation sticky, like any other fallback."""
    calls: list[str] = []
    agent = _with_noop(
        Agent(_refusal('primary', calls), capabilities=[ModelFallback(_answering('backup', calls), sticky=True)])
    )
    first = await agent.run('x')
    assert first.output == 'backup answered'
    calls.clear()
    second = await agent.run('y', message_history=first.all_messages())
    assert second.output == 'backup answered'
    assert calls == ['backup']
