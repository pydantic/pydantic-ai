"""The `compact_history` capability hook, `RunContext.request_compaction()` and the compaction events.

Every test runs `Agent`s on `FunctionModel`s: what's under test is core's attempt loop, not a provider.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from typing import Any

import pydantic
import pytest

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    CompactionEndEvent,
    CompactionPart,
    CompactionRecord,
    CompactionStartEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PendingModelResponse,
    RunContext,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai._run_context import CompactionRequests
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.capabilities import (
    AbstractCapability,
    BatchModelRequests,
    CapabilityOrdering,
    CombinedCapability,
    CompactionRequest,
    CompactionResult,
    Hooks,
    ModelFallback,
    WrapModelRequestHandler,
    WrapperCapability,
    on_event,
)
from pydantic_ai.durable_exec._toolset import guard_run_context
from pydantic_ai.exceptions import (
    CallDeferred,
    ModelAPIError,
    ModelContextWindowExceededError,
    ModelRetry,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import LoadCapabilityCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models._batch_model import BatchedModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from ._inline_snapshot import snapshot
from .batch_utils import FAST_BATCH_SETTINGS, SlowBatchTestModel
from .conftest import IsDatetime
from .continuation_utils import ScriptedContinuationModel, scripted_response

COMPACTION_USAGE = RequestUsage(input_tokens=1000, output_tokens=50)

CompactionBuilder = Callable[[CompactionRequest], CompactionResult | None]
"""Builds a `compact_history` hook's result from its request."""


def summary_window(request: CompactionRequest, summary: str) -> tuple[list[ModelMessage], CompactionPart]:
    """A request-side summary in place of all but the latest exchange, so a tool call keeps its result."""
    part = CompactionPart(content=summary)
    return [ModelRequest(parts=[part]), *request.messages[-2:]], part


@dataclass
class Summarize(AbstractCapability[Any]):
    """Compacts the history into a summary naming the handler and the model it was for."""

    handler: str = 'summarize'
    calls: list[CompactionRequest] = field(default_factory=list[CompactionRequest])

    async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
        self.calls.append(request)
        messages, part = summary_window(request, f'{self.handler} for {request.model.model_name}')
        return CompactionResult(messages=messages, part=part, handler=self.handler, usage=COMPACTION_USAGE)


@dataclass
class Decline(AbstractCapability[Any]):
    """A `compact_history` hook that leaves the request to the next capability."""

    name: str
    asked: list[str]

    async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
        self.asked.append(self.name)
        return None


@dataclass
class Ordered(AbstractCapability[Any]):
    """Records being asked, and compacts."""

    name: str
    asked: list[str]
    position: CapabilityOrdering | None = None

    def get_ordering(self) -> CapabilityOrdering | None:
        return self.position

    async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
        self.asked.append(self.name)
        messages, part = summary_window(request, self.name)
        return CompactionResult(messages=messages, part=part, handler=self.name)


@dataclass
class RequestOnce(AbstractCapability[Any]):
    """Requests compaction from `before_model_request` on the first step."""

    focus: str | None = None

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if ctx.run_step == 1:
            ctx.request_compaction(reason='threshold', focus=self.focus)
        return request_context


@dataclass
class RecordEvents(AbstractCapability[Any]):
    """Records the compaction events, optionally cancelling the start."""

    cancel: bool = False
    events: list[CompactionStartEvent | CompactionEndEvent] = field(
        default_factory=list[CompactionStartEvent | CompactionEndEvent]
    )

    @on_event(CompactionStartEvent, CompactionEndEvent)
    async def _record(self, ctx: RunContext[Any], event: CompactionStartEvent | CompactionEndEvent) -> None:
        if self.cancel and isinstance(event, CompactionStartEvent):
            event.cancel('mid-activity')
        self.events.append(event)


@dataclass
class Responses(AbstractCapability[Any]):
    """Sums the input tokens of the responses the steps were answered with."""

    input_tokens: int = 0

    async def after_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        self.input_tokens += response.usage.input_tokens
        return response


def failure(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelAPIError(model_name='primary', message='rate limited')


def seen_by(seen: list[list[ModelMessage]]) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('hello')])

    return FunctionModel(respond, model_name='fallback')


async def test_tool_requests_compaction_of_the_next_request():
    """A tool's request compacts the next request only, which the model is sent and the history keeps."""
    summarize = Summarize()
    seen: list[list[ModelMessage]] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        if len(seen) == 1:
            return ModelResponse(parts=[ToolCallPart('compact_now', {})])
        if len(seen) == 2:
            return ModelResponse(parts=[ToolCallPart('noop', {})])
        return ModelResponse(parts=[TextPart('done')])

    responses = Responses()
    agent = Agent(FunctionModel(model, model_name='primary'), capabilities=[responses, summarize])

    @agent.tool
    def compact_now(ctx: RunContext[Any]) -> str:
        ctx.request_compaction(focus='the decisions')
        return 'compacting'

    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    result = await agent.run('hi')

    assert result.output == 'done'
    [request] = summarize.calls
    assert (request.reason, request.focus, request.model.model_name) == ('manual', 'the decisions', 'primary')
    assert isinstance(request.messages[-1], ModelRequest)
    assert [type(message).__name__ for message in request.messages] == ['ModelRequest', 'ModelResponse', 'ModelRequest']
    # The second request is sent the summary in place of the first exchange, and the third isn't compacted again.
    assert [type(part).__name__ for message in seen[1] for part in message.parts] == snapshot(
        ['CompactionPart', 'ToolCallPart', 'ToolReturnPart']
    )
    assert [type(part).__name__ for message in seen[2] for part in message.parts] == snapshot(
        ['CompactionPart', 'ToolCallPart', 'ToolReturnPart', 'ToolCallPart', 'ToolReturnPart']
    )
    assert result.all_messages()[0] == ModelRequest(parts=[CompactionPart(content='summarize for primary')])
    # The compaction is billed, but it isn't a request step.
    assert result.usage.requests == 3
    assert result.usage.input_tokens == COMPACTION_USAGE.input_tokens + responses.input_tokens


async def test_capability_order_is_the_fallback_chain():
    """Hooks are asked in capability order, through wrappers and combined capabilities; the first result wins."""
    asked: list[str] = []
    agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')])),
        capabilities=[
            RequestOnce(),
            WrapperCapability(Decline('wrapped', asked)),
            CombinedCapability([Decline('combined', asked), Ordered('first', asked), Ordered('second', asked)]),
            # Innermost, so it's asked last, and never reached.
            Ordered('innermost', asked, CapabilityOrdering(position='innermost')),
        ],
    )
    result = await agent.run('hi')
    assert asked == ['wrapped', 'combined', 'first']
    assert result.all_messages()[0] == ModelRequest(parts=[CompactionPart(content='first')])


async def test_capability_ordering_decides_who_is_asked_first():
    asked: list[str] = []
    agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')])),
        capabilities=[
            RequestOnce(),
            Ordered('listed first', asked),
            Ordered('outermost', asked, CapabilityOrdering(position='outermost')),
        ],
    )
    result = await agent.run('hi')
    assert asked == ['outermost']
    assert result.all_messages()[0] == ModelRequest(parts=[CompactionPart(content='outermost')])


async def test_no_capability_compacts():
    """Without a result the history is sent as is, and the request is dropped once the attempt is accepted."""
    asked: list[str] = []
    events = RecordEvents()

    agent = Agent(TestModel(), capabilities=[Decline('only', asked), events])

    @agent.tool
    def compact_now(ctx: RunContext[Any]) -> str:
        ctx.request_compaction()
        return 'compacting'

    result = await agent.run('hi')
    assert asked == ['only']
    assert not any(isinstance(part, CompactionPart) for m in result.all_messages() for part in m.parts)
    assert [type(event).__name__ for event in events.events] == ['CompactionStartEvent']


async def test_request_from_prepare_model_request_compacts_that_attempt():
    summarize = Summarize()

    async def prepare(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        ctx.request_compaction(reason='threshold')
        return request_context

    seen: list[list[ModelMessage]] = []
    agent = Agent(seen_by(seen), capabilities=[Hooks(prepare_model_request=prepare), summarize])
    await agent.run('hi')
    assert [request.reason for request in summarize.calls] == ['threshold']
    assert isinstance(seen[0][0].parts[0], CompactionPart)


async def test_discarded_attempt_compacts_again_for_the_next_model():
    """A compaction for a model that then fails is discarded, counted as rejected, and redone for the fallback."""
    summarize = Summarize()
    seen: list[list[ModelMessage]] = []
    rejected: list[ModelResponse] = []
    responses = Responses()

    class Ledger(AbstractCapability[Any]):
        async def wrap_model_request(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
        ) -> ModelResponse:
            try:
                return await handler(request_context)
            finally:
                rejected.extend(request_context._rejected_usage_responses)  # pyright: ignore[reportPrivateUsage]

    agent = Agent(
        FunctionModel(failure, model_name='primary'),
        capabilities=[Ledger(), responses, RequestOnce(), summarize, ModelFallback(seen_by(seen))],
    )
    result = await agent.run('hi')

    assert [request.model.model_name for request in summarize.calls] == ['primary', 'fallback']
    assert seen[0][0].parts[0] == CompactionPart(content='summarize for fallback')
    assert [m for m in result.all_messages() if any(isinstance(p, CompactionPart) for p in m.parts)] == [
        ModelRequest(parts=[CompactionPart(content='summarize for fallback')])
    ]
    # The primary's compaction was billed, and is counted like a rejected attempt's response.
    assert [(response.model_name, response.usage) for response in rejected] == [('primary', COMPACTION_USAGE)]
    assert result.usage.input_tokens == 2 * COMPACTION_USAGE.input_tokens + responses.input_tokens


@dataclass
class FailFor(AbstractCapability[Any]):
    """Fails compacting for one model, like a native compaction call that was rate limited."""

    model_name: str

    async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
        if request.model.model_name == self.model_name:
            raise ModelAPIError(model_name=self.model_name, message='compaction rate limited')
        return None


async def test_compaction_failure_falls_back():
    """A `ModelAPIError` from the hooks fails the attempt, so the request falls back without calling the model."""
    called: list[str] = []

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        called.append('primary')  # pragma: no cover
        return ModelResponse(parts=[TextPart('primary')])  # pragma: no cover

    summarize = Summarize()
    seen: list[list[ModelMessage]] = []
    agent = Agent(
        FunctionModel(primary, model_name='primary'),
        capabilities=[
            RequestOnce(),
            FailFor('primary'),
            summarize,
            ModelFallback(seen_by(seen)),
        ],
    )
    result = await agent.run('hi')
    assert result.output == 'hello'
    assert called == []
    assert [request.model.model_name for request in summarize.calls] == ['fallback']
    [response] = [m for m in result.all_messages() if isinstance(m, ModelResponse)]
    assert response.failed_attempts is not None and [a.model_name for a in response.failed_attempts] == ['primary']


async def test_compaction_failure_reaches_on_model_request_error():
    """The error hooks see a failed compaction as the attempt's error, and can recover from it."""

    async def recover(ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception) -> ModelResponse:
        return ModelResponse(parts=[TextPart(f'recovered from {error}')])

    agent = Agent(
        FunctionModel(failure, model_name='primary'),
        capabilities=[RequestOnce(), FailFor('primary'), Hooks(model_request_error=recover)],
    )
    result = await agent.run('hi')
    assert result.output == 'recovered from compaction rate limited'


async def test_compaction_failure_without_fallback_fails_the_run():
    agent = Agent(FunctionModel(failure, model_name='primary'), capabilities=[RequestOnce(), FailFor('primary')])
    with pytest.raises(ModelAPIError, match='compaction rate limited'):
        await agent.run('hi')


async def test_streamed_compaction_failure_falls_back():
    async def primary_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise AssertionError('not called')  # pragma: no cover
        yield ''  # pragma: no cover

    async def fallback_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        assert isinstance(messages[0].parts[0], CompactionPart)
        yield 'hello'

    summarize = Summarize()
    agent = Agent(
        FunctionModel(stream_function=primary_stream, model_name='primary'),
        capabilities=[
            RequestOnce(),
            FailFor('primary'),
            summarize,
            ModelFallback(FunctionModel(stream_function=fallback_stream, model_name='fallback')),
        ],
    )
    async with agent.run_stream('hi') as result:
        assert await result.get_output() == 'hello'
    assert [request.model.model_name for request in summarize.calls] == ['fallback']
    assert result.all_messages()[0] == ModelRequest(parts=[CompactionPart(content='summarize for fallback')])


async def test_streamed_request_is_compacted():
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield f'saw {len(messages)} messages'

    agent = Agent(FunctionModel(stream_function=stream), capabilities=[RequestOnce(), Summarize()])
    async with agent.run_stream('hi') as result:
        assert await result.get_output() == 'saw 1 messages'
    assert result.all_messages()[0] == ModelRequest(parts=[CompactionPart(content='summarize for function::stream')])


async def test_events():
    """Listeners see the start before the hooks run, and the end after; stream consumers see both."""
    events = RecordEvents()
    streamed: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            if isinstance(event, CompactionStartEvent | CompactionEndEvent):
                streamed.append(event)

    previous = ModelResponse(parts=[TextPart('earlier')], usage=RequestUsage(input_tokens=200, output_tokens=20))
    agent = Agent(
        TestModel(custom_output_text='done'),
        capabilities=[RequestOnce(focus='decisions'), events, Summarize()],
    )
    await agent.run(
        'hi', message_history=[ModelRequest(parts=[UserPromptPart('before')]), previous], event_stream_handler=handler
    )
    assert events.events == snapshot(
        [
            CompactionStartEvent(
                reason='threshold',
                model_name='test',
                provider_name='test',
                messages_before=3,
                focus='decisions',
                tokens_before=220,
            ),
            CompactionEndEvent(
                reason='threshold',
                handler='summarize',
                model_name='test',
                provider_name='test',
                messages_before=3,
                messages_after=3,
                kept_from=1,
                tokens_before=220,
                summary_tokens=50,
                usage=RequestUsage(input_tokens=1000, output_tokens=50),
            ),
        ]
    )
    assert streamed == events.events


async def test_cancelled_start_skips_compaction():
    """A cancelled start skips the hooks for this attempt and drops the request."""
    summarize = Summarize()
    events = RecordEvents(cancel=True)
    agent = Agent(TestModel(), capabilities=[events, summarize])

    @agent.tool
    def compact_now(ctx: RunContext[Any]) -> str:
        ctx.request_compaction()
        return 'compacting'

    result = await agent.run('hi')
    assert summarize.calls == []
    assert [(type(e).__name__, isinstance(e, CompactionStartEvent) and e.cancel_reason) for e in events.events] == [
        ('CompactionStartEvent', 'mid-activity')
    ]
    assert not any(isinstance(part, CompactionPart) for m in result.all_messages() for part in m.parts)


def test_events_round_trip():
    adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    start = CompactionStartEvent(reason='manual', model_name='m', provider_name='p', messages_before=4)
    end = CompactionEndEvent(
        reason='manual',
        handler='h',
        model_name='m',
        provider_name='p',
        messages_before=4,
        messages_after=2,
        kept_from=1,
    )
    assert adapter.validate_python(adapter.dump_python(start)) == start
    assert adapter.validate_python(adapter.dump_python(end)) == end
    assert adapter.dump_python(start, mode='json')['event_kind'] == 'compaction_start'


@dataclass
class Returns(AbstractCapability[Any]):
    """Returns a result built from the request."""

    build: CompactionBuilder

    async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
        return self.build(request)


def _without_part(request: CompactionRequest) -> CompactionResult:
    return CompactionResult(messages=list(request.messages), part=CompactionPart(content='lost'), handler='h')


def _not_ending_in_request(request: CompactionRequest) -> CompactionResult:
    part = CompactionPart(content='summary')
    return CompactionResult(
        messages=[ModelRequest(parts=[part]), ModelResponse(parts=[TextPart('x')])], part=part, handler='h'
    )


def _empty_summary(request: CompactionRequest) -> CompactionResult:
    messages, part = summary_window(request, '  ')
    return CompactionResult(messages=messages, part=part, handler='h')


@pytest.mark.parametrize(
    ('build', 'error', 'match'),
    [
        (_without_part, UserError, "compacted by 'h' must contain its `CompactionResult.part`"),
        (_not_ending_in_request, UserError, "compacted by 'h' must end with the request being answered"),
        (_empty_summary, UnexpectedModelBehavior, "The compaction by 'h' produced an empty summary"),
    ],
)
async def test_result_is_checked(build: CompactionBuilder, error: type[Exception], match: str):
    agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')])),
        capabilities=[RequestOnce(), Returns(build)],
    )
    with pytest.raises(error, match=match):
        await agent.run('hi')


async def test_compaction_only_response_records_the_summary_size():
    """A provider's compaction response is stamped with the summary's size as its context size."""
    part = CompactionPart(provider_name='openai', provider_details={'encrypted_content': 'gAAA'})
    compaction = ModelResponse(parts=[part], usage=COMPACTION_USAGE, provider_name='openai')

    def native(request: CompactionRequest) -> CompactionResult:
        return CompactionResult(
            messages=[compaction, request.messages[-1]],
            part=part,
            handler='native',
            usage=compaction.usage,
        )

    seen: list[list[ModelMessage]] = []
    agent = Agent(seen_by(seen), capabilities=[RequestOnce(), Returns(native)])
    result = await agent.run('hi')
    assert seen[0][0] is compaction
    assert result.all_messages()[0] == snapshot(
        ModelResponse(
            parts=[CompactionPart(provider_name='openai', provider_details={'encrypted_content': 'gAAA'})],
            usage=RequestUsage(input_tokens=1000, output_tokens=50),
            timestamp=IsDatetime(),
            provider_name='openai',
            metadata={'__pydantic_ai__': {'context_input_tokens': 0}},
        )
    )


async def test_model_retry_keeps_the_compaction():
    """A response a hook retries stays in history, and so does the compacted history it answered."""
    retried: list[bool] = []

    async def after(
        ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        if not retried:
            retried.append(True)
            raise ModelRetry('again')
        return response

    summarize = Summarize()
    agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')])),
        capabilities=[RequestOnce(), summarize, Hooks(after_model_request=after)],
    )
    result = await agent.run('hi')
    assert len(summarize.calls) == 1
    assert [type(m).__name__ for m in result.all_messages()] == snapshot(
        ['ModelRequest', 'ModelRequest', 'ModelResponse', 'ModelRequest', 'ModelResponse']
    )
    assert isinstance(result.all_messages()[0].parts[0], CompactionPart)


async def test_retried_error_keeps_the_request_pending():
    """An attempt retried with a retry prompt after its model failed sends no response, so it compacts again next step."""
    errors: list[Exception] = []

    async def retry(ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception) -> ModelResponse:
        errors.append(error)
        raise ModelRetry('try again')

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not errors:
            raise ModelAPIError(model_name='m', message='overloaded')
        return ModelResponse(parts=[TextPart('done')])

    summarize = Summarize()
    agent = Agent(FunctionModel(model), capabilities=[RequestOnce(), summarize, Hooks(model_request_error=retry)])
    result = await agent.run('hi')
    assert result.output == 'done'
    assert len(summarize.calls) == 2
    assert isinstance(result.all_messages()[0].parts[0], CompactionPart)


async def test_request_made_during_the_attempt_stays_pending():
    """Accepting an attempt settles the request it compacted for, not one made while it ran."""

    async def after(
        ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        if ctx.run_step == 1:
            ctx.request_compaction(reason='threshold')
        return response

    summarize = Summarize()
    agent = Agent(TestModel(), capabilities=[RequestOnce(), summarize, Hooks(after_model_request=after)])

    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    await agent.run('hi')
    assert [request.reason for request in summarize.calls] == ['threshold', 'threshold']


async def test_compaction_usage_is_checked_before_the_request():
    """A compaction that uses up the token budget stops the run before the request is sent."""
    called: list[bool] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        called.append(True)  # pragma: no cover
        return ModelResponse(parts=[TextPart('done')])  # pragma: no cover

    agent = Agent(FunctionModel(model), capabilities=[RequestOnce(), Summarize()])
    with pytest.raises(UsageLimitExceeded, match='total_tokens_limit'):
        await agent.run('hi', usage_limits=UsageLimits(total_tokens_limit=500))
    assert called == []


async def test_resumed_suspended_response_is_not_compacted():
    """A continuation completes a turn the provider already has, so a request waits for the step after it."""
    summarize = Summarize()
    model = ScriptedContinuationModel(
        responses=[scripted_response(texts=['done'], provider_response_id='job-1', input_tokens=5, output_tokens=2)]
    )
    agent = Agent(model, capabilities=[RequestOnce(), summarize])
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('hi')]),
        scripted_response(
            texts=['partial '], state='suspended', provider_response_id='job-1', input_tokens=3, output_tokens=1
        ),
    ]
    result = await agent.run(message_history=history)
    assert summarize.calls == []
    assert result.output == 'done'


async def test_resumed_request_stays_out_of_new_messages():
    """Compacting a run resumed from history without a prompt keeps that request out of `new_messages()`."""
    agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')])),
        capabilities=[RequestOnce(), Summarize()],
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('earlier')]),
        ModelRequest(parts=[UserPromptPart('second')]),
    ]
    result = await agent.run(message_history=history)
    assert [type(m).__name__ for m in result.all_messages()] == [
        'ModelRequest',
        'ModelResponse',
        'ModelRequest',
        'ModelResponse',
    ]
    assert result.new_messages() == [result.all_messages()[-1]]


async def test_request_compaction_needs_a_run():
    ctx = RunContext[None](deps=None, model=FunctionModel(lambda m, i: ModelResponse(parts=[])), usage=RunUsage())
    with pytest.raises(UserError, match='`request_compaction` is only available during an agent run'):
        ctx.request_compaction()


def test_request_compaction_in_a_durable_unit():
    ctx = RunContext[None](
        deps=None,
        model=FunctionModel(lambda m, i: ModelResponse(parts=[])),
        usage=RunUsage(),
        _compaction_requests=CompactionRequests(),
    )
    guarded = guard_run_context(ctx, unit_noun='step', container_noun='workflow')
    with pytest.raises(UserError, match='`request_compaction` is not supported inside a durable step'):
        guarded.request_compaction()


# --- Context overflow ---


def overflowing_model(calls: list[str], name: str = 'primary', *, until_compacted: bool) -> FunctionModel:
    """A model that rejects requests for exceeding its context window: always, or until the history is compacted.

    It streams too, as a run whose capabilities listen to events streams its requests.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(name)
        if not until_compacted or not any(
            isinstance(part, CompactionPart) for message in messages for part in message.parts
        ):
            raise ModelContextWindowExceededError(name, f'context_length_exceeded {len(calls)}')
        return ModelResponse(parts=[TextPart('fits now')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        response = respond(messages, info)
        yield response.text or ''

    return FunctionModel(respond, stream_function=stream, model_name=name)


def overflows_until_compacted(calls: list[str], name: str = 'primary') -> FunctionModel:
    return overflowing_model(calls, name, until_compacted=True)


def always_overflows(calls: list[str], name: str = 'primary') -> FunctionModel:
    return overflowing_model(calls, name, until_compacted=False)


async def test_context_overflow_is_compacted_and_retried_on_the_same_model():
    calls: list[str] = []
    summarize = Summarize()
    events = RecordEvents()
    agent = Agent(overflows_until_compacted(calls), capabilities=[events, summarize])
    result = await agent.run('hi')

    assert result.output == 'fits now'
    assert calls == ['primary', 'primary']
    assert [(request.reason, request.model.model_name) for request in summarize.calls] == [
        ('context_overflow', 'primary')
    ]
    assert [(type(event).__name__, event.reason) for event in events.events] == [
        ('CompactionStartEvent', 'context_overflow'),
        ('CompactionEndEvent', 'context_overflow'),
    ]
    [record] = result.compactions
    assert (record.reason, record.handler, record.model_name) == ('context_overflow', 'summarize', 'primary')
    assert result.all_messages()[0] == ModelRequest(parts=[CompactionPart(content='summarize for primary')])
    # The overflow isn't a failed attempt of its own: the retry is part of the same attempt.
    assert result.response.failed_attempts is None


async def test_context_overflow_twice_raises_the_overflow():
    calls: list[str] = []
    summarize = Summarize()
    agent = Agent(always_overflows(calls), capabilities=[summarize])
    with pytest.raises(ModelContextWindowExceededError, match='context_length_exceeded 2'):
        await agent.run('hi')
    # Retried once, not again.
    assert calls == ['primary', 'primary']
    assert len(summarize.calls) == 1


async def test_context_overflow_without_a_compaction_raises_the_overflow():
    calls: list[str] = []
    asked: list[str] = []
    with pytest.raises(ModelContextWindowExceededError, match='context_length_exceeded 1'):
        await Agent(always_overflows(calls), capabilities=[Decline('only', asked)]).run('hi')
    assert (calls, asked) == (['primary'], ['only'])

    # Without any `compact_history` hook, there's nothing to ask and no event.
    events = RecordEvents()
    with pytest.raises(ModelContextWindowExceededError):
        await Agent(always_overflows(calls), capabilities=[events]).run('hi')
    assert events.events == []


async def test_failed_context_overflow_compaction_raises_the_overflow_with_a_note():
    calls: list[str] = []
    agent = Agent(always_overflows(calls), capabilities=[FailFor('primary')])
    with pytest.raises(ModelContextWindowExceededError) as exc_info:
        await agent.run('hi')
    assert calls == ['primary']
    assert exc_info.value.__notes__ == [
        "Compacting the request to retry it failed: ModelAPIError('compaction rate limited')"
    ]


async def test_cancelled_context_overflow_compaction_raises_the_overflow():
    calls: list[str] = []
    summarize = Summarize()
    agent = Agent(always_overflows(calls), capabilities=[RecordEvents(cancel=True), summarize])
    with pytest.raises(ModelContextWindowExceededError):
        await agent.run('hi')
    assert (calls, summarize.calls) == (['primary'], [])


async def test_context_overflow_retry_does_not_advance_the_fallback_chain():
    calls: list[str] = []
    summarize = Summarize()
    agent = Agent(
        overflows_until_compacted(calls),
        capabilities=[summarize, ModelFallback(overflows_until_compacted(calls, 'fallback'))],
    )
    result = await agent.run('hi')
    assert result.output == 'fits now'
    assert calls == ['primary', 'primary']
    assert [request.model.model_name for request in summarize.calls] == ['primary']


async def test_context_overflow_after_the_retry_falls_back_as_before():
    """A second overflow reaches the error hooks, so `ModelFallback` moves on as it does without compaction."""
    calls: list[str] = []
    summarize = Summarize()
    agent = Agent(
        always_overflows(calls),
        capabilities=[summarize, ModelFallback(seen_by([]))],
    )
    result = await agent.run('hi')
    assert result.output == 'hello'
    assert calls == ['primary', 'primary']
    # The primary's compaction was discarded with its attempt, and the fallback was sent the history as it was.
    assert result.compactions == []
    assert not any(isinstance(part, CompactionPart) for message in result.all_messages() for part in message.parts)


async def test_context_overflow_of_a_compacted_attempt_compacts_again():
    """An attempt that was already compacted and still overflows is compacted again, and only the retry is kept."""
    calls: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('m')
        if len(calls) == 1:
            raise ModelContextWindowExceededError('m', 'context_length_exceeded')
        return ModelResponse(parts=[TextPart('fits now')])

    summarize = Summarize()
    result = await Agent(FunctionModel(respond), capabilities=[RequestOnce(), summarize]).run('hi')
    assert [request.reason for request in summarize.calls] == ['threshold', 'context_overflow']
    assert [record.reason for record in result.compactions] == ['context_overflow']


async def test_streamed_context_overflow_is_compacted_and_retried():
    calls: list[str] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        calls.append('stream')
        if not any(isinstance(part, CompactionPart) for message in messages for part in message.parts):
            raise ModelContextWindowExceededError('m', 'context_length_exceeded')
        yield 'fits now'

    summarize = Summarize()
    agent = Agent(FunctionModel(stream_function=stream), capabilities=[summarize])
    async with agent.run_stream('hi') as result:
        assert await result.get_output() == 'fits now'
    assert calls == ['stream', 'stream']
    assert [record.reason for record in result.compactions] == ['context_overflow']


async def test_streamed_context_overflow_twice_raises_the_overflow():
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise ModelContextWindowExceededError('m', 'context_length_exceeded')
        yield ''  # pragma: no cover

    agent = Agent(FunctionModel(stream_function=stream), capabilities=[Summarize()])
    with pytest.raises(ModelContextWindowExceededError):
        async with agent.run_stream('hi') as result:
            await result.get_output()  # pragma: no cover


# --- Compaction records ---


async def test_run_result_reports_the_kept_compactions():
    summarize = Summarize()
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('earlier')]),
    ]
    agent = Agent(TestModel(custom_output_text='done'), capabilities=[RequestOnce(), summarize])
    result = await agent.run('second', message_history=history)

    [record] = result.compactions
    [request] = summarize.calls
    assert record.replaced is request.messages
    assert (record.messages_before, record.messages_after, record.kept_from) == (3, 3, 1)
    # The last two messages were kept as they were, so the first one was summarized.
    assert record.summarized == [history[0]]
    assert record == snapshot(
        CompactionRecord(
            part=CompactionPart(content='summarize for test'),
            handler='summarize',
            reason='threshold',
            model_name='test',
            provider_name='test',
            messages_before=3,
            messages_after=3,
            kept_from=1,
        )
    )
    # `new_messages()` are this run's messages still in the compacted history.
    assert result.new_messages() == result.all_messages()[2:]


def store_run(
    log: list[ModelMessage | CompactionRecord], history: list[ModelMessage], result: AgentRunResult[Any]
) -> list[ModelMessage]:
    """The append-only store from the compaction docs."""
    position = len(history)
    for record in result.compactions:
        assert record.summarized is not None
        log.extend(record.summarized[position:])
        log.append(record)
        position = record.kept_from + max(0, position - len(record.summarized))
    log.extend(result.all_messages()[position:])
    return result.all_messages()


def describe(entry: ModelMessage | CompactionRecord) -> str:
    if isinstance(entry, CompactionRecord):
        return f'compaction: {entry.part.content}'
    part = entry.parts[0]
    if isinstance(part, UserPromptPart | TextPart):
        return f'{entry.kind}: {part.content}'
    return f'{entry.kind}: {part.part_kind}'


async def test_records_rebuild_the_full_conversation():
    """Storing each run's summarized messages before its compactions keeps every message exactly once."""
    compactions = 0

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        steps = sum(isinstance(message, ModelResponse) for message in messages)
        if steps < 3 and compactions < 2:
            return ModelResponse(parts=[ToolCallPart('compact_now', {})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model), capabilities=[Summarize()])

    @agent.tool
    def compact_now(ctx: RunContext[Any]) -> str:
        nonlocal compactions
        compactions += 1
        ctx.request_compaction()
        return f'compacting {compactions}'

    log: list[ModelMessage | CompactionRecord] = []
    history = store_run(log, [], await agent.run('first'))
    store_run(log, history, await agent.run('second', message_history=history))
    assert [describe(entry) for entry in log] == snapshot(
        [
            'request: first',
            'compaction: summarize for function:model:',
            'response: tool-call',
            'request: tool-return',
            'compaction: summarize for function:model:',
            'response: tool-call',
            'request: tool-return',
            'response: done',
            'request: second',
            'response: done',
        ]
    )


async def test_compactions_serialize_without_what_they_replaced():
    agent = Agent(TestModel(custom_output_text='done'), capabilities=[RequestOnce(), Summarize()])
    result = await agent.run('hi')
    adapter = pydantic.TypeAdapter(AgentRunResult[str])
    data = adapter.dump_python(result, mode='json')
    assert data['compactions'] == snapshot(
        [
            {
                'part': {
                    'content': 'summarize for test',
                    'id': None,
                    'provider_name': None,
                    'provider_details': None,
                    'metadata': None,
                    'part_kind': 'compaction',
                },
                'handler': 'summarize',
                'reason': 'threshold',
                'model_name': 'test',
                'provider_name': 'test',
                'messages_before': 1,
                'messages_after': 2,
                'kept_from': 1,
                'metadata': None,
            }
        ]
    )
    loaded = adapter.validate_python(data)
    [record] = loaded.compactions
    assert record == result.compactions[0]
    assert (record.replaced, record.summarized) == (None, None)

    # A run that didn't compact serializes as before.
    plain = await Agent(TestModel()).run('hi')
    assert 'compactions' not in adapter.dump_python(plain, mode='json')


async def test_discarded_compactions_are_not_reported():
    summarize = Summarize()
    agent = Agent(
        FunctionModel(failure, model_name='primary'),
        capabilities=[RequestOnce(), summarize, ModelFallback(seen_by([]))],
    )
    result = await agent.run('hi')
    assert len(summarize.calls) == 2
    assert [record.model_name for record in result.compactions] == ['fallback']


async def test_streamed_and_iterated_runs_report_compactions():
    agent = Agent(TestModel(custom_output_text='done'), capabilities=[RequestOnce(), Summarize()])
    async with agent.run_stream('hi') as streamed:
        await streamed.get_output()
        assert [record.handler for record in streamed.compactions] == ['summarize']
    assert streamed.result.compactions == streamed.compactions

    async with agent.iter('hi') as run:
        async for _ in run:
            pass
    assert [record.handler for record in run.compactions] == ['summarize']
    assert run.result is not None and run.result.compactions == run.compactions


async def test_streamed_run_ending_without_a_streamed_response_reports_compactions():
    """A streamed run whose output comes from deferred tool calls reports the compactions of its settled result."""
    agent = Agent(TestModel(), output_type=[str, DeferredToolRequests], capabilities=[RequestOnce(), Summarize()])

    @agent.tool_plain
    def later() -> str:
        raise CallDeferred

    async with agent.run_stream('hi') as streamed:
        assert isinstance(await streamed.get_output(), DeferredToolRequests)
        assert [record.handler for record in streamed.compactions] == ['summarize']


def test_sync_streamed_runs_report_compactions():
    agent = Agent(TestModel(custom_output_text='done'), capabilities=[RequestOnce(), Summarize()])
    result = agent.run_stream_sync('hi')
    result.get_output()
    assert [record.handler for record in result.compactions] == ['summarize']


def _explicit_kept_from(kept_from: int) -> CompactionBuilder:
    def build(request: CompactionRequest) -> CompactionResult:
        messages, part = summary_window(request, 'summary')
        return CompactionResult(
            messages=messages, part=part, handler='h', kept_from=kept_from, metadata={'kept': kept_from}
        )

    return build


def _rebuilt_tail(request: CompactionRequest) -> CompactionResult:
    """A summary followed by a rewritten copy of the previous response, then the request."""
    part = CompactionPart(content='summary')
    rewritten = ModelResponse(parts=[TextPart('rewritten')])
    return CompactionResult(
        messages=[ModelRequest(parts=[part]), rewritten, request.messages[-1]], part=part, handler='h'
    )


@pytest.mark.parametrize(('build', 'kept_from'), [(_explicit_kept_from(2), 2), (_rebuilt_tail, 2)])
async def test_kept_from(build: CompactionBuilder, kept_from: int):
    """`kept_from` is the handler's, or the longest tail after the summary kept as it was."""
    agent = Agent(TestModel(custom_output_text='done'), capabilities=[RequestOnce(), Returns(build)])
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('first')]), ModelResponse(parts=[TextPart('x')])]
    result = await agent.run('second', message_history=history)
    [record] = result.compactions
    assert record.kept_from == kept_from
    assert record.summarized == history
    # A handler's metadata is reported on the record.
    assert record.metadata == ({'kept': 2} if build is not _rebuilt_tail else None)


@pytest.mark.parametrize('kept_from', [0, 4])
async def test_kept_from_is_checked(kept_from: int):
    agent = Agent(TestModel(), capabilities=[RequestOnce(), Returns(_explicit_kept_from(kept_from))])
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('first')]), ModelResponse(parts=[TextPart('x')])]
    with pytest.raises(UserError, match="The `kept_from` of the history compacted by 'h' must be after"):
        await agent.run('second', message_history=history)


def _rebuilt_tail_claimed_kept(request: CompactionRequest) -> CompactionResult:
    return replace(_rebuilt_tail(request), kept_from=1)


def test_kept_tail_must_be_unchanged():
    agent = Agent(TestModel(), capabilities=[RequestOnce(), Returns(_rebuilt_tail_claimed_kept)])
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('first')]), ModelResponse(parts=[TextPart('x')])]
    with pytest.raises(UserError, match='must be the last messages of the history it was given, unchanged'):
        agent.run_sync('second', message_history=history)


# --- Batch requests ---


async def test_batch_requests_are_never_compacted_while_pending_and_compaction_is_not_batched():
    wrapped = SlowBatchTestModel(pending_checks=1)
    summarize = Summarize()

    async def every_step(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        ctx.request_compaction(reason='threshold')
        return request_context

    agent = Agent(
        wrapped,
        output_type=[str, PendingModelResponse],
        capabilities=[Hooks(before_model_request=every_step), summarize, BatchModelRequests(**FAST_BATCH_SETTINGS)],
    )
    pending = await agent.run('Hello')
    assert isinstance(pending.output, PendingModelResponse)
    # Compacted before the request was submitted, with the model the batch wrapper wraps, not the wrapper.
    [request] = summarize.calls
    assert request.model is wrapped and not isinstance(request.model, BatchedModel)

    stored = pending.all_messages()
    while isinstance((finished := await agent.run(message_history=stored)).output, PendingModelResponse):
        stored = finished.all_messages()
    assert finished.output == 'success (no tool calls)'
    # The step collecting the batch's result resumes a pending response, which is never compacted.
    assert len(summarize.calls) == 1


# --- Capability loads ---


class Weather(AbstractCapability[Any]):
    """A deferred capability with a tool."""

    def __init__(self) -> None:
        self.id = 'weather'
        self.description = 'Weather forecasts.'
        self.defer_loading = True

    def get_toolset(self) -> FunctionToolset[Any]:
        toolset = FunctionToolset[Any]()

        @toolset.tool_plain
        def get_forecast(city: str) -> str:
            return f'Sunny in {city}'  # pragma: no cover

        return toolset


def load_then(*calls: str) -> tuple[FunctionModel, list[list[str]]]:
    """A model that loads `weather`, then calls `calls` in turn, recording the tools each request offered."""
    offered: list[list[str]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.append(sorted(tool.name for tool in info.function_tools))
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step == 0:
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'weather'})])
        if step <= len(calls):
            return ModelResponse(parts=[ToolCallPart(calls[step - 1], {})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond), offered


def compaction_tools(agent: Agent[None, str]) -> None:
    @agent.tool
    def compact_now(ctx: RunContext[Any]) -> str:
        ctx.request_compaction()
        return 'compacting'

    @agent.tool_plain
    def noop() -> str:
        return 'ok'


def recorded_load_ids(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.args_as_dict()['id'])
        for message in messages
        for part in message.parts
        if isinstance(part, LoadCapabilityCallPart)
    ]


async def test_capability_loads_are_recorded_again_after_the_summary():
    """A deferred capability loaded before a compaction stays loaded, whichever capability compacted."""
    model, offered = load_then('compact_now')
    agent = Agent(model, capabilities=[Weather(), Summarize()])
    compaction_tools(agent)
    result = await agent.run('hi')

    messages = result.all_messages()
    assert isinstance(messages[0].parts[0], CompactionPart)
    # The load was summarized away, and recorded again right after the summary, ahead of the kept messages.
    assert recorded_load_ids(messages[1:3]) == ['weather']
    assert 'get_forecast' in offered[-1]
    [record] = result.compactions
    assert (record.messages_before, record.messages_after, record.kept_from) == (5, 5, 3)
    assert record.replaced is not None
    assert record.summarized == record.replaced[:3]


async def test_capability_loads_after_a_provider_compaction_end_the_kept_messages():
    """After a provider's compaction response, the records go after the requests following it, inside the kept messages."""

    def native(request: CompactionRequest) -> CompactionResult:
        part = CompactionPart(provider_name='openai', provider_details={'encrypted_content': 'gAAA'})
        return CompactionResult(
            messages=[ModelResponse(parts=[part], provider_name='openai'), *request.messages[-3:]],
            part=part,
            handler='native',
            kept_from=1,
        )

    model, offered = load_then('noop', 'compact_now')
    agent = Agent(model, capabilities=[Weather(), Returns(native)])
    compaction_tools(agent)
    result = await agent.run('hi')

    [record] = result.compactions
    # The kept request right after the compaction response now comes before the records, so the tail starts after them.
    assert (record.messages_before, record.messages_after, record.kept_from) == (7, 6, 4)
    assert recorded_load_ids(result.all_messages()[2:4]) == ['weather']
    assert 'get_forecast' in offered[-1]
