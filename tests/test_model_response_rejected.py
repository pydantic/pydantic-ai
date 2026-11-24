"""`ModelResponseRejectedEvent`: a streamed response that's rejected is reported before the response replacing it.

These run through `FunctionModel` rather than recorded provider responses: the rejection is decided by the agent
graph and its hooks after the model has answered, so no provider behavior is involved.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import anyio
import pytest

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    AgentStreamEvent,
    FinalResultEvent,
    ModelRequestAttempt,
    ModelResponse,
    ModelResponseRejectedEvent,
    ModelRetry,
    PartEndEvent,
    PartStartEvent,
    RequestUsage,
    RetryFeedbackPart,
    RetryInfo,
    RunContext,
    TextPart,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.capabilities import AbstractCapability, WrapModelRequestHandler
from pydantic_ai.exceptions import SkipModelRequest
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.output import OutputDataT
from pydantic_ai.run import AgentRun
from pydantic_ai.usage import RunUsage

from ._inline_snapshot import snapshot
from ._timeouts import READINESS_WAIT_TIMEOUT
from .conftest import IsDatetime, IsInstance, no_leaked_tasks
from .rejected_response_utils import (
    ANNOUNCEMENT,
    ANSWER,
    FEEDBACK,
    RejectAnnouncements,
    always_announce,
    announce_then_answer,
    reject_announcements,
)


@dataclass
class RetryAnnouncements(AbstractCapability[Any]):
    """Rejects the announcement from `wrap_model_request`, after its handler returned."""

    async def wrap_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        response = await handler(request_context)
        if response.text == ANNOUNCEMENT:
            raise ModelRetry(FEEDBACK)
        return response


def summarize(events: Sequence[AgentStreamEvent | AgentRunResultEvent[Any]]) -> list[object]:
    """Reduce a run's events to what a consumer sets apart: text it saw, rejections, and the output."""
    seen: list[object] = []
    for event in events:
        if isinstance(event, PartEndEvent) and isinstance(event.part, TextPart):
            seen.append(('text', event.part.content))
        elif isinstance(event, ModelResponseRejectedEvent):
            seen.append(('rejected', event.response.text, event.run_step, event.attempt))
        elif isinstance(event, AgentRunResultEvent):
            seen.append(('output', event.result.output))
    return seen


async def collect(agent: Agent[None, OutputDataT]) -> list[AgentStreamEvent | AgentRunResultEvent[OutputDataT]]:
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        async with agent.run_stream_events('When does the store open?') as stream:
            return [event async for event in stream]


async def test_output_validator_retry_is_reported_between_the_responses():
    agent = Agent(FunctionModel(stream_function=announce_then_answer))
    agent.output_validator(reject_announcements)

    events = await collect(agent)

    assert summarize(events) == [
        ('text', ANNOUNCEMENT),
        ('rejected', ANNOUNCEMENT, 1, 1),
        ('text', ANSWER),
        ('output', ANSWER),
    ]
    rejected = next(event for event in events if isinstance(event, ModelResponseRejectedEvent))
    assert rejected == snapshot(
        ModelResponseRejectedEvent(
            response=ModelResponse(
                parts=[TextPart(content="I'll look that up.")],
                usage=IsInstance(RequestUsage),
                model_name='function::announce_then_answer',
                model_id='function:function::announce_then_answer',
                timestamp=IsDatetime(),
                run_id=IsInstance(str),
                conversation_id=IsInstance(str),
            ),
            reason=RetryFeedbackPart(
                content='Answer the question.',
                retry=RetryInfo(cause='model_retry', attempt=1, max_retries=1),
                timestamp=IsDatetime(),
            ),
            run_step=1,
            attempt=1,
        )
    )
    # The reason is the very part the retry sends.
    run_result = events[-1]
    assert isinstance(run_result, AgentRunResultEvent)
    retry_request = next(
        message
        for message in run_result.result.all_messages()
        if isinstance(message, ModelRequest) and isinstance(message.parts[0], RetryFeedbackPart)
    )
    assert retry_request.parts == [rejected.reason]


async def test_response_without_output_is_reported():
    async def blank_then_answer(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield ANSWER if len(messages) > 1 else ''

    events = await collect(Agent(FunctionModel(stream_function=blank_then_answer)))

    assert summarize(events) == [('text', ''), ('rejected', '', 1, 1), ('text', ANSWER), ('output', ANSWER)]
    rejected = next(event for event in events if isinstance(event, ModelResponseRejectedEvent))
    assert isinstance(rejected.reason, RetryFeedbackPart)
    assert rejected.reason.retry.cause == 'no_output'


async def test_none_output_retry_is_reported():
    """An output that allows `None` validates a response without output, and its validator can still reject it."""

    async def blank_then_answer(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield ANSWER if len(messages) > 1 else ''

    agent = Agent(FunctionModel(stream_function=blank_then_answer), output_type=str | None)

    @agent.output_validator
    def require_an_answer(ctx: RunContext, output: str | None) -> str | None:
        if output is None:
            raise ModelRetry(FEEDBACK)
        return output

    events = await collect(agent)

    assert summarize(events) == [('text', ''), ('rejected', '', 1, 1), ('text', ANSWER), ('output', ANSWER)]


async def test_after_model_request_retry_is_reported_from_the_model_stream():
    """The rejection is reported before the model stream ends, so a consumer of the node's stream sees it there."""
    agent = Agent(FunctionModel(stream_function=announce_then_answer), capabilities=[RejectAnnouncements()])
    node_streams: list[list[AgentStreamEvent]] = []

    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        async with agent.iter('When does the store open?') as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        node_streams.append([event async for event in stream])

    assert [summarize(events) for events in node_streams] == [
        [('text', ANNOUNCEMENT), ('rejected', ANNOUNCEMENT, 1, 1)],
        [('text', ANSWER)],
    ]
    assert run.result is not None
    assert run.result.output == ANSWER


@pytest.mark.parametrize('source', ['output validator', 'after_model_request', 'wrap_model_request'])
@pytest.mark.parametrize('output_retries', [0, 1])
async def test_rejection_that_uses_up_the_retry_budget_is_reported_before_the_run_fails(
    source: str, output_retries: int
):
    """Every rejection is reported, the one that uses up the output retry budget included, before the run fails.

    A `wrap_model_request` retry decided after its handler returned has no next request to be reported from then, so
    the stream waits for its verdict once the budget is spent.
    """
    agent = Agent(
        FunctionModel(stream_function=always_announce),
        capabilities={
            'output validator': [],
            'after_model_request': [RejectAnnouncements()],
            'wrap_model_request': [RetryAnnouncements()],
        }[source],
        retries={'output': output_retries},
    )
    if source == 'output validator':
        agent.output_validator(reject_announcements)
    events: list[AgentStreamEvent | AgentRunResultEvent[str]] = []

    with pytest.raises(UnexpectedModelBehavior, match=rf'Exceeded maximum output retries \({output_retries}\)'):
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            async with agent.run_stream_events('When does the store open?') as stream:
                async for event in stream:
                    events.append(event)

    assert summarize(events) == [
        seen
        for run_step in range(1, output_retries + 2)
        for seen in (('text', ANNOUNCEMENT), ('rejected', ANNOUNCEMENT, run_step, 1))
    ]
    rejected = events[-1]
    assert isinstance(rejected, ModelResponseRejectedEvent)
    # Not stamped as a retry, as there is none left.
    assert rejected.reason == RetryFeedbackPart(
        content=FEEDBACK, retry=RetryInfo(cause='model_retry'), timestamp=IsDatetime()
    )


async def test_wrap_model_request_retry_is_reported_before_the_retried_response():
    """A retry `wrap_model_request` decides on after its handler returned is reported once the stream has ended."""
    events = await collect(
        Agent(FunctionModel(stream_function=announce_then_answer), capabilities=[RetryAnnouncements()])
    )

    assert summarize(events) == [
        ('text', ANNOUNCEMENT),
        ('rejected', ANNOUNCEMENT, 1, 1),
        ('text', ANSWER),
        ('output', ANSWER),
    ]


async def test_retry_a_wrapper_recovers_from_is_not_reported():
    @dataclass
    class AcceptAnyway(AbstractCapability[Any]):
        async def wrap_model_request(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
        ) -> ModelResponse:
            try:
                return await handler(request_context)
            except ModelRetry:
                return ModelResponse(parts=[TextPart(content='Recovered.')])

    # `AcceptAnyway` comes first, so it wraps the capability that rejects the announcement.
    agent = Agent(FunctionModel(stream_function=always_announce), capabilities=[AcceptAnyway(), RejectAnnouncements()])

    events = await collect(agent)

    assert summarize(events) == [('text', ANNOUNCEMENT), ('output', 'Recovered.')]


async def test_retry_a_wrapper_turns_into_an_error_is_not_reported():
    @dataclass
    class FailInstead(AbstractCapability[Any]):
        async def wrap_model_request(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
        ) -> ModelResponse:
            try:
                return await handler(request_context)
            except ModelRetry as e:
                raise UserError('Gave up.') from e

    agent = Agent(FunctionModel(stream_function=always_announce), capabilities=[FailInstead(), RejectAnnouncements()])
    events: list[AgentStreamEvent | AgentRunResultEvent[str]] = []

    with pytest.raises(UserError, match='Gave up'):
        async with agent.run_stream_events('When does the store open?') as stream:
            async for event in stream:
                events.append(event)

    assert summarize(events) == [('text', ANNOUNCEMENT)]


@dataclass
class CutShort(AbstractCapability[Any]):
    """Stops its handler after the model's first chunk, like a parallel input guardrail that blocks the prompt."""

    first_chunk_sent: asyncio.Event
    retry: bool = False
    calls: list[int] = field(default_factory=list[int])

    async def wrap_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        self.calls.append(ctx.run_step)
        if len(self.calls) > 1:
            return await handler(request_context)
        handler_task = asyncio.ensure_future(handler(request_context))
        await self.first_chunk_sent.wait()
        handler_task.cancel()
        await asyncio.gather(handler_task, return_exceptions=True)
        if self.retry:
            raise ModelRetry('Try again.')
        raise SkipModelRequest(ModelResponse(parts=[TextPart(content='Blocked.')]))


def stalls_after_first_chunk(first_chunk_sent: asyncio.Event):
    """Stalls after its first chunk on the first request, and answers on the next.

    A cut-short response isn't kept in history, so whether the first chunk was sent is what tells them apart.
    """

    async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        if first_chunk_sent.is_set():
            yield ANSWER
            return
        yield 'Leaked '
        first_chunk_sent.set()
        await asyncio.Event().wait()
        yield 'never'  # pragma: no cover

    return stream_function


async def test_response_a_wrapper_replaces_mid_stream_is_reported_before_the_replacement():
    first_chunk_sent = asyncio.Event()
    agent = Agent(
        FunctionModel(stream_function=stalls_after_first_chunk(first_chunk_sent)),
        capabilities=[CutShort(first_chunk_sent)],
    )

    events = await collect(agent)

    # The cut-short part never ends, so only what replaced it shows as text.
    assert [type(event).__name__ for event in events] == snapshot(
        [
            'PartStartEvent',
            'FinalResultEvent',
            'ModelResponseRejectedEvent',
            'PartStartEvent',
            'FinalResultEvent',
            'PartEndEvent',
            'AgentRunResultEvent',
        ]
    )
    assert isinstance(events[0], PartStartEvent) and isinstance(events[1], FinalResultEvent)
    rejected = events[2]
    assert rejected == snapshot(
        ModelResponseRejectedEvent(
            response=ModelResponse(
                parts=[TextPart(content='Leaked ')],
                usage=IsInstance(RequestUsage),
                model_name='function::stream_function',
                model_id='function:function::stream_function',
                timestamp=IsDatetime(),
                state='incomplete',
            ),
            reason=ModelRequestAttempt(
                model_name='function::stream_function',
                provider_name='function',
                outcome='rejected',
                timestamp=IsDatetime(),
                duration=IsInstance(timedelta),
                usage=IsInstance(RequestUsage),
            ),
            run_step=1,
            attempt=1,
        )
    )
    assert summarize(events[3:]) == [('text', 'Blocked.'), ('output', 'Blocked.')]


async def test_retry_a_wrapper_raises_mid_stream_is_reported_before_the_retried_response():
    first_chunk_sent = asyncio.Event()
    agent = Agent(
        FunctionModel(stream_function=stalls_after_first_chunk(first_chunk_sent)),
        capabilities=[CutShort(first_chunk_sent, retry=True)],
    )

    events = await collect(agent)

    assert summarize(events) == [('rejected', 'Leaked ', 1, 1), ('text', ANSWER), ('output', ANSWER)]
    rejected = next(event for event in events if isinstance(event, ModelResponseRejectedEvent))
    assert isinstance(rejected.reason, RetryFeedbackPart)
    assert rejected.reason.content == 'Try again.'


async def test_consumer_leaving_the_stream_early_still_gets_the_rejection_before_the_retry():
    """A consumer that stops reading before the model stream ends gets the rejection from the next request's stream."""
    agent = Agent(FunctionModel(stream_function=announce_then_answer), capabilities=[RejectAnnouncements()])
    node_streams: list[list[AgentStreamEvent]] = []

    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        async with agent.iter('When does the store open?') as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        events: list[AgentStreamEvent] = []
                        async for event in stream:  # pragma: no branch
                            events.append(event)
                            if isinstance(event, FinalResultEvent):
                                break
                        node_streams.append(events)

    assert [[type(event).__name__ for event in events] for events in node_streams] == snapshot(
        [
            ['PartStartEvent', 'FinalResultEvent'],
            ['ModelResponseRejectedEvent', 'PartStartEvent', 'FinalResultEvent'],
        ]
    )
    assert run.result is not None
    assert run.result.output == ANSWER


@dataclass
class AcceptEverything(AbstractCapability[Any]):
    """Has an `after_model_request` hook, so the model stream waits for its verdict, which accepts every response."""

    async def after_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        return response


@pytest.mark.parametrize('capabilities', [[], [AcceptEverything()]], ids=['no hooks', 'accepting hook'])
async def test_accepted_response_reports_nothing(capabilities: list[AbstractCapability[Any]]):
    events = await collect(Agent(FunctionModel(stream_function=always_announce), capabilities=capabilities))

    assert [type(event).__name__ for event in events] == [
        'PartStartEvent',
        'FinalResultEvent',
        'PartEndEvent',
        'AgentRunResultEvent',
    ]


async def test_run_stream_usage_counts_a_hooked_response_once():
    """The hooks pass their verdict once the model stream ends, so the response's usage is already counted then."""
    agent = Agent(FunctionModel(stream_function=always_announce), capabilities=[AcceptEverything()])

    async with agent.run_stream('When does the store open?') as result:
        assert await result.get_output() == ANNOUNCEMENT
        streamed_usage = result.usage

    assert streamed_usage == result.usage


@pytest.mark.parametrize(
    ('capabilities', 'counted_before_leaving'),
    [([], False), ([AcceptEverything()], True)],
    ids=['no after_model_request hook', 'after_model_request hook'],
)
async def test_response_usage_is_counted_when_after_model_request_passes_its_verdict(
    capabilities: list[AbstractCapability[Any]], counted_before_leaving: bool
):
    """With an `after_model_request` hook, the hooks pass their verdict when the model stream ends, so the run counts
    the response's usage then; without one, it's counted when the consumer leaves the stream, as it always was."""
    agent = Agent(FunctionModel(stream_function=always_announce), capabilities=capabilities)
    counted_after_exhausting: bool | None = None

    async with agent.iter('When does the store open?') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for _ in stream:
                        pass
                    counted_after_exhausting = run.usage.output_tokens > 0
                assert run.usage.output_tokens > 0

    assert counted_after_exhausting is counted_before_leaving


@pytest.mark.parametrize('capabilities', [[], [AcceptEverything()]], ids=['no hooks', 'accepting hook'])
async def test_consumer_raising_after_the_verdict_counts_the_response_once(capabilities: list[AbstractCapability[Any]]):
    """A consumer that raises after reading the whole stream commits the response as interrupted.

    With an `after_model_request` hook its usage was already counted when the hooks passed their verdict, so the
    interrupted commit doesn't count it again.
    """
    agent = Agent(FunctionModel(stream_function=always_announce), capabilities=capabilities)
    runs: list[AgentRun[None, str]] = []

    with pytest.raises(RuntimeError, match='consumer failed'):
        async with agent.iter('When does the store open?') as run:
            runs.append(run)
            async for node in run:  # pragma: no branch
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for _ in stream:
                            pass
                        raise RuntimeError('consumer failed')

    assert runs[0].usage == RunUsage(input_tokens=50, output_tokens=5, requests=1)


async def test_consumer_cancelled_while_the_hooks_pass_their_verdict_leaves_no_task_behind():
    """A consumer cancelled while its stream waits for the `after_model_request` verdict stops the hook and the request.

    The stream waits for the verdict once the model stream ends, so the cancellation lands in that wait, and the
    request step's teardown has to cancel the hook it is waiting on rather than leave it running.
    """
    verdict_started = anyio.Event()
    verdict_cancelled = anyio.Event()

    @dataclass
    class SlowVerdict(AbstractCapability[Any]):
        async def after_model_request(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            verdict_started.set()
            try:
                await anyio.sleep_forever()
            finally:
                verdict_cancelled.set()
            raise AssertionError('unreachable')  # pragma: no cover

    agent = Agent(FunctionModel(stream_function=always_announce), capabilities=[SlowVerdict()])
    events: list[AgentStreamEvent] = []

    with no_leaked_tasks(), anyio.fail_after(READINESS_WAIT_TIMEOUT):
        async with anyio.create_task_group() as tg:

            async def cancel_once_the_verdict_started() -> None:
                await verdict_started.wait()
                tg.cancel_scope.cancel()

            tg.start_soon(cancel_once_the_verdict_started)
            async with agent.iter('When does the store open?') as run:
                async for node in run:  # pragma: no branch
                    if Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as stream:
                            async for event in stream:  # pragma: no branch
                                events.append(event)

    assert tg.cancel_scope.cancelled_caught
    assert verdict_cancelled.is_set()
    # The whole response reached the consumer before the stream waited for the verdict.
    assert summarize(events) == [('text', ANNOUNCEMENT)]
