"""`ModelSettings['request_timeout']` and `ModelSettings['first_event_timeout']`: framework-enforced deadlines on one
request to one model.

These run against `FunctionModel`s and mock transports rather than cassettes: the behavior under test is what
happens when a request takes too long, which a recording can't reproduce on demand.
"""

from __future__ import annotations

import asyncio
import json
import pickle
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import anyio
import httpx2
import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    ModelMessage,
    ModelRequestTimeout,
    ModelResponse,
    RunTimedOut,
    TextPart,
    capture_run_messages,
)
from pydantic_ai.capabilities import AbstractCapability, ModelFallback
from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    BaseDurabilityCapability,
    DurabilityEngineSpec,
    DurableOperationBackend,
    DurableOperationId,
    JournalCallableOperationBackend,
    RoleBasedOperationConfig,
)
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import (
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    StreamedResponse,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCalls, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from .conftest import try_import
from .continuation_utils import ScriptedStreamedResponse, StreamSegment

with try_import() as openai_available:
    from openai import AsyncOpenAI

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider


SHORT_TIMEOUT = 0.05
"""For a request that never finishes: how soon it times out only affects how long the test takes."""

LONG_TIMEOUT = 10
"""For a test in which something must get done before the deadline, e.g. a first chunk reaching the consumer. Where the
deadline must then pass, the test moves the `clock` past it rather than waiting, so a loaded runner can't make the
deadline pass too soon."""


@dataclass
class Clock:
    """The event loop's clock, which a test moves forward to make a deadline pass exactly when it wants."""

    offset: float = 0

    def advance(self, seconds: float = LONG_TIMEOUT) -> None:
        self.offset += seconds


@pytest.fixture
async def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    loop = asyncio.get_running_loop()
    real_time = loop.time
    monkeypatch.setattr(loop, 'time', lambda: real_time() + clock.offset)
    return clock


async def hang(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    await anyio.sleep_forever()
    raise AssertionError('unreachable')  # pragma: no cover


async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('answer')])


async def stream_answer(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'answer'


@dataclass
class StreamThatHangs:
    """Streams one chunk, then waits for the next one forever, recording whether the stream was torn down.

    With a `clock`, the deadline passes once the next chunk is asked for, so after the first one was received.
    """

    clock: Clock | None = None
    closed: bool = False

    async def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        try:
            yield 'partial '
            if self.clock is not None:
                self.clock.advance()
            await anyio.sleep_forever()
            yield 'unreachable'  # pragma: no cover
        finally:
            self.closed = True


async def test_the_tighter_of_the_request_and_run_deadlines_wins():
    agent = Agent(FunctionModel(hang, model_name='slow'))

    with pytest.raises(ModelRequestTimeout):
        await agent.run('Hello', model_settings={'request_timeout': SHORT_TIMEOUT}, timeout=LONG_TIMEOUT)
    with pytest.raises(RunTimedOut):
        await agent.run('Hello', model_settings={'request_timeout': LONG_TIMEOUT}, timeout=SHORT_TIMEOUT)


async def test_request_timeout_raises_model_request_timeout():
    agent = Agent(FunctionModel(hang, model_name='slow'), model_settings={'request_timeout': SHORT_TIMEOUT})

    with pytest.raises(ModelRequestTimeout) as exc_info:
        await agent.run('hello')

    error = exc_info.value
    assert isinstance(error, ModelAPIError)
    assert error.model_name == 'slow'
    assert error.timeout == SHORT_TIMEOUT
    assert str(error) == snapshot("Request to model 'slow' timed out after 0.05 seconds")


async def test_request_timeout_not_reached():
    agent = Agent(FunctionModel(answer), model_settings={'request_timeout': 10})

    result = await agent.run('hello')

    assert result.output == 'answer'


async def test_request_timeout_from_model_settings_on_the_model():
    model = FunctionModel(hang, model_name='slow', settings={'request_timeout': SHORT_TIMEOUT})

    with pytest.raises(ModelRequestTimeout):
        await Agent(model).run('hello')


async def test_request_timeout_can_be_recovered_by_on_model_request_error():
    errors: list[Exception] = []

    class RecoverTimeout(AbstractCapability[Any]):
        async def on_model_request_error(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            errors.append(error)
            return ModelResponse(parts=[TextPart('recovered')])

    agent = Agent(
        FunctionModel(hang), model_settings={'request_timeout': SHORT_TIMEOUT}, capabilities=[RecoverTimeout()]
    )

    result = await agent.run('hello')

    assert result.output == 'recovered'
    assert [type(error) for error in errors] == [ModelRequestTimeout]


async def test_outer_cancellation_is_not_a_request_timeout():
    """A cancellation from outside the request propagates as one, rather than being reported as a timeout."""
    agent = Agent(FunctionModel(hang), model_settings={'request_timeout': 10})

    with anyio.move_on_after(SHORT_TIMEOUT) as scope:
        await agent.run('hello')

    assert scope.cancelled_caught


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
async def test_streamed_request_times_out_mid_stream(anyio_backend: str, clock: Clock):
    stream = StreamThatHangs(clock)
    agent = Agent(
        FunctionModel(stream_function=stream, model_name='slow'), model_settings={'request_timeout': LONG_TIMEOUT}
    )
    received: list[str] = []

    with capture_run_messages() as messages, pytest.raises(ModelRequestTimeout):
        async with agent.run_stream('hello') as result:
            async for text in result.stream_text(delta=True, debounce_by=None):
                received.append(text)

    assert received == ['partial ']
    assert stream.closed
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert (response.parts, response.model_name, response.state) == snapshot(
        ([TextPart(content='partial ')], 'slow', 'interrupted')
    )


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
async def test_streamed_request_times_out_with_event_stream_handler(anyio_backend: str, clock: Clock):
    """`run()` streams behind the scenes when there's an event stream handler, and the deadline covers that too."""
    stream = StreamThatHangs(clock)
    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[Any], stream_events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream_events:
            events.append(event)

    agent = Agent(FunctionModel(stream_function=stream), model_settings={'request_timeout': LONG_TIMEOUT})

    with pytest.raises(ModelRequestTimeout):
        await agent.run('hello', event_stream_handler=handler)

    assert stream.closed
    assert events


async def test_slow_stream_consumer_counts_towards_the_deadline():
    """The deadline covers reading the stream, so time the consumer takes between events counts too."""

    async def two_chunks(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'one '
        yield 'two'  # pragma: no cover

    agent = Agent(FunctionModel(stream_function=two_chunks), model_settings={'request_timeout': SHORT_TIMEOUT})

    with pytest.raises(ModelRequestTimeout):
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_text(delta=True, debounce_by=None):
                await anyio.sleep(SHORT_TIMEOUT * 4)


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
async def test_deadline_ends_with_the_last_chunk(anyio_backend: str, clock: Clock):
    """Once the stream has been read through its last chunk, the consumer can take as long as it likes."""
    agent = Agent(FunctionModel(stream_function=stream_answer), model_settings={'request_timeout': LONG_TIMEOUT})

    async with agent.run_stream('hello') as result:
        output = await result.get_output()
        clock.advance()
        await anyio.lowlevel.checkpoint()

    assert output == 'answer'


async def test_stream_deadline_starts_when_the_request_is_made():
    """The deadline is fixed when the request starts, so waiting to start reading counts towards it."""
    agent = Agent(FunctionModel(stream_function=stream_answer), model_settings={'request_timeout': SHORT_TIMEOUT})

    with pytest.raises(ModelRequestTimeout):
        async with agent.iter('hello') as run:
            first_node = run.next_node
            assert Agent.is_user_prompt_node(first_node)
            node = await run.next(first_node)
            assert Agent.is_model_request_node(node)
            async with node.stream(run.ctx) as stream:
                await anyio.sleep(SHORT_TIMEOUT * 4)
                async for _ in stream:
                    pass  # pragma: no cover


@pytest.mark.parametrize('wrapped', [False, True])
async def test_fallback_gives_each_model_a_fresh_deadline(wrapped: bool):
    """A timed-out model falls back by default, and the next model gets the full `request_timeout` again.

    That holds behind a wrapper model too, as durable execution puts one around the agent's model.
    """
    timeout = 0.2
    time_left: list[float] = []

    async def record_time_left(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        time_left.append(anyio.current_effective_deadline() - anyio.current_time())
        return ModelResponse(parts=[TextPart('fallback answer')])

    model: Model = FallbackModel(
        FunctionModel(hang, model_name='primary'), FunctionModel(record_time_left, model_name='fallback')
    )
    if wrapped:
        model = WrapperModel(model)
    agent = Agent(model, model_settings={'request_timeout': timeout})

    result = await agent.run('hello')

    assert result.output == 'fallback answer'
    # The primary used up a whole `request_timeout` of its own, and the fallback started with a fresh one.
    assert time_left[0] > timeout / 2
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.failed_attempts is not None
    assert [(attempt.model_name, attempt.outcome, attempt.error) for attempt in response.failed_attempts] == snapshot(
        [('primary', 'error', "ModelRequestTimeout: Request to model 'primary' timed out after 0.2 seconds")]
    )


async def test_fallback_raises_when_every_model_times_out():
    model = FallbackModel(FunctionModel(hang, model_name='primary'), FunctionModel(hang, model_name='fallback'))
    agent = Agent(model, model_settings={'request_timeout': SHORT_TIMEOUT})

    with pytest.raises(FallbackExceptionGroup) as exc_info:
        await agent.run('hello')

    assert [(type(e), e.model_name) for e in exc_info.value.exceptions if isinstance(e, ModelAPIError)] == [
        (ModelRequestTimeout, 'primary'),
        (ModelRequestTimeout, 'fallback'),
    ]


async def test_fallback_uses_each_models_own_request_timeout():
    model = FallbackModel(
        FunctionModel(hang, model_name='primary', settings={'request_timeout': SHORT_TIMEOUT}),
        FunctionModel(answer, model_name='fallback'),
    )

    result = await Agent(model).run('hello')

    assert result.output == 'answer'


async def test_fallback_falls_back_when_opening_a_stream_times_out():
    async def never_opens(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        await anyio.sleep_forever()
        yield 'unreachable'  # pragma: no cover

    model = FallbackModel(
        FunctionModel(stream_function=never_opens, model_name='primary', settings={'request_timeout': SHORT_TIMEOUT}),
        FunctionModel(stream_function=stream_answer, model_name='fallback'),
    )
    agent = Agent(model)

    async with agent.run_stream('hello') as result:
        output = await result.get_output()

    assert output == 'answer'
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.model_name == 'fallback'
    assert [attempt.model_name for attempt in response.failed_attempts or []] == ['primary']


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
async def test_fallback_stream_times_out_mid_stream_on_the_models_own_deadline(anyio_backend: str, clock: Clock):
    """Once a model's stream is open there's no falling back, and its own deadline still covers reading it."""
    stream = StreamThatHangs(clock)
    model = FallbackModel(
        FunctionModel(stream_function=stream, model_name='primary'),
        FunctionModel(stream_function=stream_answer, model_name='fallback'),
    )
    agent = Agent(model, model_settings={'request_timeout': LONG_TIMEOUT})

    with pytest.raises(ModelRequestTimeout) as exc_info:
        async with agent.run_stream('hello') as result:
            await result.get_output()

    assert exc_info.value.model_name == 'primary'
    assert stream.closed


class PollsForever(Model):
    """A background job that is still running at every poll, each of which takes half of a `LONG_TIMEOUT` deadline.

    A poll moves the `clock` rather than waiting, so the deadline passes during the second poll, the first continuation,
    however loaded the runner is. Once a poll has been cancelled, it is unavailable, so that the fallback that follows
    moves on to the next model.
    """

    def __init__(self, clock: Clock) -> None:
        super().__init__(settings={'request_timeout': LONG_TIMEOUT})
        self.clock = clock
        self.cancelled = False

    @property
    def model_name(self) -> str:
        return 'polls-forever'

    @property
    def system(self) -> str:
        return 'test'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        await self._poll()
        return self._still_running()

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        await self._poll()
        yield ScriptedStreamedResponse(
            model_request_parameters,
            _segment=StreamSegment(
                texts=['working'], state='suspended', provider_response_id='job', input_tokens=1, output_tokens=1
            ),
        )

    async def _poll(self) -> None:
        if self.cancelled:
            raise ModelAPIError(self.model_name, 'unavailable')
        self.clock.advance(LONG_TIMEOUT / 2)
        try:
            # A deadline that has passed cancels this wait before it ends.
            await anyio.sleep(0.01)
        except anyio.get_cancelled_exc_class():
            self.cancelled = True
            raise

    def _still_running(self) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart('working')], model_name=self.model_name, provider_response_id='job', state='suspended'
        )


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
@pytest.mark.parametrize('stream', [False, True])
async def test_fallback_keeps_the_picked_models_deadline_across_continuations(
    anyio_backend: str, clock: Clock, stream: bool
):
    """Each poll of a background job is a fresh call to the `FallbackModel`, but the same request to the model it
    picked: its deadline carries over, so a job that keeps running times out and falls back."""
    model = FallbackModel(
        PollsForever(clock), FunctionModel(answer, stream_function=stream_answer, model_name='fallback')
    )
    agent = Agent(model)

    if stream:
        async with agent.run_stream('hello') as result:
            output = await result.get_output()
        messages = result.all_messages()
    else:
        run_result = await agent.run('hello')
        output = run_result.output
        messages = run_result.all_messages()

    assert output == 'answer'
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert [(a.model_name, (a.error or '').split(':')[0]) for a in response.failed_attempts or []] == [
        ('polls-forever', 'ModelRequestTimeout'),
        ('polls-forever', 'ModelAPIError'),
    ]
    # The deadline was carried on the suspended responses only while the turn was being continued.
    assert all('request_deadline' not in str(message.metadata) for message in messages)


class PausesOnce(Model):
    """Pauses its response once, then completes it."""

    @property
    def model_name(self) -> str:
        return 'pauses-once'

    @property
    def system(self) -> str:
        return 'test'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        state = 'complete' if self._continuing(messages) else 'suspended'
        return ModelResponse(parts=[TextPart(state)], model_name=self.model_name, state=state)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        state = 'complete' if self._continuing(messages) else 'suspended'
        yield ScriptedStreamedResponse(
            model_request_parameters,
            _segment=StreamSegment(
                texts=[state], state=state, provider_response_id='', input_tokens=1, output_tokens=1
            ),
        )

    @staticmethod
    def _continuing(messages: list[ModelMessage]) -> bool:
        last = messages[-1]
        return isinstance(last, ModelResponse) and last.state == 'suspended'


@pytest.mark.parametrize('stream', [False, True])
async def test_carried_deadline_is_not_kept_in_history(stream: bool):
    """The deadline a `FallbackModel` carries on a suspended response is dropped once the turn completes."""
    model = FallbackModel(PausesOnce(), FunctionModel(answer, stream_function=stream_answer, model_name='fallback'))
    agent = Agent(model, model_settings={'request_timeout': 10})

    if stream:
        async with agent.run_stream('hello') as result:
            await result.get_output()
        messages = result.all_messages()
    else:
        messages = (await agent.run('hello')).all_messages()

    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert (response.state, response.metadata) == snapshot(
        ('complete', {'__pydantic_ai__': {'fallback_model_id': 'test:pauses-once'}})
    )


async def test_resumed_continuation_gets_a_fresh_deadline():
    """A later run resuming a suspended response from history is a new request, even if the deadline it carried
    has long passed."""

    async def answer_after_a_checkpoint(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await anyio.lowlevel.checkpoint()
        return ModelResponse(parts=[TextPart('answer')])

    pinned = FunctionModel(answer_after_a_checkpoint, model_name='pinned')
    suspended = ModelResponse(
        parts=[TextPart('working')],
        model_name='pinned',
        provider_response_id='job',
        state='suspended',
        metadata={
            '__pydantic_ai__': {'fallback_model_id': pinned.model_id, 'request_deadline': '2000-01-01T00:00:00+00:00'}
        },
    )
    model = FallbackModel(pinned, FunctionModel(answer, model_name='fallback'))
    agent = Agent(model, model_settings={'request_timeout': LONG_TIMEOUT})

    result = await agent.run(message_history=[ModelRequest.user_text_prompt('hello'), suspended])

    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert (response.model_name, response.failed_attempts) == ('pinned', None)


async def test_nested_fallback_gives_each_inner_model_a_deadline():
    inner = FallbackModel(
        FunctionModel(hang, model_name='inner-primary', settings={'request_timeout': SHORT_TIMEOUT}),
        FunctionModel(answer, model_name='inner'),
    )
    model = FallbackModel(inner, FunctionModel(answer, model_name='outer'))

    result = await Agent(model).run('hello')

    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.model_name == 'inner'


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
async def test_no_tasks_leak_after_timeout(anyio_backend: str):
    stream = StreamThatHangs()
    agent = Agent(FunctionModel(hang, stream_function=stream), model_settings={'request_timeout': SHORT_TIMEOUT})
    tasks_before = asyncio.all_tasks()

    with pytest.raises(ModelRequestTimeout):
        await agent.run('hello')
    with pytest.raises(ModelRequestTimeout):
        async with agent.run_stream('hello') as result:
            await result.get_output()

    assert asyncio.all_tasks() == tasks_before


async def test_direct_model_request():
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('hello')]

    with pytest.raises(ModelRequestTimeout):
        await model_request(FunctionModel(hang), messages, model_settings={'request_timeout': SHORT_TIMEOUT})

    response = await model_request(FunctionModel(answer), messages, model_settings={'request_timeout': 10})
    assert response.parts == [TextPart('answer')]


async def test_direct_model_request_stream():
    stream = StreamThatHangs()
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('hello')]
    settings: ModelSettings = {'request_timeout': SHORT_TIMEOUT}

    with pytest.raises(ModelRequestTimeout):
        async with model_request_stream(FunctionModel(stream_function=stream), messages, model_settings=settings) as sr:
            async for _ in sr:
                pass
    assert stream.closed

    async with model_request_stream(FunctionModel(stream_function=stream_answer), messages) as sr:
        events = [event async for event in sr]
    assert events


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
@pytest.mark.parametrize('anyio_backend', ['asyncio'])
@pytest.mark.parametrize('stream', [False, True])
async def test_request_timeout_covers_sdk_retries(
    allow_model_requests: None, anyio_backend: str, clock: Clock, stream: bool
):
    """Unlike `timeout`, an SDK retry doesn't re-arm `request_timeout`: it ends the request across all attempts."""
    attempts = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        # Each attempt takes half the deadline, well within `timeout`, so only a deadline spanning the retries passes.
        clock.advance(LONG_TIMEOUT / 2)
        # `retry-after-ms` keeps the SDK's backoff between attempts short.
        return httpx2.Response(503, json={'error': {'message': 'overloaded'}}, headers={'retry-after-ms': '1'})

    async with AsyncOpenAI(
        api_key='test',
        base_url='https://api.openai.com/v1',
        max_retries=1000,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    ) as openai_client:
        model = OpenAIChatModel('gpt-5', provider=OpenAIProvider(openai_client=openai_client))
        agent = Agent(model, model_settings={'request_timeout': LONG_TIMEOUT, 'timeout': LONG_TIMEOUT})

        with pytest.raises(ModelRequestTimeout):
            if stream:
                async with agent.run_stream('hello') as result:
                    await result.get_output()
            else:
                await agent.run('hello')

    assert attempts == 2


# The `ModelFallback` capability gives each candidate a deadline of its own, as `FallbackModel` does: every attempt is a
# model request of its own.


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
async def test_model_fallback_gives_each_candidate_a_fresh_deadline(anyio_backend: str, clock: Clock):
    time_left: list[float] = []

    async def hang_past_the_deadline(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        clock.advance()
        return await hang(messages, info)

    async def record_time_left(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        time_left.append(anyio.current_effective_deadline() - anyio.current_time())
        return ModelResponse(parts=[TextPart('fallback answer')])

    agent = Agent(
        FunctionModel(hang_past_the_deadline, model_name='primary'),
        model_settings={'request_timeout': LONG_TIMEOUT},
        capabilities=[ModelFallback(FunctionModel(record_time_left, model_name='fallback'))],
    )

    result = await agent.run('hello')

    assert result.output == 'fallback answer'
    assert time_left[0] > LONG_TIMEOUT / 2
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert [(attempt.model_name, attempt.outcome, attempt.error) for attempt in response.failed_attempts or []] == (
        snapshot([('primary', 'error', "ModelRequestTimeout: Request to model 'primary' timed out after 10 seconds")])
    )


async def test_model_fallback_uses_each_candidates_own_request_timeout():
    agent = Agent(
        FunctionModel(hang, model_name='primary', settings={'request_timeout': SHORT_TIMEOUT}),
        capabilities=[ModelFallback(FunctionModel(answer, model_name='fallback'))],
    )

    assert (await agent.run('hello')).output == 'answer'


async def test_model_fallback_falls_back_when_a_stream_times_out_before_its_first_event():
    """The deadline covers opening a candidate's stream and waiting for its first event."""

    async def stalls_before_first_event(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        yield {}  # opens the stream without producing an event
        await anyio.sleep_forever()

    agent = Agent(
        FunctionModel(
            stream_function=stalls_before_first_event, model_name='primary', settings={'request_timeout': SHORT_TIMEOUT}
        ),
        capabilities=[ModelFallback(FunctionModel(stream_function=stream_answer, model_name='fallback'))],
    )

    async with agent.run_stream('hello') as result:
        output = await result.get_output()

    assert output == 'answer'
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert [(attempt.model_name, attempt.error) for attempt in response.failed_attempts or []] == snapshot(
        [('primary', "ModelRequestTimeout: Request to model 'primary' timed out after 0.05 seconds")]
    )


# `ModelSettings['first_event_timeout']`: a deadline for a streamed request's first event, which no longer applies once
# that has arrived.


@dataclass
class StallsBeforeFirstEvent:
    """Opens its stream without producing an event, then waits forever, recording whether the stream was torn down.

    A `LONG_TIMEOUT` deadline passes while it waits.
    """

    clock: Clock
    closed: bool = False

    async def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        try:
            yield {}
            self.clock.advance()
            await anyio.sleep_forever()
        finally:
            self.closed = True


@dataclass
class PausesAfterFirstEvent:
    """Streams one chunk, then takes longer than a `LONG_TIMEOUT` deadline before streaming the rest."""

    clock: Clock

    async def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'partial '
        self.clock.advance(2 * LONG_TIMEOUT)
        # A deadline that has passed cancels this wait before it ends.
        await anyio.sleep(0.01)
        yield 'answer'


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
@pytest.mark.parametrize('consumer', ['run_stream', 'event_stream_handler'])
async def test_first_event_timeout_raises_model_request_timeout(anyio_backend: str, clock: Clock, consumer: str):
    """Without any fallback or error hook the stream opens lazily, and the deadline is enforced on it all the same."""
    stream = StallsBeforeFirstEvent(clock)
    agent = Agent(
        FunctionModel(stream_function=stream, model_name='slow'), model_settings={'first_event_timeout': LONG_TIMEOUT}
    )

    async def handler(ctx: RunContext[Any], stream_events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in stream_events:
            pass

    with pytest.raises(ModelRequestTimeout) as exc_info:
        if consumer == 'run_stream':
            async with agent.run_stream('hello'):
                pass  # pragma: no cover
        else:
            await agent.run('hello', event_stream_handler=handler)

    error = exc_info.value
    assert (error.model_name, error.timeout, error.setting, str(error)) == snapshot(
        (
            'slow',
            10,
            'first_event_timeout',
            "Request to model 'slow' timed out after 10 seconds without producing any output",
        )
    )
    unpickled = pickle.loads(pickle.dumps(error))
    assert (unpickled.model_name, unpickled.timeout, unpickled.setting) == ('slow', 10, 'first_event_timeout')
    assert stream.closed


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
async def test_first_event_timeout_no_longer_applies_after_the_first_event(anyio_backend: str, clock: Clock):
    agent = Agent(
        FunctionModel(stream_function=PausesAfterFirstEvent(clock)),
        model_settings={'first_event_timeout': LONG_TIMEOUT},
    )

    async with agent.run_stream('hello') as result:
        output = await result.get_output()

    assert output == 'partial answer'


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
async def test_first_event_timeout_does_not_apply_to_non_streamed_requests(anyio_backend: str, clock: Clock):
    async def slow_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        clock.advance()
        await anyio.sleep(0.01)
        return ModelResponse(parts=[TextPart('answer')])

    settings: ModelSettings = {'first_event_timeout': LONG_TIMEOUT / 2}

    result = await Agent(FunctionModel(slow_answer), model_settings=settings).run('hello')
    response = await model_request(
        FunctionModel(slow_answer), [ModelRequest.user_text_prompt('hello')], model_settings=settings
    )

    assert (result.output, response.parts) == ('answer', [TextPart('answer')])


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
async def test_direct_model_request_stream_first_event_timeout(anyio_backend: str, clock: Clock):
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('hello')]

    with pytest.raises(ModelRequestTimeout) as exc_info:
        async with model_request_stream(
            FunctionModel(stream_function=StallsBeforeFirstEvent(clock)),
            messages,
            model_settings={'first_event_timeout': LONG_TIMEOUT},
        ) as stream:
            async for _ in stream:
                pass  # pragma: no cover

    assert exc_info.value.setting == 'first_event_timeout'


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
@pytest.mark.parametrize(
    ('first_event_timeout', 'request_timeout', 'stalls', 'expected'),
    [
        pytest.param(LONG_TIMEOUT, 2 * LONG_TIMEOUT, 'before', 'first_event_timeout', id='first-event-passes-first'),
        pytest.param(2 * LONG_TIMEOUT, LONG_TIMEOUT, 'before', 'request_timeout', id='request-passes-first'),
        pytest.param(LONG_TIMEOUT / 2, LONG_TIMEOUT, 'after', 'request_timeout', id='request-after-first-event'),
    ],
)
async def test_whichever_of_first_event_and_request_timeout_passes_first_wins(
    anyio_backend: str, clock: Clock, first_event_timeout: float, request_timeout: float, stalls: str, expected: str
):
    stream_function = StallsBeforeFirstEvent(clock) if stalls == 'before' else StreamThatHangs(clock)
    agent = Agent(
        FunctionModel(stream_function=stream_function),
        model_settings={'first_event_timeout': first_event_timeout, 'request_timeout': request_timeout},
    )

    with pytest.raises(ModelRequestTimeout) as exc_info:
        async with agent.run_stream('hello') as result:
            await result.get_output()

    assert exc_info.value.setting == expected


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
@pytest.mark.parametrize('fallback', ['FallbackModel', 'ModelFallback'])
async def test_fallback_on_first_event_timeout(anyio_backend: str, clock: Clock, fallback: str):
    """Nothing has been streamed when the deadline passes, so both ways to fall back move on, with a fresh deadline."""
    primary = FunctionModel(stream_function=StallsBeforeFirstEvent(clock), model_name='primary')
    secondary = FunctionModel(stream_function=PausesAfterFirstEvent(clock), model_name='fallback')
    settings: ModelSettings = {'first_event_timeout': LONG_TIMEOUT}
    if fallback == 'FallbackModel':
        agent = Agent(FallbackModel(primary, secondary), model_settings=settings)
    else:
        agent = Agent(primary, model_settings=settings, capabilities=[ModelFallback(secondary)])

    async with agent.run_stream('hello') as result:
        output = await result.get_output()

    assert output == 'partial answer'
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert [(attempt.model_name, attempt.error) for attempt in response.failed_attempts or []] == snapshot(
        [
            (
                'primary',
                "ModelRequestTimeout: Request to model 'primary' timed out after 10 seconds without producing any output",
            )
        ]
    )


class StartsLate(Model):
    """A background job that's polled until it has finished, each poll taking half of a `LONG_TIMEOUT` deadline.

    Each poll streams the texts the next of `polls` lists; the last one completes the job. A poll moves the `clock`
    rather than waiting, so a deadline passes during the second poll however loaded the runner is. Once a poll has been
    cancelled, it is unavailable, so that a fallback that follows moves on to the next model.
    """

    def __init__(self, clock: Clock, polls: list[list[str]]) -> None:
        super().__init__()
        self.clock = clock
        self.polls = polls
        self.calls = 0
        self.cancelled = False

    @property
    def model_name(self) -> str:
        return 'starts-late'

    @property
    def system(self) -> str:
        return 'test'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        raise NotImplementedError  # pragma: no cover

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        if self.cancelled:
            raise ModelAPIError(self.model_name, 'unavailable')
        texts = self.polls[self.calls]
        self.calls += 1
        self.clock.advance(LONG_TIMEOUT / 2)
        try:
            # A deadline that has passed cancels this wait before it ends.
            await anyio.sleep(0.01)
        except anyio.get_cancelled_exc_class():
            self.cancelled = True
            raise
        state = 'complete' if self.calls == len(self.polls) else 'suspended'
        yield ScriptedStreamedResponse(
            model_request_parameters,
            _segment=StreamSegment(
                texts=texts, state=state, provider_response_id='job', input_tokens=1, output_tokens=1
            ),
        )


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
@pytest.mark.parametrize('fallback', [False, True])
async def test_first_event_timeout_covers_continuations_until_the_first_output(
    anyio_backend: str, clock: Clock, fallback: bool
):
    """A job that hasn't produced any output by the deadline times out, even though each poll is a separate request.

    A `FallbackModel` carries the deadline of the model it picked from one poll to the next, and falls back on it.
    """
    starts_late = StartsLate(clock, [[], [], ['answer']])
    model: Model = (
        FallbackModel(starts_late, FunctionModel(stream_function=stream_answer, model_name='fallback'))
        if fallback
        else starts_late
    )
    agent = Agent(model, model_settings={'first_event_timeout': LONG_TIMEOUT})

    if not fallback:
        with pytest.raises(ModelRequestTimeout) as exc_info:
            async with agent.run_stream('hello'):
                pass  # pragma: no cover
        assert exc_info.value.setting == 'first_event_timeout'
        return

    async with agent.run_stream('hello') as result:
        output = await result.get_output()

    assert output == 'answer'
    messages = result.all_messages()
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert [(a.model_name, (a.error or '').split(':')[0]) for a in response.failed_attempts or []] == [
        ('starts-late', 'ModelRequestTimeout'),
        ('starts-late', 'ModelAPIError'),
    ]
    assert all('first_event_deadline' not in str(message.metadata) for message in messages)


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
@pytest.mark.parametrize('fallback', [False, True])
async def test_first_event_timeout_no_longer_applies_to_continuations_after_the_first_output(
    anyio_backend: str, clock: Clock, fallback: bool
):
    starts_late = StartsLate(clock, [['working'], [], [], ['answer']])
    model: Model = (
        FallbackModel(starts_late, FunctionModel(stream_function=stream_answer, model_name='fallback'))
        if fallback
        else starts_late
    )
    agent = Agent(model, model_settings={'first_event_timeout': LONG_TIMEOUT})

    async with agent.run_stream('hello') as result:
        output = await result.get_output()

    assert output == 'answer'
    messages = result.all_messages()
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert (response.model_name, response.failed_attempts) == ('scripted', None)
    assert all('first_event_deadline' not in str(message.metadata) for message in messages)


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
@pytest.mark.parametrize('anyio_backend', ['asyncio'])
async def test_first_event_timeout_covers_sdk_retries(allow_model_requests: None, anyio_backend: str, clock: Clock):
    attempts = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        clock.advance(LONG_TIMEOUT / 2)
        return httpx2.Response(503, json={'error': {'message': 'overloaded'}}, headers={'retry-after-ms': '1'})

    async with AsyncOpenAI(
        api_key='test',
        base_url='https://api.openai.com/v1',
        max_retries=1000,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    ) as openai_client:
        model = OpenAIChatModel('gpt-5', provider=OpenAIProvider(openai_client=openai_client))
        agent = Agent(model, model_settings={'first_event_timeout': LONG_TIMEOUT})

        with pytest.raises(ModelRequestTimeout) as exc_info:
            async with agent.run_stream('hello') as result:
                await result.get_output()

    assert exc_info.value.setting == 'first_event_timeout'
    assert attempts == 2


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
@pytest.mark.parametrize('anyio_backend', ['asyncio'])
async def test_first_event_timeout_ignores_provider_events_without_output(
    allow_model_requests: None, anyio_backend: str, clock: Clock
):
    """OpenAI acknowledges a Responses request with `response.created` right away, which isn't output."""
    created: dict[str, Any] = {
        'type': 'response.created',
        'sequence_number': 0,
        'response': {
            'id': 'resp_1',
            'object': 'response',
            'created_at': 1700000000,
            'model': 'gpt-5',
            'output': [],
            'parallel_tool_calls': True,
            'tool_choice': 'auto',
            'tools': [],
            'status': 'in_progress',
        },
    }

    async def acknowledged_then_silent() -> AsyncIterator[bytes]:
        yield f'event: response.created\ndata: {json.dumps(created)}\n\n'.encode()
        clock.advance()
        await anyio.sleep_forever()
        yield b''  # pragma: no cover

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=acknowledged_then_silent())

    async with AsyncOpenAI(
        api_key='test',
        base_url='https://api.openai.com/v1',
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    ) as openai_client:
        model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=openai_client))
        agent = Agent(model, model_settings={'first_event_timeout': LONG_TIMEOUT})

        with pytest.raises(ModelRequestTimeout) as exc_info:
            async with agent.run_stream('hello') as result:
                await result.get_output()

    assert exc_info.value.setting == 'first_event_timeout'


# Under durable execution, a streamed request is consumed inside a durable unit, and the workflow only gets its final
# response. So the unit enforces the first-event deadline, and the workflow just `request_timeout`'s.


class _InlineBackend(JournalCallableOperationBackend[None]):
    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: None,
    ) -> object:
        return await body()


class InlineDurability(BaseDurabilityCapability[Any]):
    """A durability engine that runs each unit inline, on the same clock."""

    engine_spec = DurabilityEngineSpec(
        engine_name='Inline', durable_unit_noun='unit', durable_container_noun='workflow', codec=JSON_CODEC
    )

    @property
    def in_durable_context(self) -> bool:
        return True

    def get_durable_operation_backend(self) -> DurableOperationBackend[Any]:
        return _InlineBackend(
            agent_name='inline', config=RoleBasedOperationConfig(model=None, event=None, capability=None, tool=None)
        )


async def _drain(ctx: RunContext[Any], stream_events: AsyncIterable[AgentStreamEvent]) -> None:
    async for _ in stream_events:
        pass


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
@pytest.mark.parametrize(
    'settings',
    [
        pytest.param({'first_event_timeout': LONG_TIMEOUT}, id='first-event'),
        pytest.param({'first_event_timeout': LONG_TIMEOUT, 'request_timeout': 4 * LONG_TIMEOUT}, id='both'),
        pytest.param({'request_timeout': 4 * LONG_TIMEOUT}, id='request'),
    ],
)
async def test_durable_unit_enforces_first_event_timeout(anyio_backend: str, clock: Clock, settings: ModelSettings):
    """The workflow gets the response once the unit has consumed it, long after its first event, which isn't late."""
    agent = Agent(
        FunctionModel(stream_function=PausesAfterFirstEvent(clock)),
        name='durable',
        model_settings=settings,
        capabilities=[InlineDurability()],
    )

    result = await agent.run('hello', event_stream_handler=_drain)

    assert result.output == 'partial answer'


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
async def test_durable_unit_first_event_timeout_raises(anyio_backend: str, clock: Clock):
    stream = StallsBeforeFirstEvent(clock)
    agent = Agent(
        FunctionModel(stream_function=stream),
        name='durable',
        model_settings={'first_event_timeout': LONG_TIMEOUT},
        capabilities=[InlineDurability()],
    )

    with pytest.raises(ModelRequestTimeout) as exc_info:
        await agent.run('hello', event_stream_handler=_drain)

    assert exc_info.value.setting == 'first_event_timeout'
    assert stream.closed


@pytest.mark.parametrize('anyio_backend', ['asyncio'])
@pytest.mark.parametrize(
    ('polls', 'times_out'),
    [
        pytest.param([[], [], ['answer']], True, id='no-output-yet'),
        pytest.param([['working'], [], [], ['answer']], False, id='output-already'),
    ],
)
async def test_durable_units_carry_the_first_event_deadline_across_continuations(
    anyio_backend: str, clock: Clock, polls: list[list[str]], times_out: bool
):
    """Each poll is a unit of its own, and the first-event deadline carries over from one to the next."""
    agent = Agent(
        StartsLate(clock, polls),
        name='durable',
        model_settings={'first_event_timeout': LONG_TIMEOUT},
        capabilities=[InlineDurability()],
    )

    if times_out:
        with pytest.raises(ModelRequestTimeout) as exc_info:
            await agent.run('hello', event_stream_handler=_drain)
        assert exc_info.value.setting == 'first_event_timeout'
        return

    result = await agent.run('hello', event_stream_handler=_drain)

    assert result.output == 'answer'
    assert all('first_event_deadline' not in str(message.metadata) for message in result.all_messages())
