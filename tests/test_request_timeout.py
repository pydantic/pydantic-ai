"""`ModelSettings['request_timeout']`: one framework-enforced deadline over one request to one model.

These run against `FunctionModel`s and mock transports rather than cassettes: the behavior under test is what
happens when a request takes too long, which a recording can't reproduce on demand.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator
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
    TextPart,
    capture_run_messages,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import (
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    StreamedResponse,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from .conftest import try_import
from .continuation_utils import ScriptedStreamedResponse, StreamSegment

with try_import() as openai_available:
    from openai import AsyncOpenAI

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider


SHORT_TIMEOUT = 0.05
"""For a request that never finishes: how soon it times out only affects how long the test takes."""

GENEROUS_TIMEOUT = 0.25
"""For a test in which something must get done before the deadline, e.g. a first chunk reaching the consumer, on a
loaded runner."""


async def hang(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    await anyio.sleep_forever()
    raise AssertionError('unreachable')  # pragma: no cover


async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('answer')])


async def stream_answer(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'answer'


@dataclass
class StreamThatHangs:
    """Streams one chunk, then waits for the next one forever, recording whether the stream was torn down."""

    closed: bool = False

    async def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        try:
            yield 'partial '
            await anyio.sleep_forever()
            yield 'unreachable'  # pragma: no cover
        finally:
            self.closed = True


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


async def test_streamed_request_times_out_mid_stream():
    stream = StreamThatHangs()
    agent = Agent(
        FunctionModel(stream_function=stream, model_name='slow'), model_settings={'request_timeout': GENEROUS_TIMEOUT}
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


async def test_streamed_request_times_out_with_event_stream_handler():
    """`run()` streams behind the scenes when there's an event stream handler, and the deadline covers that too."""
    stream = StreamThatHangs()
    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[Any], stream_events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream_events:
            events.append(event)

    agent = Agent(FunctionModel(stream_function=stream), model_settings={'request_timeout': GENEROUS_TIMEOUT})

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


async def test_deadline_ends_with_the_last_chunk():
    """Once the stream has been read through its last chunk, the consumer can take as long as it likes."""
    agent = Agent(FunctionModel(stream_function=stream_answer), model_settings={'request_timeout': GENEROUS_TIMEOUT})

    async with agent.run_stream('hello') as result:
        output = await result.get_output()
        await anyio.sleep(GENEROUS_TIMEOUT + SHORT_TIMEOUT)

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


async def test_fallback_stream_times_out_mid_stream_on_the_models_own_deadline():
    """Once a model's stream is open there's no falling back, and its own deadline still covers reading it."""
    stream = StreamThatHangs()
    model = FallbackModel(
        FunctionModel(stream_function=stream, model_name='primary'),
        FunctionModel(stream_function=stream_answer, model_name='fallback'),
    )
    agent = Agent(model, model_settings={'request_timeout': GENEROUS_TIMEOUT})

    with pytest.raises(ModelRequestTimeout) as exc_info:
        async with agent.run_stream('hello') as result:
            await result.get_output()

    assert exc_info.value.model_name == 'primary'
    assert stream.closed


class PollsForever(Model):
    """A background job that is still running at every poll, each of which takes a while.

    Once a poll has been cancelled, it is unavailable, so that the fallback that follows moves on to the next model.
    """

    poll_time = SHORT_TIMEOUT / 5

    def __init__(self) -> None:
        super().__init__()
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
        try:
            await anyio.sleep(self.poll_time)
        except anyio.get_cancelled_exc_class():
            self.cancelled = True
            raise

    def _still_running(self) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart('working')], model_name=self.model_name, provider_response_id='job', state='suspended'
        )


@pytest.mark.parametrize('stream', [False, True])
async def test_fallback_keeps_the_picked_models_deadline_across_continuations(stream: bool):
    """Each poll of a background job is a fresh call to the `FallbackModel`, but the same request to the model it
    picked: its deadline carries over, so a job that keeps running times out and falls back."""
    model = FallbackModel(PollsForever(), FunctionModel(answer, stream_function=stream_answer, model_name='fallback'))
    agent = Agent(model, model_settings={'request_timeout': SHORT_TIMEOUT})

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
@pytest.mark.parametrize('stream', [False, True])
async def test_request_timeout_covers_sdk_retries(allow_model_requests: None, stream: bool):
    """Unlike `timeout`, an SDK retry doesn't re-arm `request_timeout`: it ends the request across all attempts."""
    attempts = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        await anyio.sleep(0.01)
        # `retry-after-ms` keeps the SDK's backoff between attempts short.
        return httpx2.Response(503, json={'error': {'message': 'overloaded'}}, headers={'retry-after-ms': '1'})

    async with AsyncOpenAI(
        api_key='test',
        base_url='https://api.openai.com/v1',
        max_retries=1000,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    ) as openai_client:
        model = OpenAIChatModel('gpt-5', provider=OpenAIProvider(openai_client=openai_client))
        agent = Agent(model, model_settings={'request_timeout': 0.2, 'timeout': 10})

        with pytest.raises(ModelRequestTimeout):
            if stream:
                async with agent.run_stream('hello') as result:
                    await result.get_output()
            else:
                await agent.run('hello')

    assert attempts > 1
