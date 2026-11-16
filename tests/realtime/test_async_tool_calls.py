"""Asynchronous tool calls in realtime history: the model keeps talking while the tool runs.

Gemini Live's `NON_BLOCKING` calls (`async_tool_calls=True`) let the model go on in the turn that made the call
("this might take a moment"), and the call's result cuts into that turn when it arrives. Its connection flags such a
call `ToolCall.response_continues`, and the `InferredLifecycle` keeps the calling response going past the call
until its terminal, or until the call's result goes out: the result must directly follow the response with its
call, as request-response APIs (Anthropic, OpenAI Chat) require, so whatever the model says after that is a
response of its own.

These run over scripted connections, whose lifecycle is inferred as Gemini's is, so each test can stage when the
result goes out relative to what the model says. The recorded Gemini exchange is
`test_google_ws.py::test_async_tool_speech_stays_before_its_result`; the simulator (`simulation/`) explores the
interleavings at random.
"""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SpeechPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.realtime import RealtimeSession as _RealtimeSession
from pydantic_ai.realtime.codec import (
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    ResponseDone,
    SessionUsage,
    ToolCall,
    ToolCallCancelled,
    ToolResult,
)
from pydantic_ai.usage import RequestUsage, UsageLimits

from ..conftest import try_import
from .test_session import (
    _LIVENESS_TIMEOUT,  # pyright: ignore[reportPrivateUsage]
    BlockingRealtimeConnection,
    FakeRealtimeConnection,
    RealtimeSession,
    _until,  # pyright: ignore[reportPrivateUsage]
    drain_events,
)

with try_import() as anthropic_imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as openai_imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider


def _call(call_id: str = 'bg_1', tool_name: str = 'slow') -> ToolCall:
    """An asynchronous call, as Gemini Live reports a `NON_BLOCKING` one."""
    return ToolCall(tool_call_id=call_id, tool_name=tool_name, args='{}', response_continues=True)


def _shape(messages: Sequence[ModelMessage]) -> list[tuple[str, list[str]]]:
    """Each message's kind and its parts, speech as its transcript, to read an order off at a glance."""
    return [
        (
            type(message).__name__,
            [
                part.transcript or ''
                if isinstance(part, SpeechPart)
                else part.content
                if isinstance(part, UserPromptPart) and isinstance(part.content, str)
                else type(part).__name__
                for part in message.parts
            ],
        )
        for message in messages
    ]


class _QueuedConnection(FakeRealtimeConnection):
    """A connection the test feeds event by event, so it can stage what the model says around what the session sends."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__([], **kwargs)
        self.queue: asyncio.Queue[RealtimeCodecEvent] = asyncio.Queue()

    def say(self, *events: RealtimeCodecEvent) -> None:
        for event in events:
            self.queue.put_nowait(event)

    def results(self) -> list[ToolResult]:
        return [content for content in self.sent if isinstance(content, ToolResult)]

    async def codec_events(self) -> AsyncIterator[RealtimeCodecEvent]:
        while True:
            yield await self.queue.get()


class _Tools:
    """Tools that each finish once the test lets them, returning their call id."""

    def __init__(self) -> None:
        self.release: dict[str, asyncio.Event] = {}

    def finish(self, call_id: str) -> None:
        self.release.setdefault(call_id, asyncio.Event()).set()

    async def runner(self, name: str, args: dict[str, Any], call_id: str) -> str:
        await self.release.setdefault(call_id, asyncio.Event()).wait()
        return f'result of {call_id}'


async def test_speech_after_the_call_is_recorded_before_its_result() -> None:
    """What the model says after the call, before its result, is recorded with the call, ahead of the result.

    As a response of its own it would land after the result, which directly follows its call: history would say the
    model spoke after it had the result.
    """
    conn = _QueuedConnection()
    tools = _Tools()
    session = RealtimeSession(conn, tools.runner)
    async with session:
        consumer = asyncio.create_task(drain_events(session))
        await session.send('Find me a flight.')
        conn.say(
            OutputTranscript(text='Let me look.', is_final=True),
            _call(),
            OutputTranscript(text='This might take a moment.', is_final=True),
        )
        await _until(lambda: session._core.speaking_part() is not None)  # pyright: ignore[reportPrivateUsage]
        # The model is still talking in the calling response, and the tool still runs.
        assert _shape(session.all_messages()) == snapshot([('ModelRequest', ['Find me a flight.'])])
        tools.finish('bg_1')
        await _until(lambda: bool(conn.results()))
        # Gemini cuts the turn off to take the result (`scheduling: INTERRUPT`), and reports its usage as it ends.
        conn.say(
            ResponseDone(interrupted=True),
            OutputTranscript(text='The cheapest is KLM.', is_final=True),
            SessionUsage(usage=RequestUsage(input_tokens=20, output_tokens=7)),
            ResponseDone(),
        )
        with anyio.fail_after(_LIVENESS_TIMEOUT):
            await session.wait_for_reply()
        await session.close()
        await consumer

    assert _shape(session.all_messages()) == snapshot(
        [
            ('ModelRequest', ['Find me a flight.']),
            ('ModelResponse', ['Let me look.', 'ToolCallPart', 'This might take a moment.']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelResponse', ['The cheapest is KLM.']),
        ]
    )
    calling = session.all_messages()[1]
    assert isinstance(calling, ModelResponse)
    assert (calling.state, calling.finish_reason) == snapshot(('complete', 'tool_call'))
    assert session.usage.requests == 2


async def test_answer_in_the_turn_the_result_cut_into_is_the_reply() -> None:
    """An answer in the same turn as the call, after its result, is the reply the result asked for.

    A model that said nothing past its call answers the result in the same provider turn, with no boundary between:
    the answer claims the reply the result asked for, so `wait_for_reply()` returns once it ends.
    """
    conn = _QueuedConnection()
    tools = _Tools()
    session = RealtimeSession(conn, tools.runner)
    async with session:
        consumer = asyncio.create_task(drain_events(session))
        await session.send('Find me a flight.')
        conn.say(OutputTranscript(text='Let me check.', is_final=True), _call())
        await _until(lambda: 'bg_1' in session._pending_tool_calls)  # pyright: ignore[reportPrivateUsage]
        tools.finish('bg_1')
        await _until(lambda: bool(conn.results()))
        conn.say(OutputTranscript(text='The cheapest is KLM.', is_final=True), ResponseDone())
        with anyio.fail_after(_LIVENESS_TIMEOUT):
            await session.wait_for_reply()
        await session.close()
        await consumer

    assert _shape(session.all_messages()) == snapshot(
        [
            ('ModelRequest', ['Find me a flight.']),
            ('ModelResponse', ['Let me check.', 'ToolCallPart']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelResponse', ['The cheapest is KLM.']),
        ]
    )
    assert session.usage.requests == 2


async def test_speech_the_result_cut_into_is_not_the_reply() -> None:
    """A result that goes out mid-sentence ends the calling response there; the rest isn't the reply.

    What the model said before the result stays with the call. The rest of the sentence is a response of its own,
    which answers nothing, so `wait_for_reply()` goes on waiting for the actual answer to the result.
    """
    conn = _QueuedConnection()
    tools = _Tools()
    session = RealtimeSession(conn, tools.runner)
    async with session:
        consumer = asyncio.create_task(drain_events(session))
        await session.send('Find me a flight.')
        conn.say(_call(), OutputTranscript(text='I am look', is_final=False))
        await _until(lambda: session._core.speaking_part() is not None)  # pyright: ignore[reportPrivateUsage]
        tools.finish('bg_1')
        await _until(lambda: bool(conn.results()))
        conn.say(OutputTranscript(text='ing it up.', is_final=False), ResponseDone(interrupted=True))
        await _until(lambda: len(session.all_messages()) == 4)
        # The turn the result cut into is over, and the model still owes the reply to the result.
        assert session._core.reply_outstanding()  # pyright: ignore[reportPrivateUsage]
        wait = asyncio.create_task(session.wait_for_reply())
        conn.say(OutputTranscript(text='Found it.', is_final=True), ResponseDone())
        with anyio.fail_after(_LIVENESS_TIMEOUT):
            await wait
        await session.close()
        await consumer

    assert _shape(session.all_messages()) == snapshot(
        [
            ('ModelRequest', ['Find me a flight.']),
            ('ModelResponse', ['ToolCallPart', 'I am look']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelResponse', ['ing it up.']),
            ('ModelResponse', ['Found it.']),
        ]
    )


@pytest.mark.parametrize(('cancelled', 'state'), [(['a'], 'complete'), (['a', 'b'], 'interrupted')])
async def test_cancelled_calls_mark_their_response_interrupted_only_when_all_are(
    cancelled: list[str], state: str
) -> None:
    """Cut off by the user, the calling response is recorded interrupted only if every call it waits on was cancelled.

    A call still running continues the tool round, so the response isn't abandoned: it ends `completed`, and its
    remaining call's result and answer are still waited for. With every call cancelled, it was cut off, as any
    response is.
    """
    conn = BlockingRealtimeConnection(
        [
            _call('a'),
            _call('b'),
            OutputTranscript(text='Checking both.', is_final=False),
            ToolCallCancelled(tool_call_ids=cancelled),
            ResponseDone(interrupted=True),
        ]
    )
    session = RealtimeSession(conn, _Tools().runner)
    async with session:
        consumer = asyncio.create_task(drain_events(session))
        await _until(lambda: bool(session.all_messages()))
        response = session.all_messages()[0]
        assert isinstance(response, ModelResponse)
        assert response.state == state
        # A call still running is still owed its answer.
        assert session._core.reply_outstanding() == (state == 'complete')  # pyright: ignore[reportPrivateUsage]
        await session.close()
        await consumer


async def test_cut_off_while_the_call_runs_keeps_the_round_going() -> None:
    """The user cuts the model off while it talks past a call still running: the round goes on.

    The response is recorded `complete` (the trade-off: the cut isn't recorded on it), so the result that comes later
    still directly follows it, and a reply that the call's answer completes is still waited for.
    """
    conn = _QueuedConnection()
    tools = _Tools()
    session = RealtimeSession(conn, tools.runner)
    async with session:
        consumer = asyncio.create_task(drain_events(session))
        await session.send('Find me a flight.')
        conn.say(_call(), OutputTranscript(text='Still look', is_final=False), ResponseDone(interrupted=True))
        await _until(lambda: len(session.all_messages()) == 2)
        # The call's answer is still owed.
        assert session._core.reply_outstanding()  # pyright: ignore[reportPrivateUsage]
        wait = asyncio.create_task(session.wait_for_reply())
        tools.finish('bg_1')
        await _until(lambda: bool(conn.results()))
        conn.say(OutputTranscript(text='Found it.', is_final=True), ResponseDone())
        with anyio.fail_after(_LIVENESS_TIMEOUT):
            await wait
        await session.close()
        await consumer

    assert _shape(session.all_messages()) == snapshot(
        [
            ('ModelRequest', ['Find me a flight.']),
            ('ModelResponse', ['ToolCallPart', 'Still look']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelResponse', ['Found it.']),
        ]
    )


async def test_request_limit_at_the_result_still_records_a_return() -> None:
    """A result the request limit stops leaves its call with a return, and the error reaches the caller."""
    conn = _QueuedConnection()
    tools = _Tools()
    session = RealtimeSession(conn, tools.runner, usage_limits=UsageLimits(request_limit=1))
    async with session:
        await session.send('Find me a flight.')
        conn.say(_call(), OutputTranscript(text='This might take a moment.', is_final=False))
        await _until(lambda: session._core.speaking_part() is not None)  # pyright: ignore[reportPrivateUsage]
        tools.finish('bg_1')
        with pytest.raises(UsageLimitExceeded, match='request_limit'):
            with anyio.fail_after(_LIVENESS_TIMEOUT):
                await drain_events(session)

    assert conn.results() == []
    assert _shape(session.all_messages()) == snapshot(
        [
            ('ModelRequest', ['Find me a flight.']),
            ('ModelResponse', ['ToolCallPart', 'This might take a moment.']),
            ('ModelRequest', ['ToolReturnPart']),
        ]
    )
    ret = session.all_messages()[2]
    assert isinstance(ret, ModelRequest) and isinstance(part := ret.parts[0], ToolReturnPart)
    assert part.outcome == snapshot('failed')


async def test_asap_message_from_the_tool_goes_out_before_its_result() -> None:
    """A message the tool enqueues `asap` reaches the model ahead of the tool's result.

    The calling response is still going when the tool finishes, but its result ends it, so the message doesn't wait
    for the response to end, as it doesn't for a call that ended its response.
    """
    conn = _QueuedConnection()
    session: _RealtimeSession

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        # Finishes once the model talks past the call, as a slow tool would.
        await _until(lambda: session._core.speaking_part() is not None)  # pyright: ignore[reportPrivateUsage]
        session.enqueue('a note for the model')
        return 'result'

    session = RealtimeSession(conn, runner)
    async with session:
        consumer = asyncio.create_task(drain_events(session))
        await session.send('Find me a flight.')
        conn.say(_call(), OutputTranscript(text='One moment.', is_final=False))
        await _until(lambda: bool(conn.results()))
        conn.say(ResponseDone(interrupted=True), OutputTranscript(text='Done.', is_final=True), ResponseDone())
        with anyio.fail_after(_LIVENESS_TIMEOUT):
            await session.wait_for_reply()
        await session.close()
        await consumer

    assert [content if isinstance(content, str) else type(content).__name__ for content in conn.sent] == snapshot(
        ['Find me a flight.', 'a note for the model', 'ToolResult']
    )
    assert _shape(session.all_messages()) == snapshot(
        [
            ('ModelRequest', ['Find me a flight.']),
            ('ModelResponse', ['ToolCallPart', 'One moment.']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelRequest', ['a note for the model']),
            ('ModelResponse', ['Done.']),
        ]
    )


async def test_user_turn_while_the_call_runs_follows_the_calling_response() -> None:
    """The user speaks while the model talks past a running call, without a provider boundary.

    The turn joins the conversation after the calling response, which ends there: the reply to it is a response of
    its own, and no watchdog can file the turn ahead of the call, since nothing holds it back.
    """
    conn = _QueuedConnection()
    tools = _Tools()
    session = RealtimeSession(conn, tools.runner)
    async with session:
        consumer = asyncio.create_task(drain_events(session))
        await session.send('Find me a flight.')
        conn.say(
            _call(),
            OutputTranscript(text='Still looking.', is_final=True),
            InputTranscript(text='Any luck?', is_final=True),
            OutputTranscript(text='Not yet.', is_final=True),
            ResponseDone(),
        )
        await _until(lambda: len(session.all_messages()) == 4)
        # Recorded while the tool still runs: the user's turn isn't held behind anything.
        assert 'bg_1' in session._pending_tool_calls  # pyright: ignore[reportPrivateUsage]
        tools.finish('bg_1')
        await _until(lambda: bool(conn.results()))
        conn.say(OutputTranscript(text='Found it.', is_final=True), ResponseDone())
        await _until(lambda: len(session.all_messages()) == 6)
        await session.close()
        await consumer

    assert _shape(session.all_messages()) == snapshot(
        [
            ('ModelRequest', ['Find me a flight.']),
            ('ModelResponse', ['ToolCallPart', 'Still looking.']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelRequest', ['Any luck?']),
            ('ModelResponse', ['Not yet.']),
            ('ModelResponse', ['Found it.']),
        ]
    )


async def test_typed_turn_while_the_call_runs() -> None:
    """A typed turn while the tool runs follows what the model was saying, and the reply to it is its own.

    As recorded on Gemini 2.5 (`test_google_ws.py::test_typed_turn_while_an_async_tool_runs`): the model finishes
    the sentence it was in and ends its turn, which ends the calling response; the typed turn joins the conversation
    after it, and is answered in a turn of its own while the tool still runs. `wait_for_reply()` returns once both
    that reply and the answer to the result are in.
    """
    conn = _QueuedConnection()
    tools = _Tools()
    session = RealtimeSession(conn, tools.runner)
    async with session:
        consumer = asyncio.create_task(drain_events(session))
        await session.send('Find me a flight.')
        conn.say(_call(), OutputTranscript(text='Please hold', is_final=False))
        await _until(lambda: session._core.speaking_part() is not None)  # pyright: ignore[reportPrivateUsage]
        await session.send('What is the capital of Portugal?')
        wait = asyncio.create_task(session.wait_for_reply())
        conn.say(OutputTranscript(text=' on.', is_final=False), ResponseDone())
        # Recorded once the model ended its turn, with no reply yet, while the tool still runs.
        await _until(lambda: len(session.all_messages()) == 3)
        assert _shape(session.all_messages()) == snapshot(
            [
                ('ModelRequest', ['Find me a flight.']),
                ('ModelResponse', ['ToolCallPart', 'Please hold on.']),
                ('ModelRequest', ['What is the capital of Portugal?']),
            ]
        )
        conn.say(OutputTranscript(text='Lisbon.', is_final=True))
        tools.finish('bg_1')
        await _until(lambda: bool(conn.results()))
        conn.say(ResponseDone(interrupted=True))
        await _until(lambda: len(session.all_messages()) == 5)
        # The answer to the result is still owed.
        assert session._core.reply_outstanding()  # pyright: ignore[reportPrivateUsage]
        conn.say(OutputTranscript(text='Found one.', is_final=True), ResponseDone())
        with anyio.fail_after(_LIVENESS_TIMEOUT):
            await wait
        await session.close()
        await consumer

    assert _shape(session.all_messages()) == snapshot(
        [
            ('ModelRequest', ['Find me a flight.']),
            ('ModelResponse', ['ToolCallPart', 'Please hold on.']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelRequest', ['What is the capital of Portugal?']),
            ('ModelResponse', ['Lisbon.']),
            ('ModelResponse', ['Found one.']),
        ]
    )


async def test_cost_limit_tripped_by_the_calling_response() -> None:
    """A cost limit the calling response passes is raised as it ends, with its call's return recorded."""
    conn = _QueuedConnection(model_name='gpt-realtime')
    tools = _Tools()
    session = RealtimeSession(
        conn,
        tools.runner,
        model_name='gpt-realtime',
        provider_name='openai',
        usage_limits=UsageLimits(cost_limit=Decimal('0.0001')),
    )
    async with session:
        await session.send('Find me a flight.')
        conn.say(
            _call(),
            OutputTranscript(text='Searching.', is_final=True),
            SessionUsage(usage=RequestUsage(input_tokens=1000, output_tokens=500)),
            ResponseDone(),
        )
        with pytest.raises(UsageLimitExceeded, match='cost_limit'):
            with anyio.fail_after(_LIVENESS_TIMEOUT):
                await drain_events(session)

    assert _shape(session.all_messages()) == snapshot(
        [
            ('ModelRequest', ['Find me a flight.']),
            ('ModelResponse', ['ToolCallPart', 'Searching.']),
            ('ModelRequest', ['ToolReturnPart']),
        ]
    )


async def test_session_closed_while_the_call_runs_records_the_calling_response() -> None:
    """Nothing leaves the calling response out of history: closing records it, cut off, with an interrupted return."""
    conn = BlockingRealtimeConnection([_call(), OutputTranscript(text='This might take', is_final=False)])
    session = RealtimeSession(conn, _Tools().runner)
    async with session:
        consumer = asyncio.create_task(drain_events(session))
        # The model is talking past the call, which still runs.
        await _until(lambda: session._core.speaking_part() is not None)  # pyright: ignore[reportPrivateUsage]
        assert 'bg_1' in session._pending_tool_calls  # pyright: ignore[reportPrivateUsage]
        assert session.all_messages() == []
        await session.close()
        await consumer

    assert _shape(session.all_messages()) == snapshot(
        [('ModelResponse', ['ToolCallPart', 'This might take']), ('ModelRequest', ['ToolReturnPart'])]
    )
    response = session.all_messages()[0]
    assert isinstance(response, ModelResponse)
    assert response.state == 'interrupted'


async def test_two_calls_in_one_turn_share_its_response() -> None:
    """Calls the model makes while it talks on are in the one response, with what it said between them."""
    conn = _QueuedConnection()
    tools = _Tools()
    session = RealtimeSession(conn, tools.runner)
    async with session:
        consumer = asyncio.create_task(drain_events(session))
        conn.say(
            _call('a'),
            OutputTranscript(text='And the hotel.', is_final=True),
            _call('b'),
            OutputTranscript(text='Both are running.', is_final=True),
            ResponseDone(),
        )
        await _until(lambda: len(session.all_messages()) == 1)
        tools.finish('a')
        tools.finish('b')
        await _until(lambda: len(session.all_messages()) == 3)
        await session.close()
        await consumer

    assert _shape(session.all_messages()) == snapshot(
        [
            ('ModelResponse', ['ToolCallPart', 'And the hotel.', 'ToolCallPart', 'Both are running.']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelRequest', ['ToolReturnPart']),
        ]
    )


async def test_speech_between_two_results_follows_both() -> None:
    """What the model says after the first of two results went out is a response of its own, after both returns.

    Each return directly follows the response with its call, so the speech can't sit between them.
    """
    conn = _QueuedConnection()
    tools = _Tools()
    session = RealtimeSession(conn, tools.runner)
    async with session:
        consumer = asyncio.create_task(drain_events(session))
        conn.say(_call('a'), _call('b'), OutputTranscript(text='Checking both.', is_final=True))
        await _until(lambda: session._core.speaking_part() is not None)  # pyright: ignore[reportPrivateUsage]
        tools.finish('a')
        await _until(lambda: len(conn.results()) == 1)
        conn.say(OutputTranscript(text='The first is in.', is_final=True))
        tools.finish('b')
        await _until(lambda: len(conn.results()) == 2)
        conn.say(ResponseDone(interrupted=True), OutputTranscript(text='Both are in.', is_final=True), ResponseDone())
        await _until(lambda: len(session.all_messages()) == 5)
        await session.close()
        await consumer

    assert _shape(session.all_messages()) == snapshot(
        [
            ('ModelResponse', ['ToolCallPart', 'ToolCallPart', 'Checking both.']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelResponse', ['The first is in.']),
            ('ModelResponse', ['Both are in.']),
        ]
    )


def _speech_while_tool_ran_events() -> tuple[list[RealtimeCodecEvent], list[RealtimeCodecEvent]]:
    return (
        [
            InputTranscript(text="What's the weather in Lisbon?", is_final=True),
            OutputTranscript(text='Let me check.', is_final=True),
            ToolCall(
                tool_call_id='call_1', tool_name='get_weather', args='{"city": "Lisbon"}', response_continues=True
            ),
            OutputTranscript(text='This might take a moment.', is_final=True),
            ResponseDone(),
        ],
        [OutputTranscript(text="It's sunny and 24 degrees in Lisbon.", is_final=True), ResponseDone()],
    )


async def _speech_while_tool_ran_history() -> list[ModelMessage]:
    """A Gemini Live exchange where the model spoke while an asynchronous tool ran, as the session records it."""
    conn = _QueuedConnection()
    first, after_result = _speech_while_tool_ran_events()
    tools = _Tools()

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        await tools.runner(name, args, call_id)
        return 'Sunny, 24C'

    session = RealtimeSession(
        conn, runner, model_name='gemini-2.5-flash-native-audio-latest', provider_name='google-gla'
    )
    async with session:
        consumer = asyncio.create_task(drain_events(session))
        conn.say(*first)
        await _until(lambda: len(session.all_messages()) == 2)
        tools.finish('call_1')
        await _until(lambda: bool(conn.results()))
        conn.say(*after_result)
        await _until(lambda: len(session.all_messages()) == 4)
        await session.close()
        await consumer
    messages = session.all_messages()
    assert _shape(messages) == snapshot(
        [
            ('ModelRequest', ["What's the weather in Lisbon?"]),
            ('ModelResponse', ['Let me check.', 'ToolCallPart', 'This might take a moment.']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelResponse', ["It's sunny and 24 degrees in Lisbon."]),
        ]
    )
    return messages


# The body is part of the match, so a history that serializes differently (the result no longer right after its call,
# say) can't replay this recording, which the provider accepted.
@pytest.mark.vcr(additional_matchers=['body'])
@pytest.mark.parametrize(
    'provider',
    [
        pytest.param(
            'anthropic',
            marks=pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed'),
        ),
        pytest.param(
            'openai-responses', marks=pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
        ),
        pytest.param(
            'openai-chat', marks=pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
        ),
    ],
)
async def test_speech_while_tool_ran_hands_off(
    provider: str, allow_model_requests: None, anthropic_api_key: str, openai_api_key: str
) -> None:
    """The calling response with speech after its call is accepted by the strictest request-response APIs.

    Anthropic and OpenAI Chat reject a tool call that isn't answered by the very next message, so the speech can't be a
    response of its own between the call and its result; within the calling response it is sent as assistant text
    after the call (Anthropic) or alongside it (Chat), which both accept.
    """
    if provider == 'anthropic':
        model: Model = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    elif provider == 'openai-responses':
        model = OpenAIResponsesModel('gpt-4.1-mini', provider=OpenAIProvider(api_key=openai_api_key))
    else:
        model = OpenAIChatModel('gpt-4.1-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model, instructions='Answer in one short sentence.')

    @agent.tool_plain
    # Never called: the history already has its result.
    def get_weather(city: str) -> str:  # pragma: no cover
        return 'unknown'

    result = await agent.run(
        'What temperature did the weather tool report?', message_history=await _speech_while_tool_ran_history()
    )
    assert '24' in result.output


async def test_typed_turn_after_the_result_cut_into_speech_gets_the_next_reply() -> None:
    """A typed turn sent after a result cut into speech, before the turn ended, is answered by what comes next.

    Without it, what the model says next would be the rest of the sentence the result cut into, which answers nothing.
    """
    conn = _QueuedConnection()
    tools = _Tools()
    session = RealtimeSession(conn, tools.runner)
    async with session:
        consumer = asyncio.create_task(drain_events(session))
        conn.say(_call(), OutputTranscript(text='Look', is_final=False))
        await _until(lambda: session._core.speaking_part() is not None)  # pyright: ignore[reportPrivateUsage]
        tools.finish('bg_1')
        await _until(lambda: bool(conn.results()))
        await session.send('Never mind, is it sunny?')
        conn.say(OutputTranscript(text='Yes, it is.', is_final=True), ResponseDone())
        with anyio.fail_after(_LIVENESS_TIMEOUT):
            await session.wait_for_reply()
        await session.close()
        await consumer

    assert _shape(session.all_messages()) == snapshot(
        [
            ('ModelResponse', ['ToolCallPart', 'Look']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelRequest', ['Never mind, is it sunny?']),
            ('ModelResponse', ['Yes, it is.']),
        ]
    )
