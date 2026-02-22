"""Tests for the OpenAI Realtime model and connection."""

from __future__ import annotations as _annotations

import base64
import json
from collections.abc import AsyncIterator

import pytest

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai import Agent
    from pydantic_ai.realtime import (
        AudioDelta,
        AudioInput,
        InputTranscript,
        RealtimeSessionEvent,
        SessionError,
        TextInput,
        ToolCall,
        ToolCallCompleted,
        ToolCallStarted,
        ToolResult,
        Transcript,
        TurnComplete,
    )
    from pydantic_ai.realtime.openai import OpenAIRealtimeConnection, OpenAIRealtimeModel, map_event

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='websockets not installed'),
]

# ---------------------------------------------------------------------------
# Fake WebSocket for unit testing
# ---------------------------------------------------------------------------


class FakeWebSocket:
    """A fake WebSocket that yields pre-configured messages."""

    def __init__(self, messages: list[str]) -> None:
        self._messages = messages
        self.sent: list[str] = []

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str:
        return self._messages.pop(0)

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iter_messages()

    async def _iter_messages(self) -> AsyncIterator[str]:
        for msg in self._messages:
            yield msg


# ---------------------------------------------------------------------------
# _map_event tests
# ---------------------------------------------------------------------------


def test_map_audio_delta() -> None:
    audio_bytes = b'\x00\x01\x02\x03'
    data: dict[str, object] = {
        'type': 'response.audio.delta',
        'delta': base64.b64encode(audio_bytes).decode(),
    }
    event = map_event(data)
    assert isinstance(event, AudioDelta)
    assert event.data == audio_bytes


def test_map_transcript_delta() -> None:
    data: dict[str, object] = {'type': 'response.audio_transcript.delta', 'delta': 'Hello'}
    event = map_event(data)
    assert isinstance(event, Transcript)
    assert event.text == 'Hello'
    assert event.is_final is False


def test_map_transcript_done() -> None:
    data: dict[str, object] = {'type': 'response.audio_transcript.done', 'transcript': 'Hello world'}
    event = map_event(data)
    assert isinstance(event, Transcript)
    assert event.text == 'Hello world'
    assert event.is_final is True


def test_map_input_transcript() -> None:
    data: dict[str, object] = {
        'type': 'conversation.item.input_audio_transcription.completed',
        'transcript': 'What is the weather?',
    }
    event = map_event(data)
    assert isinstance(event, InputTranscript)
    assert event.text == 'What is the weather?'
    assert event.is_final is True


def test_map_tool_call() -> None:
    data: dict[str, object] = {
        'type': 'response.function_call_arguments.done',
        'call_id': 'call_123',
        'name': 'get_weather',
        'arguments': '{"city": "London"}',
    }
    event = map_event(data)
    assert isinstance(event, ToolCall)
    assert event.tool_call_id == 'call_123'
    assert event.tool_name == 'get_weather'
    assert event.args == '{"city": "London"}'


def test_map_turn_complete() -> None:
    data: dict[str, object] = {'type': 'response.done', 'response': {'status': 'completed'}}
    event = map_event(data)
    assert isinstance(event, TurnComplete)
    assert event.interrupted is False


def test_map_turn_complete_interrupted() -> None:
    data: dict[str, object] = {'type': 'response.done', 'response': {'status': 'cancelled'}}
    event = map_event(data)
    assert isinstance(event, TurnComplete)
    assert event.interrupted is True


def test_map_error() -> None:
    data: dict[str, object] = {'type': 'error', 'error': {'message': 'rate limit exceeded'}}
    event = map_event(data)
    assert isinstance(event, SessionError)
    assert event.message == 'rate limit exceeded'


def test_map_unknown_event_returns_none() -> None:
    data: dict[str, object] = {'type': 'session.created'}
    assert map_event(data) is None


def test_map_session_updated_returns_none() -> None:
    data: dict[str, object] = {'type': 'session.updated'}
    assert map_event(data) is None


# ---------------------------------------------------------------------------
# OpenAIRealtimeConnection.send tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_send_audio() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]

    audio = b'\xff\xfe\x00\x01'
    await conn.send(AudioInput(data=audio))

    assert len(ws.sent) == 1
    msg = json.loads(ws.sent[0])
    assert msg['type'] == 'input_audio_buffer.append'
    assert base64.b64decode(msg['audio']) == audio


@pytest.mark.anyio
async def test_send_text() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]

    await conn.send(TextInput(text='Hello'))

    assert [json.loads(m) for m in ws.sent] == snapshot(
        [
            {
                'type': 'conversation.item.create',
                'item': {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Hello'}]},
            },
            {'type': 'response.create'},
        ]
    )


@pytest.mark.anyio
async def test_send_tool_result() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]

    await conn.send(ToolResult(tool_call_id='call_abc', output='{"temp": 22}'))

    assert len(ws.sent) == 2
    # First message: conversation.item.create
    item_msg = json.loads(ws.sent[0])
    assert item_msg['type'] == 'conversation.item.create'
    assert item_msg['item']['call_id'] == 'call_abc'
    assert item_msg['item']['output'] == '{"temp": 22}'
    # Second message: response.create
    create_msg = json.loads(ws.sent[1])
    assert create_msg['type'] == 'response.create'


# ---------------------------------------------------------------------------
# OpenAIRealtimeConnection iteration
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_iterates_and_maps_events() -> None:
    messages = [
        json.dumps({'type': 'response.audio_transcript.delta', 'delta': 'Hi'}),
        json.dumps({'type': 'response.done', 'response': {'status': 'completed'}}),
        json.dumps({'type': 'session.updated'}),  # should be skipped
    ]
    ws = FakeWebSocket(messages)
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]

    events = [event async for event in conn]
    assert len(events) == 2
    assert isinstance(events[0], Transcript)
    assert isinstance(events[1], TurnComplete)


# ---------------------------------------------------------------------------
# OpenAIRealtimeModel.model_name
# ---------------------------------------------------------------------------


def test_model_name() -> None:
    model = OpenAIRealtimeModel(model='gpt-4o-realtime-preview-2024-12-17')
    assert model.model_name == 'gpt-4o-realtime-preview-2024-12-17'


def test_default_model_name() -> None:
    model = OpenAIRealtimeModel()
    assert model.model_name == 'gpt-4o-realtime-preview'


# ---------------------------------------------------------------------------
# Integration tests (WebSocket cassette recording/replay)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_openai_connect_and_transcript(openai_realtime_model: OpenAIRealtimeModel) -> None:
    """Connect via Agent.realtime_session and verify transcript + turn complete events."""
    agent: Agent[None, str] = Agent(instructions='You are a helpful assistant. Be brief.')

    events: list[RealtimeSessionEvent] = []
    async with agent.realtime_session(model=openai_realtime_model) as session:
        await session.send_text('Say hello in exactly three words.')
        async for event in session:
            if not isinstance(event, AudioDelta):
                events.append(event)
            if isinstance(event, TurnComplete):
                break

    assert events == snapshot(
        [
            Transcript(text='Hi'),
            Transcript(text=' there'),
            Transcript(text=','),
            Transcript(text=' friend'),
            Transcript(text='!'),
            Transcript(text='Hi there, friend!', is_final=True),
            TurnComplete(),
        ]
    )


@pytest.mark.anyio
async def test_openai_tool_call(openai_realtime_model: OpenAIRealtimeModel) -> None:
    """Connect with a tool via Agent.realtime_session and verify tool dispatch."""
    agent: Agent[None, str] = Agent(
        instructions='You are a weather assistant. Always use the get_weather tool when asked about weather.',
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'Sunny in {city}'

    events: list[RealtimeSessionEvent] = []
    async with agent.realtime_session(model=openai_realtime_model) as session:
        await session.send_text('What is the weather in London?')
        async for event in session:
            if not isinstance(event, AudioDelta):
                events.append(event)
            if isinstance(event, (ToolCallCompleted, TurnComplete)):
                break

    assert events == snapshot(
        [
            ToolCallStarted(tool_name='get_weather', tool_call_id='call_Xm4UH9KdJ1s3B8OK'),
            ToolCallCompleted(tool_name='get_weather', tool_call_id='call_Xm4UH9KdJ1s3B8OK', result='Sunny in London'),
        ]
    )
