"""Tests for the OpenAI Realtime model and connection."""

from __future__ import annotations as _annotations

import base64
import json
from collections.abc import AsyncIterator

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.realtime import (
        AudioDelta,
        AudioInput,
        InputTranscript,
        SessionError,
        ToolCall,
        ToolResult,
        Transcript,
        TurnComplete,
    )
    from pydantic_ai.realtime.openai import OpenAIRealtimeConnection, OpenAIRealtimeModel, _map_event

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
        self._sent: list[str] = []

    async def send(self, data: str) -> None:
        self._sent.append(data)

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
    event = _map_event(data)
    assert isinstance(event, AudioDelta)
    assert event.data == audio_bytes


def test_map_transcript_delta() -> None:
    data: dict[str, object] = {'type': 'response.audio_transcript.delta', 'delta': 'Hello'}
    event = _map_event(data)
    assert isinstance(event, Transcript)
    assert event.text == 'Hello'
    assert event.is_final is False


def test_map_transcript_done() -> None:
    data: dict[str, object] = {'type': 'response.audio_transcript.done', 'transcript': 'Hello world'}
    event = _map_event(data)
    assert isinstance(event, Transcript)
    assert event.text == 'Hello world'
    assert event.is_final is True


def test_map_input_transcript() -> None:
    data: dict[str, object] = {
        'type': 'conversation.item.input_audio_transcription.completed',
        'transcript': 'What is the weather?',
    }
    event = _map_event(data)
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
    event = _map_event(data)
    assert isinstance(event, ToolCall)
    assert event.tool_call_id == 'call_123'
    assert event.tool_name == 'get_weather'
    assert event.args == '{"city": "London"}'


def test_map_turn_complete() -> None:
    data: dict[str, object] = {'type': 'response.done', 'response': {'status': 'completed'}}
    event = _map_event(data)
    assert isinstance(event, TurnComplete)
    assert event.interrupted is False


def test_map_turn_complete_interrupted() -> None:
    data: dict[str, object] = {'type': 'response.done', 'response': {'status': 'cancelled'}}
    event = _map_event(data)
    assert isinstance(event, TurnComplete)
    assert event.interrupted is True


def test_map_error() -> None:
    data: dict[str, object] = {'type': 'error', 'error': {'message': 'rate limit exceeded'}}
    event = _map_event(data)
    assert isinstance(event, SessionError)
    assert event.message == 'rate limit exceeded'


def test_map_unknown_event_returns_none() -> None:
    data: dict[str, object] = {'type': 'session.created'}
    assert _map_event(data) is None


def test_map_session_updated_returns_none() -> None:
    data: dict[str, object] = {'type': 'session.updated'}
    assert _map_event(data) is None


# ---------------------------------------------------------------------------
# OpenAIRealtimeConnection.send tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_send_audio() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]

    audio = b'\xff\xfe\x00\x01'
    await conn.send(AudioInput(data=audio))

    assert len(ws._sent) == 1
    msg = json.loads(ws._sent[0])
    assert msg['type'] == 'input_audio_buffer.append'
    assert base64.b64decode(msg['audio']) == audio


@pytest.mark.anyio
async def test_send_tool_result() -> None:
    ws = FakeWebSocket([])
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]

    await conn.send(ToolResult(tool_call_id='call_abc', output='{"temp": 22}'))

    assert len(ws._sent) == 2
    # First message: conversation.item.create
    item_msg = json.loads(ws._sent[0])
    assert item_msg['type'] == 'conversation.item.create'
    assert item_msg['item']['call_id'] == 'call_abc'
    assert item_msg['item']['output'] == '{"temp": 22}'
    # Second message: response.create
    create_msg = json.loads(ws._sent[1])
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
