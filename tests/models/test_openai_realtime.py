"""Tests for the OpenAI Realtime model and RealtimeConnection."""

from __future__ import annotations as _annotations

import base64
import json
from collections.abc import AsyncIterator

import pytest

from pydantic_ai.models.openai_realtime import OpenAIRealtimeConnection, OpenAIRealtimeModel
from pydantic_ai.models.realtime import (
    AudioDelta,
    InputTranscript,
    SessionError,
    ToolCall,
    Transcript,
    TurnComplete,
)

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
# OpenAIRealtimeConnection._map_event tests
# ---------------------------------------------------------------------------


class TestMapEvent:
    def test_audio_delta(self) -> None:
        audio_bytes = b'\x00\x01\x02\x03'
        data = {
            'type': 'response.audio.delta',
            'delta': base64.b64encode(audio_bytes).decode(),
        }
        event = OpenAIRealtimeConnection._map_event(data)
        assert isinstance(event, AudioDelta)
        assert event.data == audio_bytes

    def test_transcript_delta(self) -> None:
        data = {'type': 'response.audio_transcript.delta', 'delta': 'Hello'}
        event = OpenAIRealtimeConnection._map_event(data)
        assert isinstance(event, Transcript)
        assert event.text == 'Hello'
        assert event.is_final is False

    def test_transcript_done(self) -> None:
        data = {'type': 'response.audio_transcript.done', 'transcript': 'Hello world'}
        event = OpenAIRealtimeConnection._map_event(data)
        assert isinstance(event, Transcript)
        assert event.text == 'Hello world'
        assert event.is_final is True

    def test_input_transcript(self) -> None:
        data = {
            'type': 'conversation.item.input_audio_transcription.completed',
            'transcript': 'What is the weather?',
        }
        event = OpenAIRealtimeConnection._map_event(data)
        assert isinstance(event, InputTranscript)
        assert event.text == 'What is the weather?'
        assert event.is_final is True

    def test_tool_call(self) -> None:
        data = {
            'type': 'response.function_call_arguments.done',
            'call_id': 'call_123',
            'name': 'get_weather',
            'arguments': '{"city": "London"}',
        }
        event = OpenAIRealtimeConnection._map_event(data)
        assert isinstance(event, ToolCall)
        assert event.tool_call_id == 'call_123'
        assert event.tool_name == 'get_weather'
        assert event.args == '{"city": "London"}'

    def test_turn_complete(self) -> None:
        data = {'type': 'response.done', 'response': {'status': 'completed'}}
        event = OpenAIRealtimeConnection._map_event(data)
        assert isinstance(event, TurnComplete)
        assert event.interrupted is False

    def test_turn_complete_interrupted(self) -> None:
        data = {'type': 'response.done', 'response': {'status': 'cancelled'}}
        event = OpenAIRealtimeConnection._map_event(data)
        assert isinstance(event, TurnComplete)
        assert event.interrupted is True

    def test_error(self) -> None:
        data = {'type': 'error', 'error': {'message': 'rate limit exceeded'}}
        event = OpenAIRealtimeConnection._map_event(data)
        assert isinstance(event, SessionError)
        assert event.message == 'rate limit exceeded'

    def test_unknown_event_returns_none(self) -> None:
        data: dict[str, object] = {'type': 'session.created'}
        assert OpenAIRealtimeConnection._map_event(data) is None

    def test_session_updated_returns_none(self) -> None:
        data: dict[str, object] = {'type': 'session.updated'}
        assert OpenAIRealtimeConnection._map_event(data) is None


# ---------------------------------------------------------------------------
# OpenAIRealtimeConnection.send_audio / send_tool_result
# ---------------------------------------------------------------------------


class TestConnectionSend:
    @pytest.mark.anyio
    async def test_send_audio(self) -> None:
        ws = FakeWebSocket([])
        conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]

        audio = b'\xff\xfe\x00\x01'
        await conn.send_audio(audio)

        assert len(ws._sent) == 1
        msg = json.loads(ws._sent[0])
        assert msg['type'] == 'input_audio_buffer.append'
        assert base64.b64decode(msg['audio']) == audio

    @pytest.mark.anyio
    async def test_send_tool_result(self) -> None:
        ws = FakeWebSocket([])
        conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]

        await conn.send_tool_result('call_abc', '{"temp": 22}')

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


class TestConnectionIteration:
    @pytest.mark.anyio
    async def test_iterates_and_maps_events(self) -> None:
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


class TestOpenAIRealtimeModel:
    def test_model_name(self) -> None:
        model = OpenAIRealtimeModel(model='gpt-4o-realtime-preview-2024-12-17')
        assert model.model_name == 'gpt-4o-realtime-preview-2024-12-17'

    def test_default_model_name(self) -> None:
        model = OpenAIRealtimeModel()
        assert model.model_name == 'gpt-4o-realtime-preview'
