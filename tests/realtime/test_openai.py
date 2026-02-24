"""Tests for the OpenAI Realtime model and connection."""

from __future__ import annotations as _annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import Any

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
    from pydantic_ai.realtime.openai import (
        OpenAIRealtimeConnection,
        OpenAIRealtimeModel,
        _tool_def_to_openai,  # pyright: ignore[reportPrivateUsage]
        map_event,
    )
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition

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


def test_map_error_non_dict() -> None:
    data: dict[str, object] = {'type': 'error', 'error': 'simple string error'}
    event = map_event(data)
    assert isinstance(event, SessionError)
    assert event.message == 'simple string error'


def test_map_error_dict_no_message() -> None:
    data: dict[str, object] = {'type': 'error', 'error': {'code': 'rate_limit'}}
    event = map_event(data)
    assert isinstance(event, SessionError)
    assert 'rate_limit' in event.message


def test_map_response_done_no_response() -> None:
    data: dict[str, object] = {'type': 'response.done'}
    event = map_event(data)
    assert isinstance(event, TurnComplete)
    assert event.interrupted is False


def test_map_response_done_function_call_only() -> None:
    data: dict[str, object] = {
        'type': 'response.done',
        'response': {
            'status': 'completed',
            'output': [{'type': 'function_call', 'name': 'get_weather'}],
        },
    }
    assert map_event(data) is None


def test_map_audio_delta_non_str_delta() -> None:
    data: dict[str, object] = {'type': 'response.audio.delta', 'delta': 123}
    assert map_event(data) is None


def test_map_transcript_delta_non_str() -> None:
    data: dict[str, object] = {'type': 'response.audio_transcript.delta', 'delta': 123}
    assert map_event(data) is None


def test_map_transcript_done_non_str() -> None:
    data: dict[str, object] = {'type': 'response.audio_transcript.done', 'transcript': 123}
    assert map_event(data) is None


def test_map_input_transcript_non_str() -> None:
    data: dict[str, object] = {'type': 'conversation.item.input_audio_transcription.completed', 'transcript': 123}
    assert map_event(data) is None


def test_map_tool_call_non_str_fields() -> None:
    data: dict[str, object] = {
        'type': 'response.function_call_arguments.done',
        'call_id': 123,
        'name': 'get_weather',
        'arguments': '{}',
    }
    assert map_event(data) is None


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
async def test_skips_non_str_messages() -> None:
    """Non-string WebSocket messages (bytes) are skipped."""

    class BytesWebSocket:
        def __aiter__(self) -> AsyncIterator[str | bytes]:
            return self._iter()

        async def _iter(self) -> AsyncIterator[str | bytes]:
            yield b'\x00\x01'  # binary - should be skipped
            yield json.dumps({'type': 'response.audio_transcript.done', 'transcript': 'hi'})

    ws = BytesWebSocket()
    conn = OpenAIRealtimeConnection(ws)  # type: ignore[arg-type]
    events = [event async for event in conn]
    assert len(events) == 1
    assert isinstance(events[0], Transcript)


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
# _tool_def_to_openai tests
# ---------------------------------------------------------------------------


def test_tool_def_with_description() -> None:
    tool = ToolDefinition(
        name='get_weather',
        description='Get the weather for a city',
        parameters_json_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}},
    )
    result = _tool_def_to_openai(tool)
    assert result['name'] == 'get_weather'
    assert result['description'] == 'Get the weather for a city'
    assert result['type'] == 'function'


def test_tool_def_without_description() -> None:
    tool = ToolDefinition(
        name='noop',
        description='',
        parameters_json_schema={'type': 'object'},
    )
    result = _tool_def_to_openai(tool)
    assert 'description' not in result


# ---------------------------------------------------------------------------
# OpenAIRealtimeModel.connect tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_connect_session_setup() -> None:
    """Test that connect() sends session.update with the right config."""
    session_created = json.dumps({'type': 'session.created', 'session': {}})
    session_updated = json.dumps({'type': 'session.updated', 'session': {}})
    ws = FakeWebSocket([session_created, session_updated])

    from contextlib import asynccontextmanager
    from unittest.mock import patch

    @asynccontextmanager
    async def fake_connect(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield ws

    model = OpenAIRealtimeModel(api_key='test-key', voice='alloy')

    with patch('pydantic_ai.realtime.openai.websockets.connect', fake_connect):
        async with model.connect(instructions='Be helpful') as conn:
            assert isinstance(conn, OpenAIRealtimeConnection)

    # Verify session.update was sent
    assert len(ws.sent) == 1
    update = json.loads(ws.sent[0])
    assert update['type'] == 'session.update'
    assert update['session']['instructions'] == 'Be helpful'
    assert update['session']['voice'] == 'alloy'


@pytest.mark.anyio
async def test_connect_with_tools_and_model_settings() -> None:
    """Test that connect() applies tools and model_settings."""
    session_created = json.dumps({'type': 'session.created', 'session': {}})
    session_updated = json.dumps({'type': 'session.updated', 'session': {}})
    ws = FakeWebSocket([session_created, session_updated])

    from contextlib import asynccontextmanager
    from unittest.mock import patch

    @asynccontextmanager
    async def fake_connect(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield ws

    model = OpenAIRealtimeModel(api_key='test-key')
    tools = [
        ToolDefinition(
            name='get_weather',
            description='Get weather',
            parameters_json_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}},
        )
    ]

    with patch('pydantic_ai.realtime.openai.websockets.connect', fake_connect):
        async with model.connect(
            instructions='Be helpful',
            tools=tools,
            model_settings=ModelSettings(temperature=0.7, max_tokens=500),
        ) as conn:
            assert isinstance(conn, OpenAIRealtimeConnection)

    update = json.loads(ws.sent[0])
    assert len(update['session']['tools']) == 1
    assert update['session']['tools'][0]['name'] == 'get_weather'
    assert update['session']['temperature'] == 0.7
    assert update['session']['max_output_tokens'] == 500


@pytest.mark.anyio
async def test_connect_model_settings_partial() -> None:
    """Test that connect() handles model_settings with only some fields set."""
    session_created = json.dumps({'type': 'session.created', 'session': {}})
    session_updated = json.dumps({'type': 'session.updated', 'session': {}})
    ws = FakeWebSocket([session_created, session_updated])

    from contextlib import asynccontextmanager
    from unittest.mock import patch

    @asynccontextmanager
    async def fake_connect(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield ws

    model = OpenAIRealtimeModel(api_key='test-key')

    with patch('pydantic_ai.realtime.openai.websockets.connect', fake_connect):
        async with model.connect(
            instructions='test',
            model_settings=ModelSettings(max_tokens=200),
        ) as conn:
            assert isinstance(conn, OpenAIRealtimeConnection)

    update = json.loads(ws.sent[0])
    assert 'temperature' not in update['session']
    assert update['session']['max_output_tokens'] == 200


@pytest.mark.anyio
async def test_connect_unexpected_first_message() -> None:
    """Test that connect() raises on unexpected first message."""
    ws = FakeWebSocket([json.dumps({'type': 'error', 'error': 'bad'})])

    from contextlib import asynccontextmanager
    from unittest.mock import patch

    @asynccontextmanager
    async def fake_connect(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield ws

    model = OpenAIRealtimeModel(api_key='test-key')

    with patch('pydantic_ai.realtime.openai.websockets.connect', fake_connect):
        with pytest.raises(RuntimeError, match='Expected session.created'):
            async with model.connect(instructions='test'):
                pass


@pytest.mark.anyio
async def test_connect_unexpected_second_message() -> None:
    """Test that connect() raises on unexpected second message."""
    ws = FakeWebSocket(
        [
            json.dumps({'type': 'session.created', 'session': {}}),
            json.dumps({'type': 'error', 'error': 'bad'}),
        ]
    )

    from contextlib import asynccontextmanager
    from unittest.mock import patch

    @asynccontextmanager
    async def fake_connect(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield ws

    model = OpenAIRealtimeModel(api_key='test-key')

    with patch('pydantic_ai.realtime.openai.websockets.connect', fake_connect):
        with pytest.raises(RuntimeError, match='Expected session.updated'):
            async with model.connect(instructions='test'):
                pass


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
