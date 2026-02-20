"""Tests for RealtimeSession tool dispatch."""

from __future__ import annotations as _annotations

from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.realtime import (
    AudioDelta,
    AudioInput,
    InputTranscript,
    RealtimeEvent,
    RealtimeSession,
    SessionError,
    ToolCall,
    ToolCallCompleted,
    ToolCallStarted,
    ToolResult,
    Transcript,
    TurnComplete,
)

from .conftest import FakeRealtimeConnection, FakeRealtimeModel

# ---------------------------------------------------------------------------
# RealtimeSession passthrough tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_audio_delta_passthrough() -> None:
    conn = FakeRealtimeConnection([AudioDelta(data=b'\x00\x01')])

    async def noop_runner(name: str, args: dict[str, Any]) -> str:
        raise AssertionError('should not be called')

    session = RealtimeSession(conn, noop_runner)
    events = [e async for e in session]
    assert len(events) == 1
    assert isinstance(events[0], AudioDelta)
    assert events[0].data == b'\x00\x01'


@pytest.mark.anyio
async def test_transcript_passthrough() -> None:
    conn = FakeRealtimeConnection([Transcript(text='hello', is_final=True)])
    session = RealtimeSession(conn, lambda n, a: None)  # type: ignore
    events = [e async for e in session]
    assert isinstance(events[0], Transcript)


@pytest.mark.anyio
async def test_turn_complete_passthrough() -> None:
    conn = FakeRealtimeConnection([TurnComplete(interrupted=True)])
    session = RealtimeSession(conn, lambda n, a: None)  # type: ignore
    events = [e async for e in session]
    assert isinstance(events[0], TurnComplete)
    assert events[0].interrupted is True


@pytest.mark.anyio
async def test_input_transcript_passthrough() -> None:
    conn = FakeRealtimeConnection([InputTranscript(text='weather', is_final=False)])
    session = RealtimeSession(conn, lambda n, a: None)  # type: ignore
    events = [e async for e in session]
    assert isinstance(events[0], InputTranscript)


@pytest.mark.anyio
async def test_session_error_passthrough() -> None:
    conn = FakeRealtimeConnection([SessionError(message='oops')])
    session = RealtimeSession(conn, lambda n, a: None)  # type: ignore
    events = [e async for e in session]
    assert isinstance(events[0], SessionError)
    assert events[0].message == 'oops'


# ---------------------------------------------------------------------------
# RealtimeSession tool dispatch tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_tool_call_produces_started_and_completed() -> None:
    tool_call = ToolCall(tool_call_id='tc_1', tool_name='get_weather', args='{"city": "Paris"}')
    conn = FakeRealtimeConnection([tool_call])

    async def runner(name: str, args: dict[str, Any]) -> str:
        assert name == 'get_weather'
        assert args == {'city': 'Paris'}
        return 'Sunny, 22C'

    session = RealtimeSession(conn, runner)
    events = [e async for e in session]

    assert len(events) == 2
    started = events[0]
    assert isinstance(started, ToolCallStarted)
    assert started.tool_name == 'get_weather'
    assert started.tool_call_id == 'tc_1'

    completed = events[1]
    assert isinstance(completed, ToolCallCompleted)
    assert completed.tool_name == 'get_weather'
    assert completed.tool_call_id == 'tc_1'
    assert completed.result == 'Sunny, 22C'


@pytest.mark.anyio
async def test_tool_result_sent_to_connection() -> None:
    tool_call = ToolCall(tool_call_id='tc_2', tool_name='search', args='{"q": "test"}')
    conn = FakeRealtimeConnection([tool_call])

    async def runner(name: str, args: dict[str, Any]) -> str:
        return 'found it'

    session = RealtimeSession(conn, runner)
    _ = [e async for e in session]

    tool_results = [s for s in conn.sent if isinstance(s, ToolResult)]
    assert len(tool_results) == 1
    assert tool_results[0].tool_call_id == 'tc_2'
    assert tool_results[0].output == 'found it'


@pytest.mark.anyio
async def test_mixed_events_and_tool_calls() -> None:
    events_in: list[RealtimeEvent] = [
        Transcript(text='thinking...', is_final=False),
        ToolCall(tool_call_id='tc_3', tool_name='calc', args='{"x": 1}'),
        AudioDelta(data=b'\xaa\xbb'),
        TurnComplete(),
    ]
    conn = FakeRealtimeConnection(events_in)

    async def runner(name: str, args: dict[str, Any]) -> str:
        return '42'

    session = RealtimeSession(conn, runner)
    events = [e async for e in session]

    assert len(events) == 5
    assert isinstance(events[0], Transcript)
    assert isinstance(events[1], ToolCallStarted)
    assert isinstance(events[2], ToolCallCompleted)
    assert isinstance(events[3], AudioDelta)
    assert isinstance(events[4], TurnComplete)


@pytest.mark.anyio
async def test_invalid_json_args_default_to_empty_dict() -> None:
    tool_call = ToolCall(tool_call_id='tc_4', tool_name='noop', args='not json')
    conn = FakeRealtimeConnection([tool_call])

    received_args: dict[str, Any] = {}

    async def runner(name: str, args: dict[str, Any]) -> str:
        nonlocal received_args
        received_args = args
        return 'ok'

    session = RealtimeSession(conn, runner)
    _ = [e async for e in session]
    assert received_args == {}


# ---------------------------------------------------------------------------
# RealtimeSession.send tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_send_audio_forwards_to_connection() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, lambda n, a: None)  # type: ignore
    await session.send_audio(b'\x01\x02\x03')
    audio_inputs = [s for s in conn.sent if isinstance(s, AudioInput)]
    assert len(audio_inputs) == 1
    assert audio_inputs[0].data == b'\x01\x02\x03'


@pytest.mark.anyio
async def test_send_forwards_content_directly() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, lambda n, a: None)  # type: ignore
    content = AudioInput(data=b'\xab\xcd')
    await session.send(content)
    assert conn.sent == [content]


# ---------------------------------------------------------------------------
# Agent.realtime_session integration
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_agent_realtime_session_wires_tools() -> None:
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def greet(name: str) -> str:
        """Greet someone."""
        return f'Hello {name}!'

    tool_call = ToolCall(tool_call_id='tc_5', tool_name='greet', args='{"name": "Alice"}')
    conn = FakeRealtimeConnection([tool_call, TurnComplete()])
    fake_model = FakeRealtimeModel(conn)

    async with agent.realtime_session(model=fake_model) as session:
        events = [e async for e in session]

    assert len(events) == 3
    assert isinstance(events[0], ToolCallStarted)
    assert isinstance(events[1], ToolCallCompleted)
    assert events[1].result == 'Hello Alice!'
    assert isinstance(events[2], TurnComplete)

    # Verify tools were passed to the model
    assert fake_model.last_tools is not None
    tool_names = [t.name for t in fake_model.last_tools]
    assert 'greet' in tool_names


@pytest.mark.anyio
async def test_agent_realtime_session_passes_instructions() -> None:
    agent: Agent[None, str] = Agent(instructions='You are a helpful assistant.')

    conn = FakeRealtimeConnection([TurnComplete()])
    fake_model = FakeRealtimeModel(conn)

    async with agent.realtime_session(model=fake_model) as session:
        _ = [e async for e in session]

    assert fake_model.last_instructions == 'You are a helpful assistant.'


@pytest.mark.anyio
async def test_agent_realtime_session_custom_instructions() -> None:
    agent: Agent[None, str] = Agent(instructions='Default instructions')

    conn = FakeRealtimeConnection([TurnComplete()])
    fake_model = FakeRealtimeModel(conn)

    async with agent.realtime_session(model=fake_model, instructions='Custom instructions') as session:
        _ = [e async for e in session]

    assert fake_model.last_instructions == 'Custom instructions'


@pytest.mark.anyio
async def test_agent_realtime_session_unknown_tool() -> None:
    """Tool calls for unknown tools should return an error string, not crash."""
    agent: Agent[None, str] = Agent()

    tool_call = ToolCall(tool_call_id='tc_x', tool_name='nonexistent', args='{}')
    conn = FakeRealtimeConnection([tool_call, TurnComplete()])
    fake_model = FakeRealtimeModel(conn)

    async with agent.realtime_session(model=fake_model) as session:
        events = [e async for e in session]

    completed = events[1]
    assert isinstance(completed, ToolCallCompleted)
    assert 'Error' in completed.result
    assert 'nonexistent' in completed.result


@pytest.mark.anyio
async def test_agent_realtime_session_send_audio() -> None:
    agent: Agent[None, str] = Agent()

    conn = FakeRealtimeConnection([])
    fake_model = FakeRealtimeModel(conn)

    async with agent.realtime_session(model=fake_model) as session:
        await session.send_audio(b'\xab\xcd')

    audio_inputs = [s for s in conn.sent if isinstance(s, AudioInput)]
    assert len(audio_inputs) == 1
    assert audio_inputs[0].data == b'\xab\xcd'
