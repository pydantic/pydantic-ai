"""Tests for VoiceSession tool dispatch."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.agent.voice import VoiceSession
from pydantic_ai.models.realtime import (
    AudioDelta,
    InputTranscript,
    RealtimeConnection,
    RealtimeEvent,
    RealtimeModel,
    SessionError,
    ToolCall,
    ToolCallCompleted,
    ToolCallStarted,
    Transcript,
    TurnComplete,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

# ---------------------------------------------------------------------------
# Fake implementations for testing
# ---------------------------------------------------------------------------


class FakeRealtimeConnection(RealtimeConnection):
    """A fake connection that yields pre-configured events."""

    def __init__(self, events: list[RealtimeEvent]) -> None:
        self._events = events
        self.sent_audio: list[bytes] = []
        self.sent_tool_results: list[tuple[str, str]] = []

    async def send_audio(self, data: bytes) -> None:
        self.sent_audio.append(data)

    async def send_tool_result(self, tool_call_id: str, output: str) -> None:
        self.sent_tool_results.append((tool_call_id, output))

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        for event in self._events:
            yield event


class FakeRealtimeModel(RealtimeModel):
    """A fake model that yields a pre-configured connection."""

    def __init__(self, connection: FakeRealtimeConnection) -> None:
        self._connection = connection
        self.last_instructions: str | None = None
        self.last_tools: list[ToolDefinition] | None = None

    @property
    def model_name(self) -> str:
        return 'fake-realtime'

    @asynccontextmanager
    async def connect(
        self,
        *,
        instructions: str,
        tools: list[ToolDefinition] | None = None,
        model_settings: ModelSettings | None = None,
    ) -> AsyncIterator[FakeRealtimeConnection]:
        self.last_instructions = instructions
        self.last_tools = tools
        yield self._connection


# ---------------------------------------------------------------------------
# VoiceSession tests
# ---------------------------------------------------------------------------


class TestVoiceSessionPassthrough:
    """Non-ToolCall events should pass through unchanged."""

    @pytest.mark.anyio
    async def test_audio_delta_passthrough(self) -> None:
        conn = FakeRealtimeConnection([AudioDelta(data=b'\x00\x01')])

        async def noop_runner(name: str, args: dict[str, Any]) -> str:
            raise AssertionError('should not be called')

        session = VoiceSession(conn, noop_runner)
        events = [e async for e in session]
        assert len(events) == 1
        assert isinstance(events[0], AudioDelta)
        assert events[0].data == b'\x00\x01'

    @pytest.mark.anyio
    async def test_transcript_passthrough(self) -> None:
        conn = FakeRealtimeConnection([Transcript(text='hello', is_final=True)])
        session = VoiceSession(conn, lambda n, a: None)  # type: ignore
        events = [e async for e in session]
        assert isinstance(events[0], Transcript)

    @pytest.mark.anyio
    async def test_turn_complete_passthrough(self) -> None:
        conn = FakeRealtimeConnection([TurnComplete(interrupted=True)])
        session = VoiceSession(conn, lambda n, a: None)  # type: ignore
        events = [e async for e in session]
        assert isinstance(events[0], TurnComplete)
        assert events[0].interrupted is True

    @pytest.mark.anyio
    async def test_input_transcript_passthrough(self) -> None:
        conn = FakeRealtimeConnection([InputTranscript(text='weather', is_final=False)])
        session = VoiceSession(conn, lambda n, a: None)  # type: ignore
        events = [e async for e in session]
        assert isinstance(events[0], InputTranscript)

    @pytest.mark.anyio
    async def test_session_error_passthrough(self) -> None:
        conn = FakeRealtimeConnection([SessionError(message='oops')])
        session = VoiceSession(conn, lambda n, a: None)  # type: ignore
        events = [e async for e in session]
        assert isinstance(events[0], SessionError)
        assert events[0].message == 'oops'


class TestVoiceSessionToolDispatch:
    """ToolCall events should be intercepted and dispatched."""

    @pytest.mark.anyio
    async def test_tool_call_produces_started_and_completed(self) -> None:
        tool_call = ToolCall(tool_call_id='tc_1', tool_name='get_weather', args='{"city": "Paris"}')
        conn = FakeRealtimeConnection([tool_call])

        async def runner(name: str, args: dict[str, Any]) -> str:
            assert name == 'get_weather'
            assert args == {'city': 'Paris'}
            return 'Sunny, 22C'

        session = VoiceSession(conn, runner)
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
    async def test_tool_result_sent_to_connection(self) -> None:
        tool_call = ToolCall(tool_call_id='tc_2', tool_name='search', args='{"q": "test"}')
        conn = FakeRealtimeConnection([tool_call])

        async def runner(name: str, args: dict[str, Any]) -> str:
            return 'found it'

        session = VoiceSession(conn, runner)
        _ = [e async for e in session]

        assert conn.sent_tool_results == [('tc_2', 'found it')]

    @pytest.mark.anyio
    async def test_mixed_events_and_tool_calls(self) -> None:
        events_in = [
            Transcript(text='thinking...', is_final=False),
            ToolCall(tool_call_id='tc_3', tool_name='calc', args='{"x": 1}'),
            AudioDelta(data=b'\xaa\xbb'),
            TurnComplete(),
        ]
        conn = FakeRealtimeConnection(events_in)

        async def runner(name: str, args: dict[str, Any]) -> str:
            return '42'

        session = VoiceSession(conn, runner)
        events = [e async for e in session]

        assert len(events) == 5
        assert isinstance(events[0], Transcript)
        assert isinstance(events[1], ToolCallStarted)
        assert isinstance(events[2], ToolCallCompleted)
        assert isinstance(events[3], AudioDelta)
        assert isinstance(events[4], TurnComplete)

    @pytest.mark.anyio
    async def test_invalid_json_args_default_to_empty_dict(self) -> None:
        tool_call = ToolCall(tool_call_id='tc_4', tool_name='noop', args='not json')
        conn = FakeRealtimeConnection([tool_call])

        received_args: dict[str, Any] = {}

        async def runner(name: str, args: dict[str, Any]) -> str:
            nonlocal received_args
            received_args = args
            return 'ok'

        session = VoiceSession(conn, runner)
        _ = [e async for e in session]
        assert received_args == {}


class TestVoiceSessionSendAudio:
    @pytest.mark.anyio
    async def test_send_audio_forwards_to_connection(self) -> None:
        conn = FakeRealtimeConnection([])
        session = VoiceSession(conn, lambda n, a: None)  # type: ignore
        await session.send_audio(b'\x01\x02\x03')
        assert conn.sent_audio == [b'\x01\x02\x03']


# ---------------------------------------------------------------------------
# Agent.voice_session integration
# ---------------------------------------------------------------------------


class TestAgentVoiceSession:
    @pytest.mark.anyio
    async def test_agent_voice_session_wires_tools(self) -> None:
        agent: Agent[None, str] = Agent()

        @agent.tool_plain
        def greet(name: str) -> str:
            """Greet someone."""
            return f'Hello {name}!'

        tool_call = ToolCall(tool_call_id='tc_5', tool_name='greet', args='{"name": "Alice"}')
        conn = FakeRealtimeConnection([tool_call, TurnComplete()])
        fake_model = FakeRealtimeModel(conn)

        async with agent.voice_session(model=fake_model) as session:
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
    async def test_agent_voice_session_passes_instructions(self) -> None:
        agent: Agent[None, str] = Agent(instructions='You are a helpful assistant.')

        conn = FakeRealtimeConnection([TurnComplete()])
        fake_model = FakeRealtimeModel(conn)

        async with agent.voice_session(model=fake_model) as session:
            _ = [e async for e in session]

        assert fake_model.last_instructions == 'You are a helpful assistant.'

    @pytest.mark.anyio
    async def test_agent_voice_session_custom_instructions(self) -> None:
        agent: Agent[None, str] = Agent(instructions='Default instructions')

        conn = FakeRealtimeConnection([TurnComplete()])
        fake_model = FakeRealtimeModel(conn)

        async with agent.voice_session(model=fake_model, instructions='Custom voice instructions') as session:
            _ = [e async for e in session]

        assert fake_model.last_instructions == 'Custom voice instructions'

    @pytest.mark.anyio
    async def test_agent_voice_session_unknown_tool(self) -> None:
        """Tool calls for unknown tools should return an error string, not crash."""
        agent: Agent[None, str] = Agent()

        tool_call = ToolCall(tool_call_id='tc_x', tool_name='nonexistent', args='{}')
        conn = FakeRealtimeConnection([tool_call, TurnComplete()])
        fake_model = FakeRealtimeModel(conn)

        async with agent.voice_session(model=fake_model) as session:
            events = [e async for e in session]

        completed = events[1]
        assert isinstance(completed, ToolCallCompleted)
        assert 'Error' in completed.result
        assert 'nonexistent' in completed.result

    @pytest.mark.anyio
    async def test_agent_voice_session_send_audio(self) -> None:
        agent: Agent[None, str] = Agent()

        conn = FakeRealtimeConnection([])
        fake_model = FakeRealtimeModel(conn)

        async with agent.voice_session(model=fake_model) as session:
            await session.send_audio(b'\xab\xcd')

        assert conn.sent_audio == [b'\xab\xcd']
