"""Tests for `RealtimeSession` tool dispatch and `Agent.realtime_session` integration."""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.realtime import (
    AudioDelta,
    AudioInput,
    InputTranscript,
    RealtimeConnection,
    RealtimeEvent,
    RealtimeInput,
    RealtimeModel,
    RealtimeSession,
    SessionError,
    TextInput,
    ToolCall,
    ToolCallCompleted,
    ToolCallStarted,
    ToolResult,
    Transcript,
    TurnComplete,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

pytestmark = pytest.mark.anyio


async def _noop_runner(name: str, args: dict[str, Any], call_id: str) -> str:  # pragma: no cover
    raise AssertionError('tool runner should not be called')


class FakeRealtimeConnection(RealtimeConnection):
    """A connection that replays a fixed list of events and records what is sent."""

    def __init__(self, events: list[RealtimeEvent], *, release: asyncio.Event | None = None) -> None:
        self._events = events
        self._release = release
        self.sent: list[RealtimeInput] = []

    async def send(self, content: RealtimeInput) -> None:
        self.sent.append(content)

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        for event in self._events:
            yield event
        if self._release is not None:
            self._release.set()


class FakeRealtimeModel(RealtimeModel):
    """A model that yields a pre-built connection and records connect arguments."""

    def __init__(self, connection: FakeRealtimeConnection) -> None:
        self._connection = connection
        self.last_instructions: str | None = None
        self.last_tools: list[ToolDefinition] | None = None
        self.last_model_settings: ModelSettings | None = None

    @property
    def model_name(self) -> str:  # pragma: no cover
        return 'fake-realtime'

    @asynccontextmanager
    async def connect(
        self,
        *,
        instructions: str,
        tools: list[ToolDefinition] | None = None,
        model_settings: ModelSettings | None = None,
    ) -> AsyncGenerator[FakeRealtimeConnection]:
        self.last_instructions = instructions
        self.last_tools = tools
        self.last_model_settings = model_settings
        yield self._connection


async def test_audio_delta_passthrough() -> None:
    conn = FakeRealtimeConnection([AudioDelta(data=b'\x00\x01')])
    session = RealtimeSession(conn, _noop_runner)
    events = [e async for e in session]
    assert events == [AudioDelta(data=b'\x00\x01')]


async def test_transcript_and_input_transcript_passthrough() -> None:
    conn = FakeRealtimeConnection([Transcript(text='hi', is_final=True), InputTranscript(text='weather')])
    session = RealtimeSession(conn, _noop_runner)
    events = [e async for e in session]
    assert events == [Transcript(text='hi', is_final=True), InputTranscript(text='weather')]


async def test_turn_complete_and_error_passthrough() -> None:
    conn = FakeRealtimeConnection([TurnComplete(interrupted=True), SessionError(message='oops')])
    session = RealtimeSession(conn, _noop_runner)
    events = [e async for e in session]
    assert events == [TurnComplete(interrupted=True), SessionError(message='oops')]


async def test_sync_tool_call_emits_started_then_completed() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc_1', tool_name='get_weather', args='{"city": "Paris"}')])

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        assert name == 'get_weather'
        assert args == {'city': 'Paris'}
        return 'Sunny, 22C'

    session = RealtimeSession(conn, runner)
    events = [e async for e in session]

    assert events == [
        ToolCallStarted(tool_name='get_weather', tool_call_id='tc_1'),
        ToolCallCompleted(tool_name='get_weather', tool_call_id='tc_1', result='Sunny, 22C'),
    ]
    assert conn.sent == [ToolResult(tool_call_id='tc_1', output='Sunny, 22C')]


async def test_sync_dispatch_preserves_order_with_other_events() -> None:
    conn = FakeRealtimeConnection(
        [
            Transcript(text='thinking...', is_final=False),
            ToolCall(tool_call_id='tc_3', tool_name='calc', args='{"x": 1}'),
            AudioDelta(data=b'\xaa\xbb'),
            TurnComplete(),
        ]
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return '42'

    session = RealtimeSession(conn, runner)
    events = [e async for e in session]

    assert [type(e).__name__ for e in events] == [
        'Transcript',
        'ToolCallStarted',
        'ToolCallCompleted',
        'AudioDelta',
        'TurnComplete',
    ]


async def test_empty_args_call_runner_with_empty_dict() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='noop', args='')])
    seen: dict[str, Any] | None = None

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        nonlocal seen
        seen = args
        return 'ok'

    session = RealtimeSession(conn, runner)
    _ = [e async for e in session]
    assert seen == {}


async def test_invalid_json_args_reported_without_calling_tool() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc_4', tool_name='noop', args='not json')])
    session = RealtimeSession(conn, _noop_runner)
    events = [e async for e in session]

    completed = events[1]
    assert isinstance(completed, ToolCallCompleted)
    assert 'could not parse tool arguments' in completed.result
    assert isinstance(conn.sent[0], ToolResult)
    assert 'could not parse tool arguments' in conn.sent[0].output


async def test_non_object_json_args_reported() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='noop', args='[1, 2]')])
    session = RealtimeSession(conn, _noop_runner)
    events = [e async for e in session]
    completed = events[1]
    assert isinstance(completed, ToolCallCompleted)
    assert 'expected tool arguments to be a JSON object' in completed.result


async def test_tool_runner_exception_becomes_error_result() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='boom', args='{}')])

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        raise RuntimeError('kaboom')

    session = RealtimeSession(conn, runner)
    events = [e async for e in session]
    completed = events[1]
    assert isinstance(completed, ToolCallCompleted)
    assert completed.result == 'Error: kaboom'
    assert conn.sent == [ToolResult(tool_call_id='tc', output='Error: kaboom')]


async def test_background_tool_does_not_block_other_events() -> None:
    release = asyncio.Event()
    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='bg_1', tool_name='slow', args='{}'),
            Transcript(text='let me check', is_final=False),
            TurnComplete(),
        ],
        release=release,
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        await release.wait()
        return 'done in background'

    session = RealtimeSession(conn, runner, background_tools={'slow'})
    events = [e async for e in session]

    # Started fires immediately, the model keeps talking, and Completed lands only after the turn.
    assert [type(e).__name__ for e in events] == [
        'ToolCallStarted',
        'Transcript',
        'TurnComplete',
        'ToolCallCompleted',
    ]
    completed = events[-1]
    assert isinstance(completed, ToolCallCompleted)
    assert completed.result == 'done in background'
    assert conn.sent == [ToolResult(tool_call_id='bg_1', output='done in background')]


class AwaitBetweenConnection(RealtimeConnection):
    """A connection that yields control between events so background tasks can progress."""

    def __init__(self, events: list[RealtimeEvent]) -> None:
        self._events = events
        self.sent: list[RealtimeInput] = []

    async def send(self, content: RealtimeInput) -> None:
        self.sent.append(content)

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        for event in self._events:
            yield event
            await asyncio.sleep(0)


async def test_background_completion_drained_between_events() -> None:
    conn = AwaitBetweenConnection(
        [ToolCall(tool_call_id='bg', tool_name='fast', args='{}'), AudioDelta(data=b'\x01'), AudioDelta(data=b'\x02')]
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'quick'

    session = RealtimeSession(conn, runner, background_tools={'fast'})
    events = [e async for e in session]

    assert [type(e).__name__ for e in events] == ['ToolCallStarted', 'ToolCallCompleted', 'AudioDelta', 'AudioDelta']
    completed = events[1]
    assert isinstance(completed, ToolCallCompleted)
    assert completed.result == 'quick'


async def test_early_break_with_running_background_cancels_task() -> None:
    blocked = asyncio.Event()
    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='bg', tool_name='hang', args='{}'), AudioDelta(data=b'\x01'), AudioDelta(data=b'\x02')]
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        await blocked.wait()
        return 'never'  # pragma: no cover

    session = RealtimeSession(conn, runner, background_tools={'hang'})
    agen = cast(AsyncGenerator[Any], session.__aiter__())
    seen: list[str] = []
    async for event in agen:
        seen.append(type(event).__name__)
        if isinstance(event, AudioDelta):
            break
    await agen.aclose()

    assert seen == ['ToolCallStarted', 'AudioDelta']


async def test_send_helpers_forward_to_connection() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    await session.send_audio(b'\x01\x02')
    await session.send_text('hello')
    await session.send(AudioInput(data=b'\x03'))
    assert conn.sent == [AudioInput(data=b'\x01\x02'), TextInput(text='hello'), AudioInput(data=b'\x03')]


async def test_early_break_cancels_pump() -> None:
    conn = FakeRealtimeConnection([AudioDelta(data=b'\x00'), AudioDelta(data=b'\x01'), AudioDelta(data=b'\x02')])
    session = RealtimeSession(conn, _noop_runner)
    async for event in session:
        assert isinstance(event, AudioDelta)
        break


async def test_agent_realtime_session_wires_tools_and_instructions() -> None:
    agent: Agent[None, str] = Agent(instructions='You are a helpful assistant.')

    @agent.tool_plain
    def greet(name: str) -> str:
        """Greet someone."""
        return f'Hello {name}!'

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='tc_5', tool_name='greet', args='{"name": "Alice"}'), TurnComplete()]
    )
    model = FakeRealtimeModel(conn)

    async with agent.realtime_session(model=model) as session:
        events = [e async for e in session]

    assert events == [
        ToolCallStarted(tool_name='greet', tool_call_id='tc_5'),
        ToolCallCompleted(tool_name='greet', tool_call_id='tc_5', result='Hello Alice!'),
        TurnComplete(),
    ]
    assert model.last_instructions == 'You are a helpful assistant.'
    assert model.last_tools is not None
    assert 'greet' in [t.name for t in model.last_tools]


async def test_agent_realtime_session_custom_instructions() -> None:
    agent: Agent[None, str] = Agent(instructions='Default')
    conn = FakeRealtimeConnection([TurnComplete()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model, instructions='Custom') as session:
        _ = [e async for e in session]
    assert model.last_instructions == 'Custom'


async def test_agent_realtime_session_default_instructions_empty() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([TurnComplete()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        _ = [e async for e in session]
    assert model.last_instructions == ''


async def test_agent_realtime_session_unknown_tool() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc_x', tool_name='nonexistent', args='{}'), TurnComplete()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        events = [e async for e in session]
    completed = events[1]
    assert isinstance(completed, ToolCallCompleted)
    assert 'unknown tool' in completed.result
    assert 'nonexistent' in completed.result


async def test_agent_realtime_session_tool_exception() -> None:
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def explode() -> str:
        raise ValueError('nope')

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='explode', args='{}'), TurnComplete()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        events = [e async for e in session]
    completed = events[1]
    assert isinstance(completed, ToolCallCompleted)
    assert 'Error' in completed.result and 'nope' in completed.result


async def test_agent_realtime_session_stub_model_visible_to_tools() -> None:
    agent: Agent[None, str] = Agent()
    seen_name: str | None = None

    @agent.tool
    async def inspect_model(ctx: RunContext) -> str:
        nonlocal seen_name
        seen_name = ctx.model.model_name
        return f'system={ctx.model.system}'

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='inspect_model', args='{}'), TurnComplete()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        events = [e async for e in session]

    assert seen_name == 'realtime-session-stub'
    completed = [e for e in events if isinstance(e, ToolCallCompleted)]
    assert completed[0].result == 'system=realtime'


async def test_agent_realtime_session_uses_text_model_when_set() -> None:
    agent: Agent[None, str] = Agent('test')
    seen_system: str | None = None

    @agent.tool
    async def inspect_model(ctx: RunContext) -> str:
        nonlocal seen_system
        seen_system = ctx.model.system
        return 'ok'

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='inspect_model', args='{}'), TurnComplete()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        _ = [e async for e in session]
    assert seen_system != 'realtime'


async def test_agent_realtime_session_background_tools_end_to_end() -> None:
    agent: Agent[None, str] = Agent()
    release = asyncio.Event()

    @agent.tool_plain
    async def slow_lookup() -> str:
        await release.wait()
        return 'background result'

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='bg', tool_name='slow_lookup', args='{}'), TurnComplete()],
        release=release,
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model, background_tools={'slow_lookup'}) as session:
        events = [e async for e in session]

    assert [type(e).__name__ for e in events] == ['ToolCallStarted', 'TurnComplete', 'ToolCallCompleted']
    completed = events[-1]
    assert isinstance(completed, ToolCallCompleted)
    assert completed.result == 'background result'


async def test_agent_realtime_session_forwards_model_settings() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([TurnComplete()])
    model = FakeRealtimeModel(conn)
    settings = ModelSettings(temperature=0.5)
    async with agent.realtime_session(model=model, model_settings=settings) as session:
        _ = [e async for e in session]
    assert model.last_model_settings == settings


async def test_agent_realtime_session_send_audio() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        await session.send_audio(b'\xab\xcd')
    assert conn.sent == [AudioInput(data=b'\xab\xcd')]
