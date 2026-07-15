"""Tests for `RealtimeSession`: event translation, history assembly, tool dispatch, and `Agent.realtime_session`."""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability, NativeTool, WebFetch
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryContent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    SpeechPart,
    SpeechPartDelta,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.native_tools import AbstractNativeTool, CodeExecutionTool, WebFetchTool, WebSearchTool
from pydantic_ai.realtime import (
    AudioDelta,
    AudioInput,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    ImageInput,
    InputTranscript,
    NativeToolParts,
    RealtimeConnection,
    RealtimeEvent,
    RealtimeInput,
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
    RealtimeSession,
    SessionErrorEvent,
    SessionUsageEvent,
    SourcesEvent,
    SpeechStoppedEvent,
    TextInput,
    ToolCall,
    ToolResult,
    Transcript,
    TruncateOutput,
    TurnCompleteEvent,
    WebSource,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from ..conftest import IsDatetime, IsStr

pytestmark = pytest.mark.anyio


async def _noop_runner(name: str, args: dict[str, Any], call_id: str) -> str:  # pragma: no cover
    raise AssertionError('tool runner should not be called')


def _profile(
    *,
    supports_image_input: bool = True,
    supports_manual_turn_control: bool = True,
    supports_interruption: bool = True,
    supports_output_truncation: bool = True,
    supports_session_seeding: bool = True,
    supported_native_tools: frozenset[type[AbstractNativeTool]] = frozenset(
        {WebSearchTool, WebFetchTool, CodeExecutionTool}
    ),
) -> RealtimeModelProfile:
    """A full-support profile with per-field overrides, so a guard test can flip one flag off."""
    return RealtimeModelProfile(
        supports_image_input=supports_image_input,
        supports_manual_turn_control=supports_manual_turn_control,
        supports_interruption=supports_interruption,
        supports_output_truncation=supports_output_truncation,
        supports_session_seeding=supports_session_seeding,
        supported_native_tools=supported_native_tools,
    )


class FakeRealtimeConnection(RealtimeConnection):
    """A connection that replays a fixed list of events and records what is sent."""

    def __init__(
        self,
        events: list[RealtimeEvent],
        *,
        release: asyncio.Event | None = None,
        input_transcription_enabled: bool = True,
    ) -> None:
        self._events = events
        self._release = release
        self._input_transcription_enabled = input_transcription_enabled
        self.sent: list[RealtimeInput] = []

    @property
    def input_transcription_enabled(self) -> bool:
        return self._input_transcription_enabled

    async def send(self, content: RealtimeInput) -> None:
        self.sent.append(content)

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        for event in self._events:
            yield event
        if self._release is not None:
            self._release.set()


class FakeRealtimeModel(RealtimeModel):
    """A model that yields a pre-built connection and records connect arguments."""

    def __init__(
        self,
        connection: FakeRealtimeConnection,
        *,
        settings: RealtimeModelSettings | None = None,
        profile: RealtimeModelProfile | None = None,
    ) -> None:
        self._connection = connection
        self.settings = settings
        self._profile = profile or _profile()
        self.last_instructions: str | None = None
        self.last_tools: list[ToolDefinition] | None = None
        self.last_native_tools: list[AbstractNativeTool] | None = None
        self.last_model_settings: RealtimeModelSettings | None = None
        self.last_messages: Sequence[ModelMessage] | None = None

    @property
    def model_name(self) -> str:
        return 'fake-realtime'

    @property
    def system(self) -> str:
        return 'fake'

    @property
    def profile(self) -> RealtimeModelProfile:
        return self._profile

    @asynccontextmanager
    async def connect(
        self,
        *,
        instructions: str,
        tools: list[ToolDefinition] | None = None,
        native_tools: list[AbstractNativeTool] | None = None,
        model_settings: RealtimeModelSettings | None = None,
        messages: Sequence[ModelMessage] | None = None,
    ) -> AsyncGenerator[FakeRealtimeConnection]:
        self.last_instructions = instructions
        self.last_tools = tools
        self.last_native_tools = native_tools
        self.last_model_settings = model_settings
        self.last_messages = messages
        yield self._connection


# --- event translation -------------------------------------------------------------------------


async def test_assistant_transcript_partials_then_final() -> None:
    # Partial transcript deltas stream as PartDeltaEvents; the final (full-text) event adds nothing new.
    conn = FakeRealtimeConnection(
        [
            Transcript(text='Hi ', is_final=False),
            Transcript(text='there', is_final=False),
            Transcript(text='Hi there', is_final=True),  # provider repeats the full text
            TurnCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = [e async for e in session]
    assert events == snapshot(
        [
            PartStartEvent(index=0, part=SpeechPart(speaker='assistant', transcript='')),
            PartDeltaEvent(index=0, delta=SpeechPartDelta(transcript_delta='Hi ')),
            PartDeltaEvent(index=0, delta=SpeechPartDelta(transcript_delta='there')),
            PartEndEvent(index=0, part=SpeechPart(speaker='assistant', transcript='Hi there')),
            TurnCompleteEvent(),
        ]
    )
    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='Hi there')],
                model_name='m',
                timestamp=IsDatetime(),
            )
        ]
    )


async def test_assistant_transcript_final_only() -> None:
    # A provider that only sends a single final transcript still yields a delta then the completed part.
    conn = FakeRealtimeConnection([Transcript(text='Hello world', is_final=True), TurnCompleteEvent()])
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = [e async for e in session]
    assert events == snapshot(
        [
            PartStartEvent(index=0, part=SpeechPart(speaker='assistant', transcript='')),
            PartDeltaEvent(index=0, delta=SpeechPartDelta(transcript_delta='Hello world')),
            PartEndEvent(index=0, part=SpeechPart(speaker='assistant', transcript='Hello world')),
            TurnCompleteEvent(),
        ]
    )


async def test_user_transcript_final_becomes_request() -> None:
    conn = FakeRealtimeConnection(
        [InputTranscript(text='what is ', is_final=False), InputTranscript(text='the weather', is_final=True)]
    )
    session = RealtimeSession(conn, _noop_runner)
    events = [e async for e in session]
    assert events == snapshot(
        [
            PartStartEvent(index=0, part=SpeechPart(speaker='user', transcript='')),
            PartDeltaEvent(index=0, delta=SpeechPartDelta(transcript_delta='what is ')),
            PartDeltaEvent(index=0, delta=SpeechPartDelta(transcript_delta='the weather')),
            PartEndEvent(index=0, part=SpeechPart(speaker='user', transcript='what is the weather')),
        ]
    )
    assert session.new_messages() == snapshot(
        [ModelRequest(parts=[SpeechPart(speaker='user', transcript='what is the weather')])]
    )


async def test_audio_delta_streams_and_transcript_pairs() -> None:
    conn = FakeRealtimeConnection(
        [AudioDelta(data=b'\x00\x01'), Transcript(text='hi', is_final=True), TurnCompleteEvent()]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = [e async for e in session]
    assert [type(e).__name__ for e in events] == snapshot(
        ['PartStartEvent', 'PartDeltaEvent', 'PartDeltaEvent', 'PartEndEvent', 'TurnCompleteEvent']
    )
    # transcript_only (default): the completed part keeps the transcript but not the audio bytes.
    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='hi')],
                model_name='m',
                timestamp=IsDatetime(),
            )
        ]
    )


async def test_control_events_pass_through() -> None:
    conn = FakeRealtimeConnection([TurnCompleteEvent(interrupted=True), SessionErrorEvent(message='oops')])
    session = RealtimeSession(conn, _noop_runner)
    events = [e async for e in session]
    assert events == [TurnCompleteEvent(interrupted=True), SessionErrorEvent(message='oops')]


async def test_interrupted_turn_keeps_partial_transcript() -> None:
    # A barge-in cancels the turn; the completed part reflects the partial transcript seen so far.
    conn = FakeRealtimeConnection(
        [Transcript(text='the answer is ', is_final=False), TurnCompleteEvent(interrupted=True)]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = [e async for e in session]
    assert events[-1] == TurnCompleteEvent(interrupted=True)
    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='the answer is ')],
                model_name='m',
                timestamp=IsDatetime(),
            )
        ]
    )


# --- tool calls: history + events --------------------------------------------------------------


async def test_tool_call_round_builds_classic_history() -> None:
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text="what's the weather in Paris", is_final=True),
            ToolCall(tool_call_id='tc_1', tool_name='get_weather', args='{"city": "Paris"}'),
            Transcript(text="It's sunny in Paris", is_final=True),
            TurnCompleteEvent(),
        ]
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        assert name == 'get_weather'
        assert args == {'city': 'Paris'}
        return 'Sunny, 22C'

    session = RealtimeSession(conn, runner, model_name='m')
    events = [e async for e in session]

    assert [type(e).__name__ for e in events] == snapshot(
        [
            'PartStartEvent',  # user transcript start
            'PartDeltaEvent',
            'PartEndEvent',  # user transcript end
            'PartStartEvent',  # tool call part start
            'PartEndEvent',  # tool call part end
            'FunctionToolCallEvent',
            'FunctionToolResultEvent',
            'PartStartEvent',  # assistant answer start
            'PartDeltaEvent',
            'PartEndEvent',  # assistant answer end
            'TurnCompleteEvent',
        ]
    )
    assert conn.sent == [ToolResult(tool_call_id='tc_1', output='Sunny, 22C')]
    # History mirrors a classic tool-call round: user request, tool-call response, tool result, answer.
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user', transcript="what's the weather in Paris")]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_weather', args='{"city": "Paris"}', tool_call_id='tc_1')],
                model_name='m',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather', content='Sunny, 22C', tool_call_id='tc_1', timestamp=IsDatetime()
                    )
                ]
            ),
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript="It's sunny in Paris")],
                model_name='m',
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_tool_call_events_carry_real_parts() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='f', args='{"x": 1}')])

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return '42'

    session = RealtimeSession(conn, runner)
    events = [e async for e in session]
    call_event = next(e for e in events if isinstance(e, FunctionToolCallEvent))
    result_event = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert call_event.part == ToolCallPart(tool_name='f', args='{"x": 1}', tool_call_id='tc')
    result_part = result_event.part
    assert isinstance(result_part, ToolReturnPart)
    assert (result_part.tool_name, result_part.content, result_part.tool_call_id) == ('f', '42', 'tc')


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
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, ToolReturnPart)
    assert 'could not parse tool arguments' in str(result.part.content)
    assert isinstance(conn.sent[0], ToolResult)
    assert 'could not parse tool arguments' in conn.sent[0].output


async def test_non_object_json_args_reported() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='noop', args='[1, 2]')])
    session = RealtimeSession(conn, _noop_runner)
    events = [e async for e in session]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert 'expected tool arguments to be a JSON object' in str(result.part.content)


async def test_tool_runner_exception_becomes_error_result() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='boom', args='{}')])

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        raise RuntimeError('kaboom')

    session = RealtimeSession(conn, runner)
    events = [e async for e in session]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert result.part.content == 'Error: kaboom'
    assert conn.sent == [ToolResult(tool_call_id='tc', output='Error: kaboom')]


async def test_background_tool_does_not_block_other_events() -> None:
    release = asyncio.Event()
    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='bg_1', tool_name='slow', args='{}'),
            Transcript(text='let me check', is_final=False),
            TurnCompleteEvent(),
        ],
        release=release,
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        await release.wait()
        return 'done in background'

    session = RealtimeSession(conn, runner, background_tools={'slow'})
    events = [e async for e in session]

    # The tool call fires immediately, the model keeps talking, and the result lands only after the turn.
    assert [type(e).__name__ for e in events] == snapshot(
        [
            'PartStartEvent',  # tool call part
            'PartEndEvent',
            'FunctionToolCallEvent',
            'PartStartEvent',  # assistant transcript
            'PartDeltaEvent',
            'PartEndEvent',
            'TurnCompleteEvent',
            'FunctionToolResultEvent',
        ]
    )
    result = events[-1]
    assert isinstance(result, FunctionToolResultEvent)
    assert result.part.content == 'done in background'
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

    assert [type(e).__name__ for e in events] == snapshot(
        [
            'PartStartEvent',
            'PartEndEvent',
            'FunctionToolCallEvent',
            'FunctionToolResultEvent',
            'PartStartEvent',
            'PartDeltaEvent',
            'PartDeltaEvent',
        ]
    )
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert result.part.content == 'quick'


class IdleAfterToolConnection(RealtimeConnection):
    """Yields one ToolCall, then blocks forever — the model goes idle with no further events."""

    def __init__(self, call: ToolCall) -> None:
        self._call = call
        self.sent: list[RealtimeInput] = []

    async def send(self, content: RealtimeInput) -> None:
        self.sent.append(content)

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        yield self._call
        await asyncio.Event().wait()


async def test_background_completion_delivered_while_upstream_idle() -> None:
    # The connection goes silent after the tool call; the completion must still surface promptly
    # rather than waiting for a provider event that never arrives.
    conn = IdleAfterToolConnection(ToolCall(tool_call_id='bg', tool_name='fast', args='{}'))

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'ready'

    session = RealtimeSession(conn, runner, background_tools={'fast'})
    agen = cast(AsyncGenerator[Any], session.__aiter__())
    # Drain the tool-call part events + FunctionToolCallEvent.
    assert isinstance(await agen.__anext__(), PartStartEvent)
    assert isinstance(await agen.__anext__(), PartEndEvent)
    assert isinstance(await agen.__anext__(), FunctionToolCallEvent)
    # Without multiplexing this would hang forever waiting on the idle connection.
    completed = await asyncio.wait_for(agen.__anext__(), timeout=1.0)
    assert isinstance(completed, FunctionToolResultEvent)
    assert completed.part.content == 'ready'
    await agen.aclose()


class ExplodingConnection(RealtimeConnection):
    """A connection whose iteration raises after yielding one event."""

    def __init__(self) -> None:
        self.sent: list[RealtimeInput] = []

    async def send(self, content: RealtimeInput) -> None:  # pragma: no cover
        self.sent.append(content)

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        yield AudioDelta(data=b'\x00')
        raise RuntimeError('connection dropped')


async def test_upstream_error_propagates_to_consumer() -> None:
    session = RealtimeSession(ExplodingConnection(), _noop_runner)
    with pytest.raises(RuntimeError, match='connection dropped'):
        _ = [e async for e in session]


class SendFailsConnection(RealtimeConnection):
    """Replays events but raises on every send — a connection dropping mid tool call (the only thing
    sent through it in these tests is the tool's `ToolResult`).

    With `idle=True` it never closes after the events (an idle provider); with `release` set it
    closes immediately and the tool only runs once released, so the failure lands after upstream end.
    """

    def __init__(
        self, events: list[RealtimeEvent], *, idle: bool = False, release: asyncio.Event | None = None
    ) -> None:
        self._events = events
        self._idle = idle
        self._release = release

    async def send(self, content: RealtimeInput) -> None:
        raise RuntimeError('connection lost')

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        for event in self._events:
            yield event
        if self._release is not None:
            self._release.set()
        if self._idle:
            await asyncio.Event().wait()


async def test_background_tool_failure_propagates_while_idle() -> None:
    conn = SendFailsConnection([ToolCall(tool_call_id='bg', tool_name='boom', args='{}')], idle=True)

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'ok'

    # The background tool's ToolResult send fails while the provider is idle; without propagation the
    # consumer would hang waiting for a completion that never arrives.
    session = RealtimeSession(conn, runner, background_tools={'boom'})
    with pytest.raises(RuntimeError, match='connection lost'):
        _ = [e async for e in session]


async def test_background_tool_failure_propagates_after_close() -> None:
    release = asyncio.Event()
    conn = SendFailsConnection([ToolCall(tool_call_id='bg', tool_name='boom', args='{}')], release=release)

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        await release.wait()  # only runs after upstream has closed → failure surfaces during drain
        return 'ok'

    session = RealtimeSession(conn, runner, background_tools={'boom'})
    with pytest.raises(RuntimeError, match='connection lost'):
        _ = [e async for e in session]


async def test_early_break_with_running_background_cancels_task() -> None:
    blocked = asyncio.Event()
    started = asyncio.Event()
    # AwaitBetweenConnection yields control between events, so the background task actually starts.
    conn = AwaitBetweenConnection([ToolCall(tool_call_id='bg', tool_name='hang', args='{}'), AudioDelta(data=b'\x01')])

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        started.set()
        await blocked.wait()
        return 'never'  # pragma: no cover

    session = RealtimeSession(conn, runner, background_tools={'hang'})
    agen = cast(AsyncGenerator[Any], session.__aiter__())
    assert isinstance(await agen.__anext__(), PartStartEvent)  # tool call part
    assert isinstance(await agen.__anext__(), PartEndEvent)
    assert isinstance(await agen.__anext__(), FunctionToolCallEvent)
    assert isinstance(await agen.__anext__(), PartStartEvent)  # the audio delta's part
    assert started.is_set()  # the background tool is running by now
    await agen.aclose()  # cancels the still-running background task


# --- send helpers + history ---------------------------------------------------------------------


async def test_send_helpers_forward_to_connection() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    await session.send_audio(b'\x01\x02')
    await session.send_text('hello')
    await session.send_image(b'\xff\xd8', mime_type='image/jpeg')
    await session.send(AudioInput(data=b'\x03'))
    assert conn.sent == [
        AudioInput(data=b'\x01\x02'),
        TextInput(text='hello'),
        ImageInput(data=b'\xff\xd8', mime_type='image/jpeg'),
        AudioInput(data=b'\x03'),
    ]


async def test_send_dispatches_through_bookkeeping_helpers() -> None:
    # `send(TextInput(...))` must route through `send_text`, so the user turn lands in history rather
    # than bypassing it (the raw pass-through used to skip all session bookkeeping).
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    await session.send(TextInput(text='hello'))
    assert conn.sent == [TextInput(text='hello')]
    assert session.new_messages() == snapshot(
        [ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsDatetime())])]
    )


async def test_send_enforces_model_profile_guard() -> None:
    # `send(ImageInput(...))` must enforce the same `supports_image_input` guard as `send_image`.
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, profile=_profile(supports_image_input=False))
    with pytest.raises(UserError, match='does not support image input'):
        await session.send(ImageInput(data=b'\xff', mime_type='image/jpeg'))
    assert conn.sent == []


async def test_send_dispatches_control_inputs_through_helpers() -> None:
    # Every control `RealtimeSessionInput` variant routes through its typed helper (which applies the
    # model-profile guards) rather than reaching the connection raw.
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    await session.send(CommitAudio())
    await session.send(ClearAudio())
    await session.send(CreateResponse())
    await session.send(TruncateOutput(audio_end_ms=120))
    await session.send(CancelResponse())
    assert conn.sent == [
        CommitAudio(),
        ClearAudio(),
        CreateResponse(),
        TruncateOutput(audio_end_ms=120),
        CancelResponse(),
    ]


async def test_send_rejects_tool_result() -> None:
    # Tool results are sent by the session itself; a caller can't inject one via `send()`. `ToolResult`
    # is excluded from `RealtimeSessionInput` (hence the type-ignore), so this also guards the runtime path.
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    with pytest.raises(UserError, match='Tool results are sent automatically'):
        await session.send(ToolResult(tool_call_id='c', output='x'))  # type: ignore[arg-type]
    assert conn.sent == []


async def test_send_text_adds_user_prompt_to_history() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    await session.send_text('turn it up')
    assert session.new_messages() == snapshot(
        [ModelRequest(parts=[UserPromptPart(content='turn it up', timestamp=IsDatetime())])]
    )


async def test_manual_turn_control_helpers_forward_to_connection() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    await session.commit_audio()
    await session.create_response()
    await session.clear_audio()
    assert conn.sent == [CommitAudio(), CreateResponse(), ClearAudio()]


async def test_session_accumulates_usage_and_requests() -> None:
    conn = FakeRealtimeConnection(
        [
            SessionUsageEvent(usage=RequestUsage(input_tokens=10, output_tokens=5)),
            SessionUsageEvent(usage=RequestUsage(input_tokens=3, output_tokens=2)),
            Transcript(text='ok', is_final=True),
            TurnCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    _ = [e async for e in session]
    assert session.usage.input_tokens == 13
    assert session.usage.output_tokens == 7
    assert session.usage.requests == 2
    # The turn's combined usage lands on the finalized assistant response.
    response = session.new_messages()[0]
    assert isinstance(response, ModelResponse)
    assert response.usage == snapshot(RequestUsage(input_tokens=13, output_tokens=7))


async def test_session_counts_tool_calls() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t', tool_name='f', args='{}'), TurnCompleteEvent()])

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'ok'

    session = RealtimeSession(conn, runner)
    _ = [e async for e in session]
    assert session.usage.tool_calls == 1


async def test_truncate_output_helper_forwards_to_connection() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    await session.truncate_output(640)
    assert conn.sent == [TruncateOutput(audio_end_ms=640)]


async def test_interrupt_truncates_before_cancel() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    await session.interrupt(audio_end_ms=800)
    # Truncate must precede cancel: cancel triggers response.done, which clears the tracked item.
    assert conn.sent == [TruncateOutput(audio_end_ms=800), CancelResponse()]


async def test_interrupt_without_audio_end_ms_only_cancels() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    await session.interrupt()
    assert conn.sent == [CancelResponse()]


# --- capability guards: unsupported operations raise before sending -----------------------------


async def test_manual_turn_control_guard() -> None:
    # A model without manual turn control (e.g. Gemini Live) rejects push-to-talk verbs up front.
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, profile=_profile(supports_manual_turn_control=False))
    for method in (session.commit_audio, session.clear_audio, session.create_response):
        with pytest.raises(UserError, match='does not support manual turn-taking'):
            await method()
    assert conn.sent == []  # nothing reached the connection


async def test_interruption_guard() -> None:
    # A model without interruption (e.g. Gemini Live) rejects barge-in cancellation up front.
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, profile=_profile(supports_interruption=False))
    with pytest.raises(UserError, match='does not support interruption'):
        await session.interrupt()
    assert conn.sent == []


async def test_output_truncation_guard() -> None:
    # A model that supports cancellation but not output truncation (e.g. xAI Grok Voice) rejects
    # `truncate_output()` and `interrupt(audio_end_ms=...)`, while a plain `interrupt()` still cancels.
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, profile=_profile(supports_output_truncation=False))
    with pytest.raises(UserError, match='does not support output truncation'):
        await session.truncate_output(100)
    with pytest.raises(UserError, match='does not support output truncation'):
        await session.interrupt(audio_end_ms=100)
    assert conn.sent == []
    await session.interrupt()
    assert conn.sent == [CancelResponse()]


async def test_image_input_guard() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, profile=_profile(supports_image_input=False))
    with pytest.raises(UserError, match='does not support image input'):
        await session.send_image(b'\xff\xd8')
    assert conn.sent == []


async def test_early_break_cancels_pump() -> None:
    conn = FakeRealtimeConnection([AudioDelta(data=b'\x00'), AudioDelta(data=b'\x01'), AudioDelta(data=b'\x02')])
    session = RealtimeSession(conn, _noop_runner)
    agen = cast(AsyncGenerator[Any], session.__aiter__())
    assert isinstance(await agen.__anext__(), PartStartEvent)
    await agen.aclose()  # exits the pump early without draining the rest


# --- audio retention ----------------------------------------------------------------------------


async def test_audio_retention_output_keeps_assistant_audio() -> None:
    conn = FakeRealtimeConnection(
        [
            AudioDelta(data=b'\x00\x01'),
            AudioDelta(data=b'\x02\x03'),
            Transcript(text='hi', is_final=True),
            TurnCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m', audio_retention='output')
    _ = [e async for e in session]
    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[
                    SpeechPart(
                        speaker='assistant',
                        transcript='hi',
                        audio=BinaryContent(data=b'\x00\x01\x02\x03', media_type='audio/pcm'),
                    )
                ],
                model_name='m',
                timestamp=IsDatetime(),
            )
        ]
    )


async def test_audio_retention_input_keeps_user_audio() -> None:
    conn = FakeRealtimeConnection([InputTranscript(text='hello', is_final=True)])
    session = RealtimeSession(conn, _noop_runner, audio_retention='input')
    # send_audio before the transcript finalizes buffers into the user part.
    await session.send_audio(b'\xaa\xbb')
    await session.send_audio(b'\xcc')
    _ = [e async for e in session]
    assert session.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SpeechPart(
                        speaker='user',
                        transcript='hello',
                        audio=BinaryContent(data=b'\xaa\xbb\xcc', media_type='audio/pcm'),
                    )
                ]
            )
        ]
    )


async def test_clear_audio_discards_retained_input() -> None:
    # `clear_audio()` must drop the locally retained buffer too, or discarded audio would still attach
    # to the next finalized user turn (with `audio_retention='input'`/`'both'`).
    conn = FakeRealtimeConnection([InputTranscript(text='hello', is_final=True)])
    session = RealtimeSession(conn, _noop_runner, audio_retention='input')
    await session.send_audio(b'\xaa\xbb')
    await session.clear_audio()  # discards the buffered chunk
    await session.send_audio(b'\xcc')  # only this survives into the finalized user turn
    _ = [e async for e in session]
    assert session.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SpeechPart(
                        speaker='user',
                        transcript='hello',
                        audio=BinaryContent(data=b'\xcc', media_type='audio/pcm'),
                    )
                ]
            )
        ]
    )


async def test_audio_only_user_turn_finalized_on_speech_stopped() -> None:
    # Transcription off + input audio retained: no `InputTranscript` arrives, so the user's turn is
    # finalized from the retained audio at the speech-stopped boundary (server VAD), as an audio-only
    # `SpeechPart` (no transcript). Bracketed with start/end, since there are no transcript deltas.
    conn = FakeRealtimeConnection(
        [SpeechStoppedEvent(), Transcript(text='Hi', is_final=True), TurnCompleteEvent()],
        input_transcription_enabled=False,
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m', audio_retention='input')
    await session.send_audio(b'\xaa\xbb')
    events = [e async for e in session]
    user_part = SpeechPart(speaker='user', audio=BinaryContent(data=b'\xaa\xbb', media_type='audio/pcm'))
    assert events[:3] == [
        PartStartEvent(index=0, part=user_part),
        PartEndEvent(index=0, part=user_part),
        SpeechStoppedEvent(),
    ]
    assert session.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[SpeechPart(speaker='user', audio=BinaryContent(data=b'\xaa\xbb', media_type='audio/pcm'))]
            ),
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='Hi')], model_name='m', timestamp=IsDatetime()
            ),
        ]
    )


async def test_audio_only_user_turn_finalized_on_turn_complete() -> None:
    # Providers without a speech-stopped signal (e.g. Gemini): the audio-only user turn is finalized at
    # the turn-complete boundary, before the assistant response, so history reads user-then-assistant.
    conn = FakeRealtimeConnection(
        [Transcript(text='Hi', is_final=True), TurnCompleteEvent()],
        input_transcription_enabled=False,
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m', audio_retention='input')
    await session.send_audio(b'\xaa\xbb')
    _ = [e async for e in session]
    assert session.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[SpeechPart(speaker='user', audio=BinaryContent(data=b'\xaa\xbb', media_type='audio/pcm'))]
            ),
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='Hi')], model_name='m', timestamp=IsDatetime()
            ),
        ]
    )


async def test_audio_retained_with_transcription_enabled_waits_for_transcript() -> None:
    # With transcription enabled, a speech-stopped boundary does NOT emit an audio-only turn: the turn is
    # finalized from the (asynchronously delivered) transcript instead, so there's exactly one user turn —
    # never a duplicate audio-only one racing the transcript.
    conn = FakeRealtimeConnection(
        [SpeechStoppedEvent(), InputTranscript(text='hello', is_final=True), TurnCompleteEvent()],
        input_transcription_enabled=True,
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m', audio_retention='input')
    await session.send_audio(b'\xaa\xbb')
    _ = [e async for e in session]
    assert session.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SpeechPart(
                        speaker='user',
                        transcript='hello',
                        audio=BinaryContent(data=b'\xaa\xbb', media_type='audio/pcm'),
                    )
                ]
            )
        ]
    )


async def test_no_transcription_and_no_input_retention_raises() -> None:
    # Transcription off + retention that doesn't keep input audio = user turns silently dropped. That
    # contradictory config is rejected up front rather than producing a lossy history.
    conn = FakeRealtimeConnection([], input_transcription_enabled=False)
    with pytest.raises(UserError, match="can't capture the user's turns"):
        RealtimeSession(conn, _noop_runner)  # default audio_retention='transcript_only'


async def test_no_transcription_with_input_retention_is_allowed() -> None:
    # Disabling transcription is fine as long as input audio is retained (audio-only user turns).
    conn = FakeRealtimeConnection([], input_transcription_enabled=False)
    RealtimeSession(conn, _noop_runner, audio_retention='input')  # no error


async def test_text_output_modality_produces_text_part() -> None:
    # A text-output response (`Transcript(output_text=True)`, from `output_modalities=('text',)`) must
    # be emitted and persisted as a `TextPart`, not a `SpeechPart` — it carries no speech.
    conn = FakeRealtimeConnection(
        [
            Transcript(text='hi', is_final=False, output_text=True),
            Transcript(text='hi there', is_final=True, output_text=True),
            TurnCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = [e async for e in session]
    starts = [e for e in events if isinstance(e, PartStartEvent)]
    assert len(starts) == 1 and isinstance(starts[0].part, TextPart)
    assert any(isinstance(e, PartDeltaEvent) and isinstance(e.delta, TextPartDelta) for e in events)
    ends = [e for e in events if isinstance(e, PartEndEvent)]
    assert len(ends) == 1 and isinstance(ends[0].part, TextPart)
    messages = session.new_messages()
    assert len(messages) == 1
    response = messages[0]
    assert isinstance(response, ModelResponse)
    assert response.parts == [TextPart(content='hi there')]


async def test_empty_assistant_turn_produces_no_response() -> None:
    # Audio with no transcript and no retention leaves the assistant part contentless, so the turn
    # finalizes without appending a `ModelResponse`.
    conn = FakeRealtimeConnection([AudioDelta(data=b'\x00\x01'), TurnCompleteEvent()])
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = [e async for e in session]
    assert [type(e).__name__ for e in events] == [
        'PartStartEvent',
        'PartDeltaEvent',
        'PartEndEvent',
        'TurnCompleteEvent',
    ]
    # The finalized part carries no transcript and no audio, so it isn't recorded.
    end = next(e for e in events if isinstance(e, PartEndEvent))
    assert isinstance(end.part, SpeechPart) and end.part.transcript is None and end.part.audio is None
    assert session.new_messages() == []


async def test_empty_input_transcript_produces_no_request() -> None:
    # A final input transcript that carries no text leaves the user part contentless, so nothing is
    # appended to history.
    conn = FakeRealtimeConnection([InputTranscript(text='', is_final=True)])
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = [e async for e in session]
    assert [type(e).__name__ for e in events] == ['PartStartEvent', 'PartEndEvent']
    end = next(e for e in events if isinstance(e, PartEndEvent))
    assert isinstance(end.part, SpeechPart) and end.part.transcript is None
    assert session.new_messages() == []


async def test_transcript_only_default_drops_audio() -> None:
    conn = FakeRealtimeConnection([AudioDelta(data=b'\x00'), Transcript(text='hi', is_final=True), TurnCompleteEvent()])
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    _ = [e async for e in session]
    response = session.new_messages()[0]
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts[0], SpeechPart)
    assert response.parts[0].audio is None


# --- seeding + handoff --------------------------------------------------------------------------


async def test_all_messages_includes_seed_new_messages_excludes_it() -> None:
    seed = [ModelRequest(parts=[UserPromptPart(content='earlier')])]
    conn = FakeRealtimeConnection([Transcript(text='reply', is_final=True), TurnCompleteEvent()])
    session = RealtimeSession(conn, _noop_runner, model_name='m', message_history=seed)
    _ = [e async for e in session]
    assert session.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='earlier', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='reply')],
                model_name='m',
                timestamp=IsDatetime(),
            ),
        ]
    )
    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='reply')],
                model_name='m',
                timestamp=IsDatetime(),
            )
        ]
    )


async def test_snapshot_is_a_copy() -> None:
    conn = FakeRealtimeConnection([Transcript(text='one', is_final=True), TurnCompleteEvent()])
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    _ = [e async for e in session]
    snapshot = session.new_messages()
    assert len(snapshot) == 1
    # `new_messages()` returns an independent copy: mutating the returned list must not leak back into
    # the session's own history.
    snapshot.append(ModelRequest(parts=[UserPromptPart(content='later')]))
    assert len(session.new_messages()) == 1


async def test_handoff_to_standard_agent_run() -> None:
    # A realtime session's history feeds straight into a normal agent run.
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='hello', is_final=True),
            Transcript(text='hi there', is_final=True),
            TurnCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='gpt-realtime')
    _ = [e async for e in session]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=f'seen {len(messages)} messages')])

    agent = Agent(FunctionModel(respond))
    result = await agent.run('continue', message_history=session.all_messages())
    assert result.output == snapshot('seen 3 messages')
    assert result.all_messages()[:2] == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user', transcript='hello')]),
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='hi there')],
                model_name='gpt-realtime',
                timestamp=IsDatetime(),
            ),
        ]
    )


def _grounding_parts() -> list[NativeToolCallPart | NativeToolReturnPart]:
    """The native tool parts a grounded Gemini turn produces (see `test_google.py` for the mapping)."""
    return [
        NativeToolCallPart(
            tool_name='web_search', args={'queries': ['weather rome']}, tool_call_id='g1', provider_name='google'
        ),
        NativeToolReturnPart(
            tool_name='web_search',
            content=[{'domain': 'example.com', 'title': 'Example', 'uri': 'https://example.com'}],
            tool_call_id='g1',
            provider_name='google',
        ),
    ]


async def test_grounding_folds_into_response_and_keeps_sources_event() -> None:
    # A grounded turn emits both `SourcesEvent` (UI) and `NativeToolParts` (history). The session yields
    # `SourcesEvent` unchanged but folds `NativeToolParts` into the assistant `ModelResponse`, ahead of the
    # speech, mirroring the classic `GoogleModel` — so `all_messages()` carries the native tool parts, not
    # just the speech.
    grounding = _grounding_parts()
    sources = SourcesEvent(sources=[WebSource(url='https://example.com', title='Example')], queries=['weather rome'])
    conn = FakeRealtimeConnection(
        [
            Transcript(text='It is sunny in Rome', is_final=True),
            sources,
            NativeToolParts(parts=list(grounding)),
            TurnCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='gemini-live-2.5-flash')
    events = [e async for e in session]

    # `SourcesEvent` passes through for the UI; `NativeToolParts` is folded into history, never yielded.
    assert sources in events
    assert not any(isinstance(e, NativeToolParts) for e in events)

    assert session.new_messages() == [
        ModelResponse(
            parts=[*grounding, SpeechPart(speaker='assistant', transcript='It is sunny in Rome')],
            model_name='gemini-live-2.5-flash',
            timestamp=IsDatetime(),
        )
    ]


async def test_grounded_history_hands_off_with_native_parts_intact() -> None:
    # The native tool parts from a grounded voice turn survive the `all_messages()` → `agent.run` handoff:
    # `Model.prepare_messages` passes `NativeToolCallPart`/`NativeToolReturnPart` through untouched (only
    # the `SpeechPart`s are converted to the plain user-prompt / text shapes any model can consume).
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='weather in rome?', is_final=True),
            Transcript(text='It is sunny in Rome', is_final=True),
            NativeToolParts(parts=list(_grounding_parts())),
            TurnCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='gemini-live-2.5-flash')
    _ = [e async for e in session]

    received: list[ModelMessage] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received.extend(messages)
        return ModelResponse(parts=[TextPart(content='ok')])

    agent = Agent(FunctionModel(respond))
    await agent.run('and now?', message_history=session.all_messages())

    assert received == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='weather in rome?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['weather rome']},
                        tool_call_id='g1',
                        provider_name='google',
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content=[{'domain': 'example.com', 'title': 'Example', 'uri': 'https://example.com'}],
                        tool_call_id='g1',
                        timestamp=IsDatetime(),
                        provider_name='google',
                    ),
                    TextPart(content='It is sunny in Rome'),
                ],
                model_name='gemini-live-2.5-flash',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='and now?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def _code_execution_parts() -> list[NativeToolCallPart | NativeToolReturnPart]:
    """The native tool parts a code-execution Gemini turn produces (see `test_google.py` for the mapping)."""
    return [
        NativeToolCallPart(
            tool_name='code_execution',
            args={'code': 'print(1 + 1)', 'language': 'PYTHON'},
            tool_call_id='c1',
            provider_name='google',
        ),
        NativeToolReturnPart(
            tool_name='code_execution',
            content={'outcome': 'OUTCOME_OK', 'output': '2\n'},
            tool_call_id='c1',
            provider_name='google',
        ),
    ]


async def test_code_execution_history_hands_off_with_native_parts_intact() -> None:
    # A code-execution voice turn writes the `NativeToolCallPart`/`NativeToolReturnPart` pair into history
    # (ahead of the speech, like the classic `GoogleModel`), and those parts survive the `all_messages()`
    # → `agent.run` handoff untouched by `Model.prepare_messages` — the same path grounding takes.
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='what is 1 + 1?', is_final=True),
            Transcript(text='The answer is 2.', is_final=True),
            NativeToolParts(parts=list(_code_execution_parts())),
            TurnCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='gemini-live-2.5-flash')
    _ = [e async for e in session]

    received: list[ModelMessage] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received.extend(messages)
        return ModelResponse(parts=[TextPart(content='ok')])

    agent = Agent(FunctionModel(respond))
    await agent.run('and 2 + 2?', message_history=session.all_messages())

    assert received == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='what is 1 + 1?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        tool_name='code_execution',
                        args={'code': 'print(1 + 1)', 'language': 'PYTHON'},
                        tool_call_id='c1',
                        provider_name='google',
                    ),
                    NativeToolReturnPart(
                        tool_name='code_execution',
                        content={'outcome': 'OUTCOME_OK', 'output': '2\n'},
                        tool_call_id='c1',
                        timestamp=IsDatetime(),
                        provider_name='google',
                    ),
                    TextPart(content='The answer is 2.'),
                ],
                model_name='gemini-live-2.5-flash',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='and 2 + 2?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


# --- Agent.realtime_session integration --------------------------------------------------------


async def test_agent_realtime_session_wires_tools_and_instructions() -> None:
    agent: Agent[None, str] = Agent(instructions='You are a helpful assistant.')

    @agent.tool_plain
    def greet(name: str) -> str:
        """Greet someone."""
        return f'Hello {name}!'

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='tc_5', tool_name='greet', args='{"name": "Alice"}'), TurnCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)

    async with agent.realtime_session(model=model) as session:
        events = [e async for e in session]

    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert result.part.content == 'Hello Alice!'
    assert model.last_instructions == 'You are a helpful assistant.'
    assert model.last_tools is not None
    assert 'greet' in [t.name for t in model.last_tools]


async def test_agent_realtime_session_seeds_message_history() -> None:
    agent: Agent[None, str] = Agent()
    seed = [
        ModelRequest(parts=[UserPromptPart(content='earlier question')]),
        ModelResponse(parts=[TextPart(content='earlier answer')]),
    ]
    conn = FakeRealtimeConnection([TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model, message_history=seed) as session:
        _ = [e async for e in session]
        assert session.all_messages() == seed  # seeded into the session's history
    assert model.last_messages == seed  # forwarded to the provider for wire-level seeding


async def test_agent_realtime_session_rejects_seeding_when_unsupported() -> None:
    # A model that can't seed a session rejects `message_history` up front, before dialing.
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([TurnCompleteEvent()])
    model = FakeRealtimeModel(conn, profile=_profile(supports_session_seeding=False))
    seed = [ModelRequest(parts=[UserPromptPart(content='earlier question')])]
    with pytest.raises(UserError, match='does not support seeding a session'):
        async with agent.realtime_session(model=model, message_history=seed):
            pass  # pragma: no cover — enter raises before yielding


async def test_agent_realtime_session_audio_retention_forwarded() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([AudioDelta(data=b'\x07'), Transcript(text='hi', is_final=True), TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model, audio_retention='output') as session:
        _ = [e async for e in session]
        response = session.new_messages()[0]
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts[0], SpeechPart)
    assert response.parts[0].audio == BinaryContent(data=b'\x07', media_type='audio/pcm')


async def test_agent_realtime_session_additional_instructions() -> None:
    agent: Agent[None, str] = Agent(instructions='Default')
    conn = FakeRealtimeConnection([TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model, instructions='Custom') as session:
        _ = [e async for e in session]
    assert model.last_instructions == 'Default\nCustom'


async def test_agent_realtime_session_default_instructions_empty() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        _ = [e async for e in session]
    assert model.last_instructions == ''


async def test_agent_realtime_session_unknown_tool() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='tc_x', tool_name='nonexistent', args='{}'), TurnCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        events = [e async for e in session]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert 'Unknown tool name' in str(result.part.content)
    assert 'nonexistent' in str(result.part.content)


async def test_agent_realtime_session_tool_exception() -> None:
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def explode() -> str:
        raise ValueError('nope')

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='explode', args='{}'), TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        events = [e async for e in session]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert 'Error' in str(result.part.content) and 'nope' in str(result.part.content)


async def test_agent_realtime_session_validates_and_coerces_args() -> None:
    agent: Agent[None, str] = Agent()
    seen: int | None = None

    @agent.tool_plain
    def double(x: int) -> str:
        nonlocal seen
        seen = x
        return str(x * 2)

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='tc', tool_name='double', args='{"x": "21"}'), TurnCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        events = [e async for e in session]

    assert seen == 21
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert result.part.content == '42'


async def test_agent_realtime_session_invalid_args_return_retry_message() -> None:
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def double(x: int) -> str:  # pragma: no cover — never reached; validation fails first
        return str(x * 2)

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='tc', tool_name='double', args='{"x": "not a number"}'), TurnCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        events = [e async for e in session]

    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert 'validation error' in str(result.part.content)


async def test_agent_realtime_session_runs_args_validator() -> None:
    agent: Agent[None, str] = Agent()

    def guard(ctx: RunContext[Any], city: str) -> None:
        raise ModelRetry('not allowed')

    @agent.tool_plain(args_validator=guard)
    def weather(city: str) -> str:  # pragma: no cover — never reached; the validator rejects first
        return f'sunny in {city}'

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='tc', tool_name='weather', args='{"city": "forbidden"}'), TurnCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        events = [e async for e in session]

    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert 'not allowed' in str(result.part.content)


async def test_agent_realtime_session_tool_return_is_unwrapped() -> None:
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def info() -> ToolReturn:
        return ToolReturn(return_value='the-value', content=['extra context'])

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='info', args='{}'), TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        events = [e async for e in session]

    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert result.part.content == 'the-value'


async def test_agent_realtime_session_denied_tool_returns_denial_message() -> None:
    from pydantic_ai.capabilities import HandleDeferredToolCalls
    from pydantic_ai.exceptions import ApprovalRequired
    from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults

    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def danger() -> str:
        raise ApprovalRequired()

    def deny(ctx: RunContext[Any], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: False for call in requests.approvals})

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='danger', args='{}'), TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model, capabilities=[HandleDeferredToolCalls(handler=deny)]) as session:
        events = [e async for e in session]

    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert 'denied' in str(result.part.content).lower()


async def test_agent_realtime_session_resolves_per_run_toolsets() -> None:
    agent: Agent[str, str] = Agent(deps_type=str)

    @agent.toolset
    def per_run(ctx: RunContext[str]) -> FunctionToolset[str]:
        assert ctx.deps == 'alice'  # the factory sees the run deps
        ts: FunctionToolset[str] = FunctionToolset()

        @ts.tool
        def whoami(tool_ctx: RunContext[str]) -> str:
            return f'deps={tool_ctx.deps}'

        return ts

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='whoami', args='{}'), TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model, deps='alice') as session:
        events = [e async for e in session]

    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert result.part.content == 'deps=alice'


async def test_agent_realtime_session_model_visible_to_tools() -> None:
    agent: Agent[None, str] = Agent()
    seen_name: str | None = None

    @agent.tool
    async def inspect_model(ctx: RunContext) -> str:
        nonlocal seen_name
        seen_name = ctx.model.model_name
        return f'system={ctx.model.system}'

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='tc', tool_name='inspect_model', args='{}'), TurnCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        events = [e async for e in session]

    assert seen_name == 'fake-realtime'
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert result.part.content == 'system=fake'


async def test_agent_realtime_session_uses_realtime_model_when_text_model_set() -> None:
    agent: Agent[None, str] = Agent('test')
    seen_system: str | None = None

    @agent.tool
    async def inspect_model(ctx: RunContext) -> str:
        nonlocal seen_system
        seen_system = ctx.model.system
        return 'ok'

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='tc', tool_name='inspect_model', args='{}'), TurnCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        _ = [e async for e in session]
    assert seen_system == 'fake'


async def test_agent_realtime_session_background_tools_end_to_end() -> None:
    agent: Agent[None, str] = Agent()
    release = asyncio.Event()

    @agent.tool_plain
    async def slow_lookup() -> str:
        await release.wait()
        return 'background result'

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='bg', tool_name='slow_lookup', args='{}'), TurnCompleteEvent()],
        release=release,
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model, background_tools={'slow_lookup'}) as session:
        events = [e async for e in session]

    assert [type(e).__name__ for e in events] == snapshot(
        ['PartStartEvent', 'PartEndEvent', 'FunctionToolCallEvent', 'TurnCompleteEvent', 'FunctionToolResultEvent']
    )
    result = events[-1]
    assert isinstance(result, FunctionToolResultEvent)
    assert result.part.content == 'background result'


async def test_agent_realtime_session_forwards_model_settings() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    settings = RealtimeModelSettings(max_tokens=50)
    async with agent.realtime_session(model=model, model_settings=settings) as session:
        _ = [e async for e in session]
    assert model.last_model_settings == settings


async def test_agent_realtime_session_merges_model_and_call_settings() -> None:
    """Call-time realtime settings override the model defaults key by key."""
    agent: Agent[None, str] = Agent(model_settings=ModelSettings(temperature=0.1, max_tokens=100))
    conn = FakeRealtimeConnection([TurnCompleteEvent()])
    model = FakeRealtimeModel(conn, settings=RealtimeModelSettings(max_tokens=100, parallel_tool_calls=False))
    async with agent.realtime_session(
        model=model, model_settings=RealtimeModelSettings(parallel_tool_calls=True)
    ) as session:
        _ = [e async for e in session]
    assert model.last_model_settings == RealtimeModelSettings(max_tokens=100, parallel_tool_calls=True)


async def test_agent_realtime_session_ignores_regular_model_settings_override() -> None:
    """`Agent.override(model_settings=...)` does not affect realtime settings."""
    agent: Agent[None, str] = Agent(model_settings=ModelSettings(temperature=0.1))
    conn = FakeRealtimeConnection([TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    with agent.override(model_settings=ModelSettings(temperature=0.9)):
        async with agent.realtime_session(model=model, model_settings=RealtimeModelSettings(max_tokens=50)) as session:
            _ = [e async for e in session]
    assert model.last_model_settings == RealtimeModelSettings(max_tokens=50)


async def test_agent_realtime_session_send_audio() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        await session.send_audio(b'\xab\xcd')
    assert conn.sent == [AudioInput(data=b'\xab\xcd')]


# --- parity with run/iter: instructions, toolsets, usage, usage_limits, capabilities, metadata ---


async def test_agent_realtime_session_dynamic_instructions() -> None:
    agent: Agent[None, str] = Agent(instructions='Base')

    @agent.instructions
    def extra() -> str:
        return 'Dynamic'

    conn = FakeRealtimeConnection([TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        _ = [e async for e in session]
    assert model.last_instructions == 'Base\nDynamic'


async def test_agent_realtime_session_additional_toolsets() -> None:
    agent: Agent[None, str] = Agent()
    extra_toolset: FunctionToolset[object] = FunctionToolset()

    @extra_toolset.tool_plain
    def extra_tool() -> str:
        return 'x'

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t1', tool_name='extra_tool', args='{}'), TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model, toolsets=[extra_toolset]) as session:
        events = [e async for e in session]
    assert 'extra_tool' in [t.name for t in model.last_tools or []]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert result.part.content == 'x'  # the extra toolset's tool is offered AND callable


async def test_agent_realtime_session_external_usage_accumulates() -> None:
    usage = RunUsage()
    conn = FakeRealtimeConnection(
        [SessionUsageEvent(usage=RequestUsage(input_tokens=7, output_tokens=3)), TurnCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    agent: Agent[None, str] = Agent()
    async with agent.realtime_session(model=model, usage=usage) as session:
        assert session.usage is usage  # the provided accumulator is used
        _ = [e async for e in session]
    assert usage.input_tokens == 7
    assert usage.output_tokens == 3


async def test_agent_realtime_session_token_limit_emits_session_error() -> None:
    conn = FakeRealtimeConnection(
        [SessionUsageEvent(usage=RequestUsage(input_tokens=100, output_tokens=100)), TurnCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    agent: Agent[None, str] = Agent()
    async with agent.realtime_session(model=model, usage_limits=UsageLimits(total_tokens_limit=50)) as session:
        events = [e async for e in session]
    assert len(events) == 1
    assert isinstance(events[0], SessionErrorEvent)
    assert events[0].recoverable is False and events[0].type == 'usage_limit_exceeded'


async def test_agent_realtime_session_tool_call_limit_emits_session_error() -> None:
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def greet() -> str:  # pragma: no cover - never runs: the limit trips first
        return 'hi'

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t1', tool_name='greet', args='{}'), TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model, usage_limits=UsageLimits(tool_calls_limit=0)) as session:
        events = [e async for e in session]
    assert len(events) == 1
    assert isinstance(events[0], SessionErrorEvent) and events[0].recoverable is False


async def test_agent_realtime_session_usage_limits_within_budget() -> None:
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def greet() -> str:
        return 'hi'

    conn = FakeRealtimeConnection(
        [
            SessionUsageEvent(usage=RequestUsage(input_tokens=1, output_tokens=1)),
            ToolCall(tool_call_id='t1', tool_name='greet', args='{}'),
            TurnCompleteEvent(),
        ]
    )
    model = FakeRealtimeModel(conn)
    limits = UsageLimits(total_tokens_limit=1000, tool_calls_limit=10)
    async with agent.realtime_session(model=model, usage_limits=limits) as session:
        events = [e async for e in session]
    assert not any(isinstance(e, SessionErrorEvent) for e in events)
    assert any(isinstance(e, FunctionToolResultEvent) for e in events)


async def test_agent_realtime_session_native_tools_from_capability() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model, capabilities=[NativeTool(WebSearchTool())]) as session:
        _ = [e async for e in session]
    assert model.last_native_tools is not None
    assert any(isinstance(t, WebSearchTool) for t in model.last_native_tools)


async def test_agent_realtime_session_rejects_unsupported_native_tool() -> None:
    # A native tool the model doesn't support (per its `supported_native_tools` profile) fails up front
    # with the uniform error, before connecting — even when contributed by a capability. This is the
    # signal a caller/capability needs to fall back; the session doesn't fall back automatically.
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([])
    model = FakeRealtimeModel(conn, profile=_profile(supported_native_tools=frozenset()))
    with pytest.raises(
        UserError,
        match=r"'fake-realtime' realtime model does not support the WebSearchTool native tool\(s\)\. "
        r'Supported native tools: none\.',
    ):
        async with agent.realtime_session(model=model, capabilities=[NativeTool(WebSearchTool())]):
            pass  # pragma: no cover - validation raises before yielding


async def test_agent_realtime_session_local_capability_tool_declared() -> None:
    def fetch(url: str) -> str:
        return f'content of {url}'  # pragma: no cover - not executed in this wiring test

    agent: Agent[None, str] = Agent(capabilities=[WebFetch(native=False, local=fetch)])
    conn = FakeRealtimeConnection([TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        _ = [e async for e in session]
    assert model.last_tools is not None
    assert 'fetch' in [t.name for t in model.last_tools]
    assert model.last_native_tools == []  # native=False -> no url_context forwarded


class _HookCapability(AbstractCapability[object]):
    """Records and rewrites tool execution through the tool-lifecycle hooks."""

    def __init__(self) -> None:
        self.events: list[str] = []

    async def before_tool_execute(
        self, ctx: RunContext[object], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> dict[str, Any]:
        self.events.append(f'before:{call.tool_name}')
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[object],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        self.events.append(f'after:{result}')
        return f'[hooked] {result}'


async def test_agent_realtime_session_capability_tool_hooks() -> None:
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def greet() -> str:
        return 'hi'

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t1', tool_name='greet', args='{}'), TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    cap = _HookCapability()
    async with agent.realtime_session(model=model, capabilities=[cap]) as session:
        events = [e async for e in session]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert result.part.content == '[hooked] hi'
    assert cap.events == ['before:greet', 'after:hi']


async def test_agent_realtime_session_metadata_and_conversation_id() -> None:
    agent: Agent[None, str] = Agent()

    @agent.tool
    def whoami(ctx: RunContext) -> str:
        return f'{ctx.conversation_id}|{ctx.metadata}'

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t1', tool_name='whoami', args='{}'), TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model, conversation_id='conv-1', metadata={'tier': 'gold'}) as session:
        events = [e async for e in session]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert 'conv-1' in str(result.part.content)
    assert 'gold' in str(result.part.content)


async def test_agent_realtime_session_native_tools_override_honored() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    with agent.override(native_tools=[WebSearchTool()]):
        async with agent.realtime_session(model=model) as session:
            _ = [e async for e in session]
    assert model.last_native_tools is not None
    assert any(isinstance(t, WebSearchTool) for t in model.last_native_tools)


async def test_wrapper_agent_realtime_session_proxies() -> None:
    from pydantic_ai.agent import WrapperAgent

    inner: Agent[None, str] = Agent(instructions='Inner')
    wrapper = WrapperAgent(inner)
    conn = FakeRealtimeConnection([TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with wrapper.realtime_session(model=model) as session:
        _ = [e async for e in session]
    assert model.last_instructions == 'Inner'  # the wrapped agent's session was used


async def test_agent_realtime_session_drops_auto_injected_tool_search() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([TurnCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime_session(model=model) as session:
        _ = [e async for e in session]
    assert model.last_native_tools == []
