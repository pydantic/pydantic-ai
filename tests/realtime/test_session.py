"""Tests for `RealtimeSession`: event translation, history assembly, tool dispatch, and `Agent.realtime_session`."""

from __future__ import annotations as _annotations

import asyncio
import io
import wave
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from threading import Event as ThreadEvent
from typing import Any, Literal, TypeVar

import anyio
import pytest
from inline_snapshot import snapshot
from pydantic_core import SchemaValidator, core_schema

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai._agent_graph import resolve_conversation_id
from pydantic_ai._enqueue import PendingMessage
from pydantic_ai._instrumentation import get_instructions
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls, NativeTool, WebFetch
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelAPIError,
    ToolFailed,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import (
    BinaryContent,
    BinaryImage,
    DeferredToolRequestsEvent,
    DeferredToolResultsEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SpeechPart,
    SpeechPartDelta,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import AbstractNativeTool, CodeExecutionTool, WebFetchTool, WebSearchTool
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.realtime import (
    AudioInput,
    InputSpeechEndEvent,
    InputSpeechStartEvent,
    InputTranscriptionErrorEvent,
    RealtimeError,
    RealtimeEvent,
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
    RealtimeSession as _RealtimeSession,
    ResponseCompleteEvent,
    SessionReconnectEvent,
    SessionUsageEvent,
    TranscriptUpdate,
    TurnCompleteEvent,
)
from pydantic_ai.realtime._base import (
    ConversationCreated,
    ConversationItemCreated,
    ImageInput,
    SessionErrorEvent,
    TextInput,
    resolve_advertised_tools,
    seed_speech_content,
)
from pydantic_ai.realtime.codec import (
    AudioDelta,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    ToolCall,
    ToolCallCancelled,
    ToolResult,
    TruncateOutput,
)
from pydantic_ai.settings import ModelSettings, ToolOrOutput
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from ..conftest import IsDatetime, IsStr

pytestmark = pytest.mark.anyio
T = TypeVar('T')


def test_resolve_advertised_tools_enforces_tool_choice() -> None:
    tools = [
        ToolDefinition(name='weather', parameters_json_schema={'type': 'object'}),
        ToolDefinition(name='time', parameters_json_schema={'type': 'object'}),
    ]

    assert resolve_advertised_tools(tools, None) == (tools, None)
    assert resolve_advertised_tools(tools, 'auto') == (tools, 'auto')
    assert resolve_advertised_tools(tools, 'required') == (tools, 'required')
    assert resolve_advertised_tools(tools, 'none') == ([], 'none')
    assert resolve_advertised_tools(tools, []) == ([], 'none')
    assert resolve_advertised_tools(tools, ['weather']) == (tools[:1], ('required', {'weather'}))
    assert resolve_advertised_tools(tools, ToolOrOutput(['time'])) == (tools[1:], ('auto', {'time'}))
    assert resolve_advertised_tools(None, None) == ([], None)


def test_resolve_advertised_tools_rejects_a_tool_choice_it_cannot_honor() -> None:
    """Same errors as a standard run: a tool_choice that names nothing real is a bug, not a no-op."""
    tools = [ToolDefinition(name='weather', parameters_json_schema={'type': 'object'})]

    with pytest.raises(UserError, match='not_a_tool'):
        resolve_advertised_tools(tools, ['not_a_tool'])
    with pytest.raises(UserError, match='no function tools are defined'):
        resolve_advertised_tools(None, 'required')


def _wav_content(pcm: bytes, sample_rate: int = 24000) -> BinaryContent:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return BinaryContent(data=buffer.getvalue(), media_type='audio/wav')


async def _noop_runner(name: str, args: dict[str, Any], call_id: str) -> str:  # pragma: no cover
    raise AssertionError('tool runner should not be called')


_TEST_TOOL_NAMES = {
    'boom',
    'f',
    'fast',
    'get_weather',
    'hang',
    'noop',
    'slow',
}


class _RunnerToolset(AbstractToolset[None]):
    """Adapt legacy-shaped test callables to the real tool-management path."""

    def __init__(self, runner: Any):
        self.runner = runner

    @property
    def id(self) -> str:
        return 'realtime-test-runner'

    async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
        return {name: _toolset_tool(self, name) for name in _TEST_TOOL_NAMES}

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
    ) -> Any:
        assert ctx.tool_call_id is not None
        return await self.runner(name, tool_args, ctx.tool_call_id)


def _toolset_tool(toolset: AbstractToolset[None], name: str) -> ToolsetTool[None]:
    return ToolsetTool(
        toolset=toolset,
        tool_def=ToolDefinition(name=name, parameters_json_schema={'type': 'object', 'additionalProperties': True}),
        max_retries=1,
        args_validator=SchemaValidator(core_schema.dict_schema()),
    )


def make_tool_manager(runner: Any = _noop_runner) -> ToolManager[None]:
    toolset = _RunnerToolset(runner)
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_step=0)
    manager = ToolManager(toolset, ctx=ctx, tools={name: _toolset_tool(toolset, name) for name in _TEST_TOOL_NAMES})
    ctx.tool_manager = manager
    return manager


def test_runner_toolset_has_stable_id() -> None:
    assert _RunnerToolset(_noop_runner).id == 'realtime-test-runner'


def RealtimeSession(connection: RealtimeConnection, runner: Any = _noop_runner, **kwargs: Any) -> _RealtimeSession:
    """Construct a session with the real `ToolManager` API while keeping test setup compact."""
    return _RealtimeSession(connection, make_tool_manager(runner), **kwargs)


async def collect_events(session: _RealtimeSession) -> list[RealtimeEvent]:
    """Enter a directly constructed session and drain its public event iterator."""
    async with session:
        return [event async for event in session]


async def drain_events(session: _RealtimeSession) -> list[RealtimeEvent]:
    """Drain an already-entered session."""
    return [event async for event in session]


async def aiter_to_list(iterator: AsyncIterator[T]) -> list[T]:
    return [item async for item in iterator]


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
        events: list[RealtimeCodecEvent],
        *,
        release: asyncio.Event | None = None,
        input_transcription_enabled: bool = True,
        model_name: str | None = None,
    ) -> None:
        self._events = events
        self._release = release
        self._input_transcription_enabled = input_transcription_enabled
        self._model_name = model_name
        self.sent: list[RealtimeInput] = []

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def input_transcription_enabled(self) -> bool:
        return self._input_transcription_enabled

    async def send(self, content: RealtimeInput) -> None:
        self.sent.append(content)

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        for event in self._events:
            yield event
        if self._release is not None:
            self._release.set()


class BlockingRealtimeConnection(FakeRealtimeConnection):
    """Replay fixed events, then remain open until the session closes."""

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        async for event in super().__aiter__():
            yield event
        await asyncio.Event().wait()


class FakeRealtimeModel(RealtimeModel):
    """A model that yields a pre-built connection and records connect arguments."""

    def __init__(
        self,
        connection: RealtimeConnection,
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
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[RealtimeConnection]:
        self.last_instructions = get_instructions(messages) or ''
        self.last_tools = model_request_parameters.function_tools
        self.last_native_tools = model_request_parameters.native_tools
        self.last_model_settings = model_settings
        self.last_messages = messages
        yield self._connection


# --- event translation -------------------------------------------------------------------------


async def test_consumption_views_run_concurrently_with_event_stream() -> None:
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='hello', is_final=True),
            AudioDelta(b'audio-1'),
            OutputTranscript(text='hi', is_final=False),
            AudioDelta(b'audio-2'),
            OutputTranscript(text='hi there', is_final=True),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn)

    async with session:
        events, audio, transcripts, deltas = await asyncio.gather(
            drain_events(session),
            aiter_to_list(session.stream_audio()),
            aiter_to_list(session.stream_transcripts()),
            aiter_to_list(session.stream_transcripts(delta=True)),
        )

        assert events == [
            PartStartEvent(index=0, part=SpeechPart(speaker='user', transcript='')),
            PartDeltaEvent(
                index=0, delta=SpeechPartDelta(speaker='user', transcript_delta='hello', transcript='hello')
            ),
            PartEndEvent(index=0, part=SpeechPart(speaker='user', transcript='hello')),
            PartStartEvent(index=1, part=SpeechPart(speaker='assistant', transcript='')),
            PartDeltaEvent(index=1, delta=SpeechPartDelta(speaker='assistant', audio_chunk=b'audio-1')),
            PartDeltaEvent(index=1, delta=SpeechPartDelta(speaker='assistant', transcript_delta='hi', transcript='hi')),
            PartDeltaEvent(index=1, delta=SpeechPartDelta(speaker='assistant', audio_chunk=b'audio-2')),
            PartDeltaEvent(
                index=1, delta=SpeechPartDelta(speaker='assistant', transcript_delta=' there', transcript='hi there')
            ),
            PartEndEvent(index=1, part=SpeechPart(speaker='assistant', transcript='hi there')),
            ResponseCompleteEvent(),
            TurnCompleteEvent(),
        ]
        assert audio == [b'audio-1', b'audio-2']
        assert transcripts == [
            SpeechPart(speaker='user', transcript='hello'),
            SpeechPart(speaker='assistant', transcript='hi there'),
        ]
        # Each update names its turn, so a caption UI keeps the two speakers apart, and carries the
        # turn's text so far so it can replace instead of accumulating itself.
        assert deltas == [
            TranscriptUpdate(index=0, speaker='user', delta='hello', transcript='hello'),
            TranscriptUpdate(index=1, speaker='assistant', delta='hi', transcript='hi'),
            TranscriptUpdate(index=1, speaker='assistant', delta=' there', transcript='hi there'),
        ]


async def test_cumulative_transcripts_revise_the_turn_instead_of_doubling_up() -> None:
    """A provider that revises its own partials reaches captions as a replacement, not an append.

    xAI Grok Voice streams the user's transcript as cumulative snapshots and corrects earlier words
    (`'Hello?'` becomes `'Hello, my name is'`). Appending that would double the corrected text, so the
    session replaces — including on the `stream_transcripts` view, which is what captions render.
    """
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='Hello?', cumulative=True),
            InputTranscript(text='Hello, my name is', cumulative=True),
            InputTranscript(text='Hello, my name is Marcelo.', cumulative=True, is_final=True),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn)

    async with session:
        events, transcripts, deltas = await asyncio.gather(
            drain_events(session),
            aiter_to_list(session.stream_transcripts()),
            aiter_to_list(session.stream_transcripts(delta=True)),
        )

    assert [event for event in events if isinstance(event, PartDeltaEvent)] == [
        PartDeltaEvent(index=0, delta=SpeechPartDelta(speaker='user', transcript_delta='Hello?', transcript='Hello?')),
        # The revision added no text, so only the corrected whole is reported.
        PartDeltaEvent(
            index=0, delta=SpeechPartDelta(speaker='user', transcript_delta='', transcript='Hello, my name is')
        ),
        PartDeltaEvent(
            index=0,
            delta=SpeechPartDelta(
                speaker='user', transcript_delta=' Marcelo.', transcript='Hello, my name is Marcelo.'
            ),
        ),
    ]
    # The revision reaches the caption view, and `transcript` is the turn's real text throughout.
    assert deltas == [
        TranscriptUpdate(index=0, speaker='user', delta='Hello?', transcript='Hello?'),
        # A revision adds nothing, so `delta` is empty and `transcript` carries the correction.
        TranscriptUpdate(index=0, speaker='user', delta='', transcript='Hello, my name is'),
        TranscriptUpdate(index=0, speaker='user', delta=' Marcelo.', transcript='Hello, my name is Marcelo.'),
    ]
    assert transcripts == [SpeechPart(speaker='user', transcript='Hello, my name is Marcelo.')]


async def test_cumulative_transcript_repeating_itself_emits_nothing() -> None:
    """An unchanged snapshot is not news; re-emitting it would flicker a caption for no reason."""
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='Hello', cumulative=True),
            InputTranscript(text='Hello', cumulative=True),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn)

    async with session:
        events, deltas = await asyncio.gather(
            drain_events(session), aiter_to_list(session.stream_transcripts(delta=True))
        )

    assert [event for event in events if isinstance(event, PartDeltaEvent)] == [
        PartDeltaEvent(index=0, delta=SpeechPartDelta(speaker='user', transcript_delta='Hello', transcript='Hello'))
    ]
    assert deltas == [TranscriptUpdate(index=0, speaker='user', delta='Hello', transcript='Hello')]


async def test_audio_view_drops_oldest_chunk_on_overflow_without_instrumentation() -> None:
    chunks = [bytes([index]) for index in range(40)]
    session = RealtimeSession(FakeRealtimeConnection([AudioDelta(chunk) for chunk in chunks]))

    async with session:
        assert [chunk async for chunk in session.stream_audio()] == chunks[-32:]
        assert len(await drain_events(session)) == 41


async def test_final_transcripts_survive_a_flood_of_deltas() -> None:
    # The two speakers' transcripts stream at once, so a finalized user turn is routinely followed by
    # a long run of assistant deltas. Final parts and deltas are separate subscriptions for exactly
    # this reason: sharing one bounded queue and filtering on the way out let those deltas evict the
    # finalized part a `delta=False` consumer was waiting for, losing it silently.
    words = [f'word{index} ' for index in range(60)]
    session = RealtimeSession(
        FakeRealtimeConnection(
            [
                InputTranscript(text='what is the weather', is_final=True),
                *[OutputTranscript(text=''.join(words[: index + 1]), is_final=False) for index in range(len(words))],
                OutputTranscript(text=''.join(words), is_final=True),
                ResponseCompleteEvent(),
            ]
        )
    )

    async with session:
        transcripts, _ = await asyncio.gather(
            aiter_to_list(session.stream_transcripts()),
            drain_events(session),
        )

    # Both speakers' finalized turns arrive, despite far more deltas than any single window holds.
    assert [(part.speaker, part.transcript) for part in transcripts] == [
        ('user', 'what is the weather'),
        # Assistant transcripts are recorded verbatim; only user turns are stripped at finalization.
        ('assistant', ''.join(words)),
    ]


async def test_send_only_session_still_runs_tools() -> None:
    # Tool execution, turn tracking, and usage limits all run off the receive loop, and a caller who
    # sends a prompt and then goes off to do something else has no reason to think iterating is what
    # switches the agent on. Sending starts the loop, so the tool runs without anyone consuming.
    ran = asyncio.Event()

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        ran.set()
        return 'done'

    conn = BlockingRealtimeConnection([ToolCall(tool_call_id='c1', tool_name='fast', args='{}')])
    async with RealtimeSession(conn, runner) as session:
        await session.send('do the task')
        # No `async for` and no view: the agent still has to run the tool and return its result.
        with anyio.fail_after(5):
            await ran.wait()
            while len(conn.sent) < 2:
                await asyncio.sleep(0)
    assert [type(sent).__name__ for sent in conn.sent] == ['TextInput', 'ToolResult']


async def test_close_discards_buffered_view_items() -> None:
    # Closing is a hangup, not a flush: whatever a view had buffered is dropped rather than delivered
    # after the session is over. This is the case where a consumer is behind at close time, so the
    # queue is non-empty -- distinct from closing a view that has already caught up.
    session = RealtimeSession(BlockingRealtimeConnection([AudioDelta(bytes([index])) for index in range(5)]))

    async with session:
        stream = session.stream_audio()
        assert await anext(stream) == b'\x00'
        # Let the pump run so the rest of the chunks pile up behind the consumer.
        for _ in range(10):
            await asyncio.sleep(0)
        await session.close()
        assert [chunk async for chunk in stream] == []


async def test_session_reports_the_connected_model_profile() -> None:
    """A session exposes the model's profile, because it may be all the caller holds.

    `agent.realtime('google:…')` takes a model *name* and builds the model internally, so without this
    there is nothing to read the input sample rate (or any capability flag) off of.
    """
    profile = RealtimeModelProfile(supports_interruption=False, audio_input_sample_rate=16000)
    model = FakeRealtimeModel(FakeRealtimeConnection([]), profile=profile)
    agent = Agent()
    async with agent.realtime(model).session() as session:
        assert session.profile == profile
        assert session.profile.get('audio_input_sample_rate') == 16000


async def test_close_ends_views_and_is_idempotent() -> None:
    session = RealtimeSession(BlockingRealtimeConnection([AudioDelta(b'audio')]))

    async with session:
        stream = session.stream_audio()
        assert await anext(stream) == b'audio'
        transcript_task = asyncio.create_task(aiter_to_list(session.stream_transcripts()))
        await asyncio.sleep(0)
        assert session.closed is False
        await session.close()
        await session.close()
        assert session.closed is True
        assert [chunk async for chunk in stream] == []
        assert await transcript_task == []
        with pytest.raises(UserError, match='This realtime session is closed'):
            await session.send_audio(b'more audio')
        with pytest.raises(UserError, match='closed and cannot be streamed'):
            await anext(session.stream_audio())


async def test_view_is_lazy_and_does_not_replay_events() -> None:
    session = RealtimeSession(FakeRealtimeConnection([AudioDelta(b'audio')]))

    async with session:
        unused_audio = session.stream_audio()
        unused_transcripts = session.stream_transcripts()
        assert [event async for event in session]
        assert [chunk async for chunk in unused_audio] == []
        assert [part async for part in unused_transcripts] == []


async def test_view_requires_entered_session() -> None:
    session = RealtimeSession(FakeRealtimeConnection([]))

    with pytest.raises(UserError, match='Enter the realtime session'):
        await anext(session.stream_audio())


async def test_assistant_transcript_partials_then_final() -> None:
    # Partial transcript deltas stream as PartDeltaEvents; the final (full-text) event adds nothing new.
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='Hi ', is_final=False),
            OutputTranscript(text='there', is_final=False),
            OutputTranscript(text='Hi there', is_final=True),  # provider repeats the full text
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = await collect_events(session)
    assert events == snapshot(
        [
            PartStartEvent(index=0, part=SpeechPart(speaker='assistant', transcript='')),
            PartDeltaEvent(
                index=0, delta=SpeechPartDelta(speaker='assistant', transcript_delta='Hi ', transcript='Hi ')
            ),
            PartDeltaEvent(
                index=0, delta=SpeechPartDelta(speaker='assistant', transcript_delta='there', transcript='Hi there')
            ),
            PartEndEvent(index=0, part=SpeechPart(speaker='assistant', transcript='Hi there')),
            ResponseCompleteEvent(),
            TurnCompleteEvent(),
        ]
    )
    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='Hi there')],
                model_name='m',
                timestamp=IsDatetime(),
                finish_reason='stop',
            )
        ]
    )


async def test_assistant_transcript_final_only() -> None:
    # A provider that only sends a single final transcript still yields a delta then the completed part.
    conn = FakeRealtimeConnection([OutputTranscript(text='Hello world', is_final=True), ResponseCompleteEvent()])
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = await collect_events(session)
    assert events == snapshot(
        [
            PartStartEvent(index=0, part=SpeechPart(speaker='assistant', transcript='')),
            PartDeltaEvent(
                index=0,
                delta=SpeechPartDelta(speaker='assistant', transcript_delta='Hello world', transcript='Hello world'),
            ),
            PartEndEvent(index=0, part=SpeechPart(speaker='assistant', transcript='Hello world')),
            ResponseCompleteEvent(),
            TurnCompleteEvent(),
        ]
    )


async def test_multiple_assistant_items_fold_into_one_response() -> None:
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='first', is_final=True, item_id='item-1'),
            OutputTranscript(text='second', is_final=True, item_id='item-2', output_text=True),
            ResponseCompleteEvent(provider_response_id='response-1', finish_reason='stop'),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, provider_name='openai')

    _ = await collect_events(session)

    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[
                    SpeechPart(speaker='assistant', transcript='first', id='item-1', provider_name='openai'),
                    TextPart(content='second', id='item-2', provider_name='openai'),
                ],
                provider_name='openai',
                provider_response_id='response-1',
                timestamp=IsDatetime(),
                finish_reason='stop',
            )
        ]
    )


@pytest.mark.parametrize(
    ('finish_reason', 'provider_details', 'expected_messages'),
    [
        (
            'length',
            {'status': 'incomplete', 'finish_reason': 'max_output_tokens'},
            # Each case carries its own `snapshot(...)` call site: `inline-snapshot` keys snapshots by
            # source location, so a single shared `snapshot()` in the test body would raise `UsageError`
            # when the two parametrized cases evaluate it to different values.
            snapshot(
                [
                    ModelResponse(
                        parts=[],
                        provider_details={'status': 'incomplete', 'finish_reason': 'max_output_tokens'},
                        provider_response_id='response-empty',
                        timestamp=IsDatetime(),
                        finish_reason='length',
                        conversation_id='conversation-1',
                    )
                ]
            ),
        ),
        (
            'error',
            {'status': 'failed'},
            snapshot(
                [
                    ModelResponse(
                        parts=[],
                        provider_details={'status': 'failed'},
                        provider_response_id='response-empty',
                        timestamp=IsDatetime(),
                        finish_reason='error',
                        conversation_id='conversation-1',
                    )
                ]
            ),
        ),
    ],
)
async def test_empty_terminal_response_is_recorded(
    finish_reason: Literal['length', 'error'],
    provider_details: dict[str, Any],
    expected_messages: list[ModelResponse],
) -> None:
    conn = FakeRealtimeConnection(
        [
            ResponseCompleteEvent(
                provider_response_id='response-empty',
                finish_reason=finish_reason,
                provider_details=provider_details,
            )
        ]
    )
    session = RealtimeSession(conn, _noop_runner, conversation_id='conversation-1')

    _ = await collect_events(session)

    assert session.new_messages() == expected_messages
    assert session.usage.requests == 1


async def test_empty_interrupted_response_is_recorded() -> None:
    conn = FakeRealtimeConnection(
        [
            ResponseCompleteEvent(
                interrupted=True,
                provider_response_id='response-cancelled',
                provider_details={'status': 'cancelled'},
            )
        ]
    )
    session = RealtimeSession(conn, _noop_runner)

    _ = await collect_events(session)

    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[],
                provider_details={'status': 'cancelled'},
                provider_response_id='response-cancelled',
                timestamp=IsDatetime(),
                state='interrupted',
            )
        ]
    )


async def test_bare_turn_boundary_does_not_create_empty_response() -> None:
    session = RealtimeSession(FakeRealtimeConnection([ResponseCompleteEvent()]), _noop_runner)

    _ = await collect_events(session)

    assert session.new_messages() == []


async def test_user_transcript_final_becomes_request() -> None:
    conn = FakeRealtimeConnection(
        [InputTranscript(text='what is ', is_final=False), InputTranscript(text='the weather', is_final=True)]
    )
    session = RealtimeSession(conn, _noop_runner)
    events = await collect_events(session)
    assert events == snapshot(
        [
            PartStartEvent(index=0, part=SpeechPart(speaker='user', transcript='')),
            PartDeltaEvent(
                index=0, delta=SpeechPartDelta(speaker='user', transcript_delta='what is ', transcript='what is ')
            ),
            PartDeltaEvent(
                index=0,
                delta=SpeechPartDelta(speaker='user', transcript_delta='the weather', transcript='what is the weather'),
            ),
            PartEndEvent(index=0, part=SpeechPart(speaker='user', transcript='what is the weather')),
        ]
    )
    assert session.new_messages() == snapshot(
        [ModelRequest(parts=[SpeechPart(speaker='user', transcript='what is the weather')])]
    )


async def test_interleaved_user_transcripts_use_item_ids() -> None:
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='first ', item_id='item-1'),
            InputTranscript(text='second ', item_id='item-2'),
            InputTranscript(text='second turn', is_final=True, item_id='item-2'),
            InputTranscript(text='first turn', is_final=True, item_id='item-1'),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, provider_name='openai')
    _ = await collect_events(session)

    parts = [message.parts[0] for message in session.new_messages() if isinstance(message, ModelRequest)]
    assert parts == [
        SpeechPart(speaker='user', transcript='first turn', id='item-1', provider_name='openai'),
        SpeechPart(speaker='user', transcript='second turn', id='item-2', provider_name='openai'),
    ]


async def test_session_close_flushes_user_transcripts_blocked_by_missing_final() -> None:
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='first partial', item_id='item-1'),
            InputTranscript(text='second final', is_final=True, item_id='item-2'),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, provider_name='openai')
    _ = await collect_events(session)

    assert session.new_messages() == [
        ModelRequest(
            parts=[SpeechPart(speaker='user', transcript='first partial', id='item-1', provider_name='openai')]
        ),
        ModelRequest(
            parts=[SpeechPart(speaker='user', transcript='second final', id='item-2', provider_name='openai')]
        ),
    ]


async def test_partial_only_user_transcript_finalized_on_turn_complete() -> None:
    # Gemini streams only partial input transcripts (never `is_final`) and no `InputSpeechEndEvent`, so
    # the user turn is finalized at the turn boundary. Without that, the transcribed user turn is
    # dropped from history entirely (only the assistant response would remain).
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='what is ', is_final=False),
            InputTranscript(text='the weather', is_final=False),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner)
    _ = await collect_events(session)
    assert session.new_messages() == snapshot(
        [ModelRequest(parts=[SpeechPart(speaker='user', transcript='what is the weather')])]
    )


async def test_partial_only_user_transcript_strips_leading_space() -> None:
    # Gemini streams partial-only transcripts whose first delta carries a leading space; with no final
    # snapshot to reconcile against (unlike OpenAI's `.completed`), the finalized turn would keep the
    # space. Finalization strips it so the result matches the OpenAI transcription of the same utterance.
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text=' Hello, my name', is_final=False),
            InputTranscript(text=' is Marcelo.', is_final=False),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner)
    _ = await collect_events(session)
    [request] = [message for message in session.new_messages() if isinstance(message, ModelRequest)]
    [part] = request.parts
    assert isinstance(part, SpeechPart)
    assert part.transcript == 'Hello, my name is Marcelo.'


async def test_user_transcript_final_snapshot_reconciles_whitespace_drift() -> None:
    # OpenAI's input-transcription deltas can carry a leading space that the `.completed` full-text
    # snapshot trims. The final snapshot must replace the accumulated deltas, not append a near-
    # duplicate (` Hello…` + `Hello…` = ` Hello…Hello…`).
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text=' Hello, my name', is_final=False),
            InputTranscript(text=' is Marcelo.', is_final=False),
            InputTranscript(text='Hello, my name is Marcelo.', is_final=True),
        ]
    )
    session = RealtimeSession(conn, _noop_runner)
    _ = await collect_events(session)
    assert session.new_messages() == snapshot(
        [ModelRequest(parts=[SpeechPart(speaker='user', transcript='Hello, my name is Marcelo.')])]
    )


async def test_audio_delta_streams_and_transcript_pairs() -> None:
    conn = FakeRealtimeConnection(
        [AudioDelta(data=b'\x00\x01'), OutputTranscript(text='hi', is_final=True), ResponseCompleteEvent()]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = await collect_events(session)
    assert [type(e).__name__ for e in events] == snapshot(
        [
            'PartStartEvent',
            'PartDeltaEvent',
            'PartDeltaEvent',
            'PartEndEvent',
            'ResponseCompleteEvent',
            'TurnCompleteEvent',
        ]
    )
    # transcript_only (default): the completed part keeps the transcript but not the audio bytes.
    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='hi')],
                model_name='m',
                timestamp=IsDatetime(),
                finish_reason='stop',
            )
        ]
    )


async def test_control_events_and_recoverable_error_pass_through() -> None:
    conn = FakeRealtimeConnection([ResponseCompleteEvent(interrupted=True), SessionErrorEvent(message='oops')])
    session = RealtimeSession(conn, _noop_runner)
    events = await collect_events(session)
    # A recoverable error is mid-stream: the session keeps running and surfaces the event to the
    # consumer (rather than swallowing it) so a quiet failure is observable.
    assert events == [
        ResponseCompleteEvent(interrupted=True),
        TurnCompleteEvent(),
        SessionErrorEvent(message='oops'),
    ]


async def test_input_transcription_failure_passes_through_and_session_continues() -> None:
    # Failures finalize placeholder turns whether or not they identify their turn (`item_id` may be absent).
    identified = InputTranscriptionErrorEvent(message='audio unintelligible', item_id='user-1', content_index=0)
    anonymous = InputTranscriptionErrorEvent(message='transcription unavailable')
    conn = FakeRealtimeConnection([identified, anonymous, ResponseCompleteEvent()])
    session = RealtimeSession(conn, _noop_runner)

    events = await collect_events(session)
    assert [type(event).__name__ for event in events] == [
        'PartStartEvent',
        'PartEndEvent',
        'InputTranscriptionErrorEvent',
        'PartStartEvent',
        'PartEndEvent',
        'InputTranscriptionErrorEvent',
        'ResponseCompleteEvent',
        'TurnCompleteEvent',
    ]
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user', id='user-1')]),
            ModelRequest(parts=[SpeechPart(speaker='user')]),
        ]
    )


async def test_input_transcription_failure_after_partial_does_not_block_later_turns() -> None:
    # Item A streams a partial transcript, item B finalizes out of order, then A's transcription fails.
    # A's placeholder must unblock the ordered prefix so both turns reach history in provider order.
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='partial A', is_final=False, item_id='A'),
            InputTranscript(text='hello from B', is_final=True, item_id='B'),
            InputTranscriptionErrorEvent(message='transcription failed', item_id='A'),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner)
    _ = await collect_events(session)
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user', id='A')]),
            ModelRequest(parts=[SpeechPart(speaker='user', transcript='hello from B', id='B')]),
        ]
    )


@pytest.mark.parametrize('item_id', [None, 'user-1'])
async def test_input_transcription_failure_retained_audio_fallback(item_id: str | None) -> None:
    conn = FakeRealtimeConnection([InputTranscriptionErrorEvent(message='failed', item_id=item_id)])
    session = RealtimeSession(conn, _noop_runner, audio_retention='input_audio')
    if item_id is None:
        await session.send_audio(b'\xaa')

    _ = await collect_events(session)

    expected_audio = _wav_content(b'\xaa') if item_id is None else None
    assert session.new_messages() == [
        ModelRequest(parts=[SpeechPart(speaker='user', audio=expected_audio, id=item_id)])
    ]


async def test_fatal_session_error_raises() -> None:
    conn = FakeRealtimeConnection([SessionErrorEvent(message='provider failed', recoverable=False)])
    session = RealtimeSession(conn, _noop_runner)
    with pytest.raises(RealtimeError, match='provider failed'):
        _ = await collect_events(session)


async def test_interrupted_turn_keeps_partial_transcript() -> None:
    # A barge-in cancels the turn; the completed part reflects the partial transcript seen so far.
    conn = FakeRealtimeConnection(
        [OutputTranscript(text='the answer is ', is_final=False), ResponseCompleteEvent(interrupted=True)]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = await collect_events(session)
    # A barge-in ends the exchange too: nothing more is coming, so the turn completes with it.
    assert events[-2:] == [ResponseCompleteEvent(interrupted=True), TurnCompleteEvent()]
    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='the answer is ')],
                model_name='m',
                timestamp=IsDatetime(),
                state='interrupted',
            )
        ]
    )


async def test_explicit_interrupt_records_audio_offset_on_last_speech_part() -> None:
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='first', is_final=True, item_id='item-1'),
            OutputTranscript(text='second', is_final=True, item_id='item-2'),
            ResponseCompleteEvent(interrupted=True),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')

    await session.interrupt(played_ms=640)
    _ = await collect_events(session)

    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[
                    SpeechPart(speaker='assistant', transcript='first', id='item-1'),
                    SpeechPart(speaker='assistant', transcript='second', interrupted_at_ms=640, id='item-2'),
                ],
                model_name='m',
                timestamp=IsDatetime(),
                state='interrupted',
            )
        ]
    )


async def test_interrupted_turn_without_trailing_speech_records_no_offset() -> None:
    # The offset is recorded on the last *speech* part, so a turn interrupted while the model was
    # calling a tool (its trailing part is a `ToolCallPart`) walks past it to the speech before it.
    # With no speech in the response at all, nothing is marked and `state='interrupted'` carries the
    # whole meaning.
    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='tc_1', tool_name='noop', args='', response_usage_follows=True),
            ResponseCompleteEvent(interrupted=True),
        ]
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'ok'

    session = RealtimeSession(conn, runner, model_name='m')

    await session.interrupt(played_ms=120)
    _ = await collect_events(session)

    response = next(m for m in session.new_messages() if isinstance(m, ModelResponse))
    assert response.state == 'interrupted'
    assert not any(isinstance(part, SpeechPart) for part in response.parts)


async def test_speech_part_provider_item_id_and_gemini_fallback() -> None:
    openai = RealtimeSession(
        FakeRealtimeConnection(
            [OutputTranscript(text='hello', is_final=True, item_id='item-a'), ResponseCompleteEvent()]
        ),
        _noop_runner,
        provider_name='openai',
    )
    gemini = RealtimeSession(
        FakeRealtimeConnection([OutputTranscript(text='hello', is_final=True), ResponseCompleteEvent()]),
        _noop_runner,
        provider_name='google',
    )
    _ = await collect_events(openai)
    _ = await collect_events(gemini)

    openai_part = openai.new_messages()[0].parts[0]
    gemini_part = gemini.new_messages()[0].parts[0]
    assert isinstance(openai_part, SpeechPart) and isinstance(gemini_part, SpeechPart)
    assert (openai_part.id, openai_part.provider_name) == ('item-a', 'openai')
    assert (gemini_part.id, gemini_part.provider_name) == (None, None)


# --- tool calls: history + events --------------------------------------------------------------


async def test_turn_completes_once_the_tool_round_is_over_not_before() -> None:
    """`TurnCompleteEvent` waits for the response *after* the tool, which is the whole point of it.

    The provider's own turn boundary arrives once per response, so a tool round produces several and
    which one is last differs by model — `gpt-realtime-2.1` and xAI Grok Voice speak *before* calling the
    tool, so stopping on their first boundary truncates the exchange. This connection answers the tool
    result the way a provider does, so the ordering is the real one rather than the fixture's.
    """
    tool_answered = asyncio.Event()

    class _AnswersTheTool(FakeRealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            self.sent.append(content)
            if isinstance(content, ToolResult):
                tool_answered.set()

        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            yield ToolCall(tool_call_id='tc_1', tool_name='get_weather', args='{"city": "Paris"}')
            yield ResponseCompleteEvent()  # the tool-call response; the exchange is not over
            await tool_answered.wait()
            yield OutputTranscript(text='It is sunny in Paris', is_final=True)
            yield ResponseCompleteEvent()  # the answer; now it is

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'Sunny, 22C'

    session = RealtimeSession(_AnswersTheTool([]), runner, model_name='m')
    events = await collect_events(session)

    assert [type(event).__name__ for event in events] == snapshot(
        [
            'PartStartEvent',
            'PartEndEvent',
            'FunctionToolCallEvent',
            'ResponseCompleteEvent',
            'FunctionToolResultEvent',
            'PartStartEvent',
            'PartDeltaEvent',
            'PartEndEvent',
            'ResponseCompleteEvent',
            'TurnCompleteEvent',
        ]
    )
    # Exactly one, and it lands after the model has actually answered.
    assert [i for i, e in enumerate(events) if isinstance(e, TurnCompleteEvent)] == [len(events) - 1]


async def test_tool_call_round_builds_classic_history() -> None:
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text="what's the weather in Paris", is_final=True),
            ToolCall(tool_call_id='tc_1', tool_name='get_weather', args='{"city": "Paris"}'),
            OutputTranscript(text="It's sunny in Paris", is_final=True),
            ResponseCompleteEvent(),
        ]
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        assert name == 'get_weather'
        assert args == {'city': 'Paris'}
        return 'Sunny, 22C'

    session = RealtimeSession(conn, runner, model_name='m')
    events = await collect_events(session)

    assert [type(e).__name__ for e in events] == snapshot(
        [
            'PartStartEvent',  # user transcript start
            'PartDeltaEvent',
            'PartEndEvent',  # user transcript end
            'PartStartEvent',  # tool call part start
            'PartEndEvent',  # tool call part end
            'FunctionToolCallEvent',
            'PartStartEvent',  # assistant answer start
            'PartDeltaEvent',
            'PartEndEvent',  # assistant answer end
            'ResponseCompleteEvent',
            'FunctionToolResultEvent',
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
                finish_reason='stop',
            ),
        ]
    )


async def test_late_input_transcript_still_precedes_the_response_it_prompted() -> None:
    """A user turn keeps its place in history however late the provider transcribes it.

    Input transcription is asynchronous, and the final event can land after the response the speech
    prompted has already been recorded — measured live on Azure and Gemini Live, and reachable on OpenAI
    and xAI under load, because a function-call-only response finalizes long before the transcript. Read
    back, an appended user turn says the model called a tool unprompted and only then heard the question.

    The wire order here is Azure's, measured live: speech start, the transcript's partials, then the whole
    tool round, and only afterwards the `.completed` snapshot.
    """
    conn = FakeRealtimeConnection(
        [
            InputSpeechStartEvent(),
            InputTranscript(text="what's the weather in Paris", item_id='item-1'),
            ToolCall(tool_call_id='tc_1', tool_name='get_weather', args='{"city": "Paris"}'),
            InputTranscript(text="what's the weather in Paris", is_final=True, item_id='item-1'),
            OutputTranscript(text="It's sunny in Paris", is_final=True),
            ResponseCompleteEvent(),
        ]
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'Sunny, 22C'

    session = RealtimeSession(conn, runner, model_name='m')
    await collect_events(session)

    assert [
        (type(message).__name__, [type(part).__name__ for part in message.parts]) for message in session.new_messages()
    ] == snapshot(
        [
            ('ModelRequest', ['SpeechPart']),
            ('ModelResponse', ['ToolCallPart']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelResponse', ['SpeechPart']),
        ]
    )


async def test_late_input_transcript_of_a_second_turn_follows_the_first_exchange() -> None:
    """A late transcript joins the conversation where its turn started, not at the front of history.

    The first exchange is already recorded when the second spoken turn opens, so that turn's place is
    after it — the same rule that keeps a first turn ahead of its own response has to know how far along
    the conversation was.
    """
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='what is the weather', is_final=True, item_id='item-1'),
            OutputTranscript(text='Sunny.', is_final=True),
            ResponseCompleteEvent(),
            InputSpeechStartEvent(),
            InputTranscript(text='and tomorrow', item_id='item-2'),
            ToolCall(tool_call_id='tc_1', tool_name='get_weather', args='{"city": "Paris"}'),
            InputTranscript(text='and tomorrow', is_final=True, item_id='item-2'),
            OutputTranscript(text='Rain.', is_final=True),
            ResponseCompleteEvent(),
        ]
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'Rain, 12C'

    session = RealtimeSession(conn, runner, model_name='m')
    await collect_events(session)

    assert [
        (type(message).__name__, [type(part).__name__ for part in message.parts]) for message in session.new_messages()
    ] == snapshot(
        [
            ('ModelRequest', ['SpeechPart']),
            ('ModelResponse', ['SpeechPart']),
            ('ModelRequest', ['SpeechPart']),
            ('ModelResponse', ['ToolCallPart']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelResponse', ['SpeechPart']),
        ]
    )


async def test_late_input_transcript_anchors_from_sent_audio_without_speech_boundaries() -> None:
    """The turn is placed even on a provider that reports no speech boundary at all.

    Gemini Live sends neither `InputSpeechStartEvent` nor a final input transcript, so the turn is only
    finalized at `ResponseCompleteEvent` — by which time its tool round is long recorded. Audio starting is
    then the only signal that a user turn began, so that is where its place in history comes from.
    """
    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='tc_1', tool_name='get_weather', args='{"city": "Paris"}'),
            InputTranscript(text="what's the weather in Paris"),
            OutputTranscript(text="It's sunny in Paris", is_final=True),
            ResponseCompleteEvent(),
        ]
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'Sunny, 22C'

    session = RealtimeSession(conn, runner, model_name='m')
    async with session:
        await session.send_audio(b'\x00\x01')
        await drain_events(session)

    assert [
        (type(message).__name__, [type(part).__name__ for part in message.parts]) for message in session.new_messages()
    ] == snapshot(
        [
            ('ModelRequest', ['SpeechPart']),
            ('ModelResponse', ['ToolCallPart']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelResponse', ['SpeechPart']),
        ]
    )


async def test_tool_response_finalized_on_usage_is_not_duplicated_at_terminal() -> None:
    conn = FakeRealtimeConnection(
        [
            ToolCall(
                tool_call_id='tc-1',
                tool_name='noop',
                args='{}',
                response_usage_follows=True,
            ),
            SessionUsageEvent(
                usage=RequestUsage(output_tokens=1),
                provider_response_id='response-tool',
                finish_reason='tool_call',
            ),
            ResponseCompleteEvent(
                provider_response_id='response-tool',
                finish_reason='stop',
                provider_details={'status': 'completed'},
            ),
        ]
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'done'

    session = RealtimeSession(conn, runner)
    _ = await collect_events(session)

    responses = [message for message in session.new_messages() if isinstance(message, ModelResponse)]
    assert len(responses) == 1
    assert isinstance(responses[0].parts[0], ToolCallPart)


async def test_tool_call_events_carry_real_parts() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='f', args='{"x": 1}')])

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return '42'

    session = RealtimeSession(conn, runner)
    events = await collect_events(session)
    call_event = next(e for e in events if isinstance(e, FunctionToolCallEvent))
    result_event = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert call_event.part == ToolCallPart(tool_name='f', args='{"x": 1}', tool_call_id='tc')
    assert call_event.args_valid is True
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
    _ = await collect_events(session)
    assert seen == {}


async def test_invalid_json_args_reported_without_calling_tool() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc_4', tool_name='noop', args='not json')])
    session = RealtimeSession(conn, _noop_runner)
    events = await collect_events(session)
    call = next(e for e in events if isinstance(e, FunctionToolCallEvent))
    assert call.args_valid is False
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, RetryPromptPart)
    assert 'could not parse tool arguments' in str(result.part.content)
    assert isinstance(conn.sent[0], ToolResult)
    assert conn.sent[0].output == result.part.model_response()


async def test_non_object_json_args_reported() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='noop', args='[1, 2]')])
    session = RealtimeSession(conn, _noop_runner)
    events = await collect_events(session)
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, RetryPromptPart)
    assert 'expected tool arguments to be a JSON object' in str(result.part.content)


async def test_tool_runner_exception_ends_session() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='boom', args='{}')])

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        raise RuntimeError('kaboom')

    session = RealtimeSession(conn, runner)
    with pytest.raises(RuntimeError, match='kaboom'):
        await collect_events(session)
    assert conn.sent == []


@pytest.mark.parametrize(
    ('exception', 'reason'),
    [(ApprovalRequired, 'requires approval'), (CallDeferred, 'runs externally')],
)
async def test_deferred_tool_becomes_deliberate_error_result(exception: type[Exception], reason: str) -> None:
    # Deferred-tool flows are graph-only: a live session can't pause for out-of-band approval or an
    # external result, so the model gets a deliberate explanation (not a leaked exception repr) and
    # the conversation keeps flowing.
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='boom', args='{}')])

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        raise exception

    session = RealtimeSession(conn, runner)
    events = await collect_events(session)
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert result.part.content == f"Error: The 'boom' tool {reason} and cannot be completed during a realtime session."


async def test_response_model_name_prefers_server_reported() -> None:
    # `ModelResponse.model_name` records the model the connection reports the server actually served
    # (it can differ from the requested id — xAI silently substitutes its default for unknown slugs),
    # mirroring how request-response models stamp the response's reported model, not the requested one.
    conn = FakeRealtimeConnection(
        [OutputTranscript(text='hi', is_final=True), ResponseCompleteEvent()], model_name='grok-voice-latest'
    )
    session = RealtimeSession(conn, _noop_runner, model_name='grok-voice-4-turbo')
    _ = await collect_events(session)
    response = next(m for m in session.all_messages() if isinstance(m, ModelResponse))
    assert response.model_name == 'grok-voice-latest'


async def test_agent_realtime_session_threads_provider_name() -> None:
    agent: Agent[object, str] = Agent()
    model = FakeRealtimeModel(
        FakeRealtimeConnection([OutputTranscript(text='hi', is_final=True), ResponseCompleteEvent()])
    )
    async with agent.realtime(model).session() as session:
        _ = [event async for event in session]
    response = next(message for message in session.new_messages() if isinstance(message, ModelResponse))
    assert response.provider_name == 'fake'


async def test_tool_does_not_block_other_events() -> None:
    release = asyncio.Event()
    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='bg_1', tool_name='slow', args='{}'),
            OutputTranscript(text='let me check', is_final=False),
            ResponseCompleteEvent(),
        ],
        release=release,
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        await release.wait()
        return 'done in background'

    session = RealtimeSession(conn, runner)
    events = await collect_events(session)

    # The tool call fires immediately, the model keeps talking, and the result lands only after the turn.
    assert [type(e).__name__ for e in events] == snapshot(
        [
            'PartStartEvent',  # tool call part
            'PartEndEvent',
            'FunctionToolCallEvent',
            'PartStartEvent',  # assistant transcript
            'PartDeltaEvent',
            'PartEndEvent',
            'ResponseCompleteEvent',
            'FunctionToolResultEvent',
        ]
    )
    result = events[-1]
    assert isinstance(result, FunctionToolResultEvent)
    assert result.part.content == 'done in background'
    assert conn.sent == [ToolResult(tool_call_id='bg_1', output='done in background')]


async def test_tool_result_adjacent_to_call_in_history() -> None:
    """A late result streams last, but sits right after its call in `all_messages()`.

    Request-response APIs demand call/return adjacency (OpenAI rejects a `tool` message that doesn't
    directly follow the assistant message carrying the call), so the portable history must keep it
    even when the model spoke again before the tool finished.
    """
    release = asyncio.Event()
    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='bg_1', tool_name='slow', args='{}'),
            OutputTranscript(text='still working on it', is_final=False),
            ResponseCompleteEvent(),
        ],
        release=release,
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        await release.wait()
        return 'late result'

    session = RealtimeSession(conn, runner)
    events = await collect_events(session)

    # The result event streams in completion order: after the intervening assistant turn.
    assert isinstance(events[-1], FunctionToolResultEvent)

    # But in history the return is adjacent to its call, with the intervening turn after it.
    call_response, tool_return, speech_response = session.all_messages()
    assert isinstance(call_response, ModelResponse)
    assert isinstance(call_response.parts[0], ToolCallPart)
    assert isinstance(tool_return, ModelRequest)
    assert isinstance(tool_return.parts[0], ToolReturnPart)
    assert tool_return.parts[0].tool_call_id == 'bg_1'
    assert tool_return.parts[0].content == 'late result'
    assert isinstance(speech_response, ModelResponse)
    assert isinstance(speech_response.parts[0], SpeechPart)
    assert speech_response.parts[0].transcript == 'still working on it'


async def test_parallel_tool_returns_stay_grouped_after_calling_response() -> None:
    release = asyncio.Event()
    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='one', tool_name='fast', args='{}'),
            ToolCall(tool_call_id='two', tool_name='slow', args='{}'),
            ResponseCompleteEvent(),
        ],
        release=release,
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        await release.wait()
        return call_id

    session = RealtimeSession(conn, runner)
    _ = await collect_events(session)

    assert [
        message.parts[0].tool_call_id
        for message in session.new_messages()[1:]
        if isinstance(message, ModelRequest) and isinstance(message.parts[0], ToolReturnPart)
    ] == ['one', 'two']


def test_insert_tool_return_skips_existing_parallel_returns() -> None:
    session = RealtimeSession(FakeRealtimeConnection([]))
    call_one = ToolCallPart(tool_name='one', args={}, tool_call_id='one')
    call_two = ToolCallPart(tool_name='two', args={}, tool_call_id='two')
    first_return = ModelRequest(parts=[ToolReturnPart(tool_name='one', content='1', tool_call_id='one')])
    second_return = ModelRequest(parts=[ToolReturnPart(tool_name='two', content='2', tool_call_id='two')])
    response = ModelResponse(parts=[call_one, call_two])
    session._history.extend([response, first_return])  # pyright: ignore[reportPrivateUsage]

    session._insert_tool_return(call_two, second_return)  # pyright: ignore[reportPrivateUsage]

    assert session.new_messages() == [response, first_return, second_return]


class AwaitBetweenConnection(RealtimeConnection):
    """A connection that yields control between events so tool tasks can progress."""

    def __init__(self, events: list[RealtimeCodecEvent]) -> None:
        self._events = events
        self.sent: list[RealtimeInput] = []

    async def send(self, content: RealtimeInput) -> None:
        self.sent.append(content)

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        for event in self._events:
            yield event
            await asyncio.sleep(0)


async def test_tool_completion_drained_between_events() -> None:
    conn = AwaitBetweenConnection(
        [ToolCall(tool_call_id='bg', tool_name='fast', args='{}'), AudioDelta(data=b'\x01'), AudioDelta(data=b'\x02')]
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'quick'

    session = RealtimeSession(conn, runner)
    events = await collect_events(session)

    # The point is that the finished tool's result is drained while the upstream is still producing,
    # rather than waiting for the stream to end. Its exact position among the audio deltas depends on
    # how many event-loop checkpoints the send path takes, so read this as "early", not as a contract.
    assert [type(e).__name__ for e in events] == snapshot(
        [
            'PartStartEvent',
            'PartEndEvent',
            'FunctionToolCallEvent',
            'PartStartEvent',
            'PartDeltaEvent',
            'FunctionToolResultEvent',
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
        self.iteration_task: asyncio.Task[Any] | None = None

    async def send(self, content: RealtimeInput) -> None:
        self.sent.append(content)

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        self.iteration_task = asyncio.current_task()
        yield self._call
        await asyncio.Event().wait()


async def test_tool_completion_delivered_while_upstream_idle() -> None:
    # The connection goes silent after the tool call; the completion must still surface promptly
    # rather than waiting for a provider event that never arrives.
    conn = IdleAfterToolConnection(ToolCall(tool_call_id='bg', tool_name='fast', args='{}'))

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'ready'

    session = RealtimeSession(conn, runner)
    async with session:
        events = session.__aiter__()
        # Drain the tool-call part events + FunctionToolCallEvent.
        assert isinstance(await anext(events), PartStartEvent)
        assert isinstance(await anext(events), PartEndEvent)
        assert isinstance(await anext(events), FunctionToolCallEvent)
        # Without multiplexing this would hang forever waiting on the idle connection.
        completed = await asyncio.wait_for(anext(events), timeout=1.0)
        assert isinstance(completed, FunctionToolResultEvent)
        assert completed.part.content == 'ready'


class ExplodingConnection(RealtimeConnection):
    """A connection whose iteration raises after yielding one event."""

    def __init__(self) -> None:
        self.sent: list[RealtimeInput] = []

    async def send(self, content: RealtimeInput) -> None:  # pragma: no cover
        self.sent.append(content)

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        yield AudioDelta(data=b'\x00')
        raise RuntimeError('connection dropped')


async def test_upstream_error_propagates_to_consumer() -> None:
    session = RealtimeSession(ExplodingConnection(), _noop_runner)
    with pytest.raises(RuntimeError, match='connection dropped'):
        _ = await collect_events(session)


async def test_upstream_error_surfaces_at_close_when_the_stream_was_never_iterated() -> None:
    """A session used without its event stream still reports what ended it.

    The pump's error is normally raised out of the event iterator, so a caller driving the session with
    `send()` and the audio/transcript views alone — which the quickstart shows — would otherwise exit
    *cleanly* from a provider hangup, or from an exceeded `usage_limits` it silently spent past.
    """
    session = RealtimeSession(ExplodingConnection(), _noop_runner)
    with pytest.raises(RuntimeError, match='connection dropped'):
        async with session:
            assert [chunk async for chunk in session.stream_audio()] == [b'\x00']

    # Not re-raised over an exception already leaving the body, which would hide the caller's own error.
    other = RealtimeSession(ExplodingConnection(), _noop_runner)
    with pytest.raises(ValueError, match='mine'):
        async with other:
            assert [chunk async for chunk in other.stream_audio()] == [b'\x00']
            raise ValueError('mine')


async def test_upstream_error_does_not_wait_for_running_tool() -> None:
    class _ExplodingAfterTool(RealtimeConnection):
        # Tool is cancelled first.
        async def send(self, content: RealtimeInput) -> None:  # pragma: no cover
            raise AssertionError

        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            yield ToolCall(tool_call_id='bg', tool_name='hang', args='{}')
            await asyncio.sleep(0)  # let the tool start before the pump fails
            raise RuntimeError('connection dropped')

    blocked = asyncio.Event()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    tool_task: asyncio.Task[Any] | None = None

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        nonlocal tool_task
        tool_task = asyncio.current_task()
        started.set()
        try:
            await blocked.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return 'never'  # pragma: no cover

    session = RealtimeSession(_ExplodingAfterTool(), runner)
    with pytest.raises(RuntimeError, match='connection dropped'):
        await asyncio.wait_for(collect_events(session), timeout=1)

    assert started.is_set()
    assert tool_task is not None and tool_task.done() and tool_task.cancelled()
    assert cancelled.is_set()


class SendFailsConnection(RealtimeConnection):
    """Replays events but raises on every send — a connection dropping mid tool call (the only thing
    sent through it in these tests is the tool's `ToolResult`).

    With `idle=True` it never closes after the events (an idle provider); with `release` set it
    closes immediately and the tool only runs once released, so the failure lands after upstream end.
    """

    def __init__(
        self, events: list[RealtimeCodecEvent], *, idle: bool = False, release: asyncio.Event | None = None
    ) -> None:
        self._events = events
        self._idle = idle
        self._release = release

    async def send(self, content: RealtimeInput) -> None:
        raise RuntimeError('connection lost')

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        for event in self._events:
            yield event
        if self._release is not None:
            self._release.set()
        if self._idle:
            await asyncio.Event().wait()


async def test_tool_failure_propagates_while_idle() -> None:
    conn = SendFailsConnection([ToolCall(tool_call_id='bg', tool_name='boom', args='{}')], idle=True)

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'ok'

    # The tool's ToolResult send fails while the provider is idle; without propagation the
    # consumer would hang waiting for a completion that never arrives.
    session = RealtimeSession(conn, runner)
    with pytest.raises(RuntimeError, match='connection lost'):
        _ = await collect_events(session)


async def test_tool_failure_propagates_after_close() -> None:
    release = asyncio.Event()
    conn = SendFailsConnection([ToolCall(tool_call_id='bg', tool_name='boom', args='{}')], release=release)

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        await release.wait()  # only runs after upstream has closed → failure surfaces during drain
        return 'ok'

    session = RealtimeSession(conn, runner)
    with pytest.raises(RuntimeError, match='connection lost'):
        _ = await collect_events(session)


async def test_early_break_with_running_tool_cancels_task() -> None:
    blocked = asyncio.Event()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    tool_task: asyncio.Task[Any] | None = None
    conn = IdleAfterToolConnection(ToolCall(tool_call_id='bg', tool_name='hang', args='{}'))
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    async def hang() -> str:
        nonlocal tool_task
        tool_task = asyncio.current_task()
        started.set()
        try:
            await blocked.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return 'never'  # pragma: no cover

    async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
        async for event in session:
            if isinstance(event, FunctionToolCallEvent):
                await started.wait()
                break

    assert tool_task is not None and tool_task.done() and tool_task.cancelled()
    assert conn.iteration_task is not None and conn.iteration_task.done()
    assert cancelled.is_set()


async def test_tool_call_cancellation_cancels_running_tool() -> None:
    # The model cancels an in-flight tool call (e.g. the user barged in mid-call). The running task is
    # cancelled, no `ToolResult` is sent back to the model, and a cancelled result is recorded so the
    # call still has a matching return in history.
    started = asyncio.Event()
    cancelled = asyncio.Event()
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    async def slow() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        # Always cancelled first.
        return 'never'  # pragma: no cover

    class _CancelAfterStart(FakeRealtimeConnection):
        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            yield ToolCall(tool_call_id='c1', tool_name='slow', args='{}')
            await started.wait()  # let the tool task start before the model cancels it
            yield ToolCallCancelled(tool_call_ids=['c1'])

    events: list[Any] = []
    conn = _CancelAfterStart([])
    async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
        async for event in session:
            events.append(event)

    assert cancelled.is_set()  # the running tool observed cancellation
    assert conn.sent == []
    results = [e for e in events if isinstance(e, FunctionToolResultEvent)]
    assert len(results) == 1 and isinstance(results[0].part, ToolReturnPart)
    # The cancelled call still has exactly one matching return in history (valid for a handoff).
    returns = [
        part
        for message in session.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert [(part.tool_call_id, part.content) for part in returns] == [
        ('c1', 'Tool call cancelled before it completed.')
    ]


async def test_tool_call_cancellation_unknown_id_is_ignored() -> None:
    # A cancellation for an id with no matching in-flight call (already finished, or never started) must
    # be a no-op: no crash, no spurious result event, nothing sent. Covers the race where a tool finishes
    # in the window before its cancellation arrives (the `finally`-pop makes that atomic).
    conn = FakeRealtimeConnection(
        [
            ToolCallCancelled(tool_call_ids=['never-started']),
            OutputTranscript(text='hi', is_final=True),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner)
    events = await collect_events(session)
    assert [event for event in events if isinstance(event, FunctionToolResultEvent)] == []
    assert conn.sent == []


async def test_interrupt_does_not_cancel_in_flight_tool() -> None:
    # A user barge-in via `interrupt()` cancels the *model's* response server-side (`CancelResponse`),
    # but deliberately does NOT cancel a local tool that's already running: the work was dispatched, so it
    # runs to completion and its `ToolResult` is still sent back to the model. This is the intended design
    # (matching the OpenAI Agents SDK) and contrasts with a provider-driven `ToolCallCancelled` (above),
    # which *does* cancel the local task. On OpenAI/xAI, sending the result then auto-triggers a fresh
    # response server-side; suppressing that is the model's concern, not ours to second-guess here.
    started = asyncio.Event()
    release = asyncio.Event()
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    async def slow() -> str:
        started.set()
        await release.wait()
        return 'done'

    class _IdleAfterCall(FakeRealtimeConnection):
        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            yield ToolCall(tool_call_id='c1', tool_name='slow', args='{}')
            await asyncio.Event().wait()  # stay open; the consumer breaks out on the tool result

    conn = _IdleAfterCall([])
    async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
        async for event in session:
            if isinstance(event, FunctionToolCallEvent):
                await started.wait()
                await session.interrupt()
                release.set()
            elif isinstance(event, FunctionToolResultEvent):
                break

    # The barge-in reached the model, and the tool still completed and reported its result afterwards.
    assert CancelResponse() in conn.sent
    assert ToolResult(tool_call_id='c1', output='done') in conn.sent


# --- send helpers + history ---------------------------------------------------------------------


async def test_send_helpers_forward_to_connection() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    await session.send_audio(b'\x01\x02')
    await session.send('hello')
    await session.send(BinaryImage(data=b'\xff\xd8', media_type='image/jpeg'))
    await session.send(AudioInput(data=b'\x03'))
    assert conn.sent == [
        AudioInput(data=b'\x01\x02'),
        TextInput(text='hello'),
        ImageInput(data=b'\xff\xd8', media_type='image/jpeg'),
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


async def test_send_accepts_plain_content() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)

    await session.send('hello')
    await session.send(BinaryImage(data=b'image', media_type='image/png'))
    # A WAV container (e.g. retained `SpeechPart.audio`) is unwrapped to the raw PCM the wire expects.
    await session.send(_wav_content(b'\x01\x02\x03\x04'))
    # Raw PCM `BinaryContent` passes through verbatim.
    await session.send(BinaryContent(data=b'\xaa\xbb', media_type='audio/pcm'))

    assert conn.sent == snapshot(
        [
            TextInput(text='hello'),
            ImageInput(data=b'image', media_type='image/png'),
            AudioInput(data=b'\x01\x02\x03\x04'),
            AudioInput(data=b'\xaa\xbb'),
        ]
    )
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsDatetime())]),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[BinaryImage(data=b'image', media_type='image/png')],
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )


async def test_send_accepts_sequence() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)

    await session.send(['look at this', BinaryImage(data=b'image', media_type='image/png')])

    assert conn.sent == [TextInput(text='look at this'), ImageInput(data=b'image', media_type='image/png')]


async def test_image_history_retention_samples_and_round_trips() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, retain_images_every_n=2)
    images = [BinaryImage(data=f'image-{index}'.encode(), media_type='image/png') for index in range(4)]

    for image in images:
        await session.send(image)

    assert conn.sent == [ImageInput(data=image.data, media_type='image/png') for image in images]
    assert session.all_messages() == [
        ModelRequest(parts=[UserPromptPart(content=[images[0]], timestamp=IsDatetime())]),
        ModelRequest(parts=[UserPromptPart(content=[images[2]], timestamp=IsDatetime())]),
    ]
    serialized = ModelMessagesTypeAdapter.dump_json(session.all_messages())
    assert ModelMessagesTypeAdapter.validate_json(serialized) == session.all_messages()


async def test_image_history_retention_must_be_positive() -> None:
    with pytest.raises(UserError, match='`retain_images_every_n` must be at least 1'):
        RealtimeSession(FakeRealtimeConnection([]), _noop_runner, retain_images_every_n=0)


async def test_send_rejects_unsupported_binary_content() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)

    with pytest.raises(UserError, match=r"Unsupported binary media type 'application/pdf'.*WAV audio, or raw PCM"):
        await session.send(BinaryContent(data=b'document', media_type='application/pdf'))

    # A non-WAV audio container can't be unwrapped, so it's rejected rather than streamed as noise.
    with pytest.raises(UserError, match=r"Unsupported binary media type 'audio/mpeg'.*WAV audio, or raw PCM"):
        await session.send(BinaryContent(data=b'\x00mp3', media_type='audio/mpeg'))

    assert conn.sent == []


async def test_send_rejects_raw_bytes_with_audio_hint() -> None:
    # `bytes` is a `Sequence[int]`; sending it must give a clear "use `send_audio`" error, not iterate into
    # a confusing per-byte failure.
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    with pytest.raises(UserError, match=r'Raw audio bytes cannot be sent.*send_audio'):
        await session.send(b'\x00\x01')  # type: ignore[arg-type]
    assert conn.sent == []


async def test_send_enforces_model_profile_guard() -> None:
    # `send(ImageInput(...))` must enforce the same `supports_image_input` guard as `send_image`.
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, profile=_profile(supports_image_input=False))
    with pytest.raises(UserError, match='does not support image input'):
        await session.send(ImageInput(data=b'\xff', media_type='image/jpeg'))
    assert conn.sent == []


async def test_send_rejects_control_verbs() -> None:
    # Turn-control verbs are connection-level vocabulary excluded from `RealtimeSessionInput`; `send()`
    # rejects them at runtime and directs callers to the dedicated methods (which apply profile guards).
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    for verb in (CommitAudio(), ClearAudio(), CreateResponse(), CancelResponse(), TruncateOutput(audio_end_ms=120)):
        with pytest.raises(UserError, match=r'Turn-control verbs cannot be sent.*commit_audio'):
            await session.send(verb)  # type: ignore[arg-type]
    assert conn.sent == []


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
    session = RealtimeSession(conn, _noop_runner, conversation_id='c1')
    await session.send('turn it up')
    assert session.new_messages() == snapshot(
        [ModelRequest(parts=[UserPromptPart(content='turn it up', timestamp=IsDatetime())], conversation_id='c1')]
    )


async def test_send_during_response_is_recorded_after_response() -> None:
    response_started = asyncio.Event()
    continue_response = asyncio.Event()

    class MidResponseConnection(RealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            assert content == TextInput(text='next turn')

        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            yield OutputTranscript(text='first ', item_id='assistant-1')
            response_started.set()
            await continue_response.wait()
            yield OutputTranscript(text='response', is_final=True, item_id='assistant-1')
            yield ResponseCompleteEvent(provider_response_id='response-1', finish_reason='stop')

    session = RealtimeSession(MidResponseConnection(), _noop_runner)
    async with session:
        stream = session.__aiter__()

        async def next_event() -> RealtimeEvent:
            return await anext(stream)

        events_task = asyncio.create_task(next_event())
        await response_started.wait()
        await session.send('next turn')
        continue_response.set()
        first_event = await events_task
        remaining_events = [event async for event in stream]

    assert isinstance(first_event, PartStartEvent)
    assert isinstance(remaining_events[-2], ResponseCompleteEvent)
    assert isinstance(remaining_events[-1], TurnCompleteEvent)
    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='first response', id='assistant-1')],
                provider_response_id='response-1',
                timestamp=IsDatetime(),
                finish_reason='stop',
            ),
            ModelRequest(parts=[UserPromptPart(content='next turn', timestamp=IsDatetime())]),
        ]
    )


async def test_manual_turn_control_helpers_forward_to_connection() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    await session.commit_audio()
    await session.create_response()
    await session.clear_audio()
    assert conn.sent == [CommitAudio(), CreateResponse(), ClearAudio()]


async def test_text_request_reserved_before_response_finishes_during_send() -> None:
    class _FinishingSend(FakeRealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            self.sent.append(content)
            session._translate_event(OutputTranscript(text='done', is_final=True))  # pyright: ignore[reportPrivateUsage]
            session._translate_event(ResponseCompleteEvent())  # pyright: ignore[reportPrivateUsage]

    conn = _FinishingSend([])
    session = RealtimeSession(conn)
    await session.send('question')
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='question', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='done')],
                timestamp=IsDatetime(),
                finish_reason='stop',
            ),
        ]
    )


async def test_send_audio_reserved_before_speech_boundary_during_send() -> None:
    class _BoundarySend(FakeRealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            self.sent.append(content)
            session._translate_event(InputSpeechEndEvent())  # pyright: ignore[reportPrivateUsage]

    conn = _BoundarySend([], input_transcription_enabled=False)
    session = RealtimeSession(conn, audio_retention='input_audio')
    await session.send_audio(b'\xaa\xbb')
    assert session.new_messages() == snapshot(
        [ModelRequest(parts=[SpeechPart(speaker='user', audio=_wav_content(b'\xaa\xbb'))])]
    )


async def test_failed_sends_leave_no_phantom_history_or_audio() -> None:
    class _FailingSend(FakeRealtimeConnection):
        fail = True

        async def send(self, content: RealtimeInput) -> None:
            if self.fail:
                raise RuntimeError('send failed')
            self.sent.append(content)

    conn = _FailingSend([])
    session = RealtimeSession(conn, audio_retention='input_audio')
    with pytest.raises(RuntimeError, match='send failed'):
        await session.send('question')
    with pytest.raises(RuntimeError, match='send failed'):
        await session.send(TextInput(text='typed question'))
    with pytest.raises(RuntimeError, match='send failed'):
        await session.send_audio(b'\xaa\xbb')
    conn.fail = False
    await session.send_audio(b'\xcc')
    session._translate_event(InputTranscript(text='successful', is_final=True))  # pyright: ignore[reportPrivateUsage]
    assert session.new_messages() == snapshot(
        [ModelRequest(parts=[SpeechPart(speaker='user', transcript='successful', audio=_wav_content(b'\xcc'))])]
    )

    # Without input-audio retention there is no buffered audio to roll back; the failure still
    # propagates and leaves no history.
    unretained = RealtimeSession(_FailingSend([]))
    with pytest.raises(RuntimeError, match='send failed'):
        await unretained.send_audio(b'\xaa')
    assert unretained.new_messages() == []


async def test_transport_failure_while_sending_becomes_a_realtime_error() -> None:
    # A send that fails because the link is gone is the same failure the receive side reports, so it
    # surfaces as `RealtimeError` (a `ModelAPIError`) instead of leaking the transport's own exception.
    # Only the types a connection declares are mapped: anything else is a bug, not a lost connection.
    class _DisconnectedConnection(FakeRealtimeConnection):
        transport_errors = (ConnectionResetError,)

        async def send(self, content: RealtimeInput) -> None:
            raise ConnectionResetError('connection reset by peer')

    session = RealtimeSession(_DisconnectedConnection([]), model_name='gpt-realtime')
    with pytest.raises(RealtimeError) as exc_info:
        await session.send('anyone there?')
    assert str(exc_info.value) == snapshot('Realtime connection failed while sending: connection reset by peer')
    assert exc_info.value.model_name == 'gpt-realtime'
    assert isinstance(exc_info.value.__cause__, ConnectionResetError)
    assert isinstance(exc_info.value, ModelAPIError)

    # `interrupt()` sends a truncate and a cancel as one group, so the link can also drop *between*
    # them; the second send is mapped like the first, and the interruption is not recorded as having
    # happened. A connection that failed on the first send would never reach the cancel at all.
    class _DisconnectedAfterTruncate(FakeRealtimeConnection):
        transport_errors = (ConnectionResetError,)

        async def send(self, content: RealtimeInput) -> None:
            if isinstance(content, CancelResponse):
                raise ConnectionResetError('connection reset by peer')
            self.sent.append(content)

    interrupted = _DisconnectedAfterTruncate([])
    session = RealtimeSession(interrupted, model_name='gpt-realtime')
    with pytest.raises(RealtimeError, match='failed while sending'):
        await session.interrupt(played_ms=120)
    assert interrupted.sent == [TruncateOutput(audio_end_ms=120)]
    assert session._pending_interrupted_at_ms is None  # pyright: ignore[reportPrivateUsage]

    # A session built straight from a connection may not know any model id to attribute this to.
    anonymous = RealtimeSession(_DisconnectedConnection([]))
    with pytest.raises(RealtimeError) as exc_info:
        await anonymous.send('anyone there?')
    assert exc_info.value.model_name == 'unknown'


async def test_undeclared_send_failure_is_left_alone() -> None:
    # A connection that fails for a reason it didn't declare as a transport error is reporting a bug,
    # not a lost connection; dressing it up as a `RealtimeError` would hide that.
    class _BuggyConnection(FakeRealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            raise ValueError('bug in the connection')

    session = RealtimeSession(_BuggyConnection([]))
    with pytest.raises(ValueError, match='bug in the connection'):
        await session.send('hello')


def test_remove_sent_request_ignores_unknown_request() -> None:
    # Unit test: the rollback helper is only ever called with the request the failing send just
    # reserved, so the not-found fall-through can't be reached through the public API — pin directly
    # that it stays tolerant of a request that is in neither pending nor history.
    session = RealtimeSession(FakeRealtimeConnection([]))
    session._remove_sent_request(ModelRequest(parts=[]))  # pyright: ignore[reportPrivateUsage]
    assert session.new_messages() == []


async def test_session_accumulates_usage_and_requests() -> None:
    conn = FakeRealtimeConnection(
        [
            SessionUsageEvent(usage=RequestUsage(input_tokens=10, output_tokens=5)),
            SessionUsageEvent(usage=RequestUsage(input_tokens=3, output_tokens=2)),
            OutputTranscript(text='ok', is_final=True),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    _ = await collect_events(session)
    assert session.usage.input_tokens == 13
    assert session.usage.output_tokens == 7
    assert session.usage.requests == 1
    # The turn's combined usage lands on the finalized assistant response.
    response = session.new_messages()[0]
    assert isinstance(response, ModelResponse)
    assert response.usage == snapshot(RequestUsage(input_tokens=13, output_tokens=7))


async def test_session_counts_tool_calls() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t', tool_name='f', args='{}'), ResponseCompleteEvent()])

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'ok'

    session = RealtimeSession(conn, runner)
    _ = await collect_events(session)
    assert session.usage.tool_calls == 1


async def test_truncate_output_helper_forwards_to_connection() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    await session.interrupt(played_ms=640)
    assert conn.sent == [TruncateOutput(audio_end_ms=640), CancelResponse()]


async def test_interrupt_truncates_before_cancel() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    await session.interrupt(played_ms=800)
    # Truncate must precede cancel: cancel triggers response.done, which clears the tracked item.
    assert conn.sent == [TruncateOutput(audio_end_ms=800), CancelResponse()]


async def test_interrupt_without_played_ms_only_cancels() -> None:
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
    # `interrupt(played_ms=...)`, while a plain `interrupt()` still cancels.
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, profile=_profile(supports_output_truncation=False))
    with pytest.raises(UserError, match='does not support output truncation'):
        await session.interrupt(played_ms=100)
    assert conn.sent == []
    await session.interrupt()
    assert conn.sent == [CancelResponse()]


class SlowSendConnection(FakeRealtimeConnection):
    """A connection that yields control inside `send`, so concurrent senders can interleave."""

    async def send(self, content: RealtimeInput) -> None:
        await asyncio.sleep(0)
        await super().send(content)


async def test_interrupt_truncate_and_cancel_cannot_be_split() -> None:
    # `interrupt(played_ms=...)` is two frames, and the cancel is only correct for the response the
    # truncate targeted. On OpenAI-shaped providers a concurrent send starts a new response, so a frame
    # landing in the gap would leave the cancel killing that one instead of the barge-in's target. The
    # session's send lock has to keep the pair adjacent even when sends overlap.
    conn = SlowSendConnection([])
    session = RealtimeSession(conn, _noop_runner, model_name='m')

    await asyncio.gather(
        session.interrupt(played_ms=100),
        *(session.send('concurrent') for _ in range(3)),
    )

    kinds = [type(frame).__name__ for frame in conn.sent]
    truncate = kinds.index('TruncateOutput')
    assert kinds[truncate + 1] == 'CancelResponse', f'a frame was interleaved into the transaction: {kinds}'
    # The concurrent sends still all got through, just never in the middle.
    assert kinds.count('TextInput') == 3


async def test_image_input_guard() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, profile=_profile(supports_image_input=False))
    with pytest.raises(UserError, match='does not support image input'):
        await session.send(BinaryImage(data=b'\xff\xd8', media_type='image/jpeg'))
    assert conn.sent == []


async def test_early_break_cancels_pump() -> None:
    # Breaking out early must cancel the background pump task so it doesn't leak, parked forever
    # awaiting an upstream event that never comes. A finite connection wouldn't test this — its pump
    # ends on its own; here the pump blocks mid-iteration and only stops if it is cancelled.
    parked = asyncio.Event()
    cancelled = asyncio.Event()

    class _BlockAfterFirst(RealtimeConnection):
        def __init__(self) -> None:
            self.iteration_task: asyncio.Task[Any] | None = None

        # Never sent to.
        async def send(self, content: RealtimeInput) -> None:  # pragma: no cover
            raise AssertionError

        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            self.iteration_task = asyncio.current_task()
            yield AudioDelta(data=b'\x00')
            parked.set()
            try:
                await asyncio.Event().wait()  # park until the pump task is cancelled
            except asyncio.CancelledError:
                cancelled.set()
                raise
            # Unreachable while parked.
            yield AudioDelta(data=b'\x01')  # pragma: no cover

    conn = _BlockAfterFirst()
    agent: Agent[None, str] = Agent()
    async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
        async for event in session:
            assert isinstance(event, PartStartEvent)
            await parked.wait()  # the pump has consumed the first event and is parked on the next
            break

    # The owner's exit drains cancellation synchronously; no async-generator close or GC pumping is
    # needed before the connection observes cancellation and the receive task is done.
    assert cancelled.is_set()
    assert conn.iteration_task is not None and conn.iteration_task.done()


def test_asap_notification_is_ignored_after_loop_closes() -> None:
    session = RealtimeSession(FakeRealtimeConnection([]), _noop_runner)
    closed_loop = asyncio.new_event_loop()
    closed_loop.close()
    session._loop = closed_loop  # pyright: ignore[reportPrivateUsage]
    session._notify_asap_pending_messages()  # pyright: ignore[reportPrivateUsage]


async def test_concurrent_iteration_raises() -> None:
    class _IdleConnection(RealtimeConnection):
        # Never sent to.
        async def send(self, content: RealtimeInput) -> None:  # pragma: no cover
            raise AssertionError

        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            yield AudioDelta(data=b'\x00')
            await asyncio.Event().wait()

    agent: Agent[None, str] = Agent()
    async with agent.realtime(FakeRealtimeModel(_IdleConnection())).session() as session:
        first = session.__aiter__()
        assert isinstance(await anext(first), PartStartEvent)

        second = session.__aiter__()
        with pytest.raises(UserError, match='already being iterated'):
            await anext(second)

    late = session.__aiter__()
    with pytest.raises(UserError, match='closed'):
        await anext(late)


async def test_direct_session_must_be_entered_and_streams_once() -> None:
    session = RealtimeSession(FakeRealtimeConnection([]), _noop_runner)

    unentered = session.__aiter__()
    with pytest.raises(UserError, match='async with'):
        await anext(unentered)

    async with session:
        assert [event async for event in session] == []
        exhausted = session.__aiter__()
        with pytest.raises(UserError, match='already ended'):
            await anext(exhausted)

    with pytest.raises(UserError, match='cannot be entered more than once'):
        await session.__aenter__()


# --- audio retention ----------------------------------------------------------------------------


async def test_audio_retention_output_keeps_assistant_audio() -> None:
    conn = FakeRealtimeConnection(
        [
            AudioDelta(data=b'\x00\x01'),
            AudioDelta(data=b'\x02\x03'),
            OutputTranscript(text='hi', is_final=True),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m', audio_retention='all')
    _ = await collect_events(session)
    response = session.new_messages()[0]
    assert isinstance(response, ModelResponse)
    part = response.parts[0]
    assert isinstance(part, SpeechPart)
    assert part.transcript == 'hi'
    assert part.audio is not None
    assert part.audio.media_type == 'audio/wav'
    assert part.audio.format == 'wav'
    with wave.open(io.BytesIO(part.audio.data), 'rb') as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000)
        assert wav.readframes(wav.getnframes()) == b'\x00\x01\x02\x03'


async def test_audio_retention_input_keeps_user_audio() -> None:
    conn = FakeRealtimeConnection([InputTranscript(text='hello', is_final=True)])
    session = RealtimeSession(conn, _noop_runner, audio_retention='input_audio')
    # send_audio before the transcript finalizes buffers into the user part.
    await session.send_audio(b'\xaa\xbb')
    await session.send_audio(b'\xcc')
    _ = await collect_events(session)
    assert session.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SpeechPart(
                        speaker='user',
                        transcript='hello',
                        audio=_wav_content(b'\xaa\xbb\xcc'),
                    )
                ]
            )
        ]
    )


async def test_audio_retention_segmentation_follows_provider_boundaries() -> None:
    """Speech-end providers cut input early; boundary-less providers retain through response completion."""
    speech_ended = asyncio.Event()
    finish_openai = asyncio.Event()

    class _SpeechEndConnection(FakeRealtimeConnection):
        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            await speech_ended.wait()
            yield InputSpeechEndEvent(item_id='turn')
            await finish_openai.wait()
            yield InputTranscript(text='hello', is_final=True, item_id='turn')
            yield ResponseCompleteEvent()

    openai_session = RealtimeSession(_SpeechEndConnection([]), _noop_runner, audio_retention='input_audio')
    await openai_session.send_audio(b'speech')
    speech_ended.set()
    async with openai_session:
        events = openai_session.__aiter__()
        assert isinstance(await anext(events), InputSpeechEndEvent)
        await openai_session.send_audio(b'inter-turn silence')
        finish_openai.set()
        _ = [event async for event in events]

    gemini_session = RealtimeSession(
        FakeRealtimeConnection(
            [InputTranscript(text='hello', is_final=False, cumulative=True), ResponseCompleteEvent()]
        ),
        _noop_runner,
        audio_retention='input_audio',
    )
    await gemini_session.send_audio(b'speech')
    await gemini_session.send_audio(b'model-response silence')
    _ = await collect_events(gemini_session)

    assert openai_session.new_messages()[0] == ModelRequest(
        parts=[SpeechPart(speaker='user', transcript='hello', audio=_wav_content(b'speech'), id='turn')]
    )
    assert gemini_session.new_messages()[0] == ModelRequest(
        parts=[
            SpeechPart(
                speaker='user',
                transcript='hello',
                audio=_wav_content(b'speechmodel-response silence'),
            )
        ]
    )


async def test_audio_retention_uses_profile_rate_for_each_speaker() -> None:
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='hello', is_final=True),
            AudioDelta(data=b'\x01\x02'),
            OutputTranscript(text='hi', is_final=True),
            ResponseCompleteEvent(),
        ]
    )
    profile = RealtimeModelProfile(audio_input_sample_rate=16000, audio_output_sample_rate=24000)
    session = RealtimeSession(conn, _noop_runner, audio_retention='all', profile=profile)
    await session.send_audio(b'\xaa\xbb')
    _ = await collect_events(session)

    request, response = session.new_messages()
    assert isinstance(request, ModelRequest) and isinstance(response, ModelResponse)
    user, assistant = request.parts[0], response.parts[0]
    assert isinstance(user, SpeechPart) and user.audio is not None
    assert isinstance(assistant, SpeechPart) and assistant.audio is not None
    with wave.open(io.BytesIO(user.audio.data), 'rb') as user_wav:
        assert user_wav.getframerate() == 16000
    with wave.open(io.BytesIO(assistant.audio.data), 'rb') as assistant_wav:
        assert assistant_wav.getframerate() == 24000


async def test_clear_audio_discards_retained_input() -> None:
    # `clear_audio()` must drop the locally retained buffer too, or discarded audio would still attach
    # to the next finalized user turn (with `audio_retention='input_audio'`/`'all'`).
    conn = FakeRealtimeConnection([InputTranscript(text='hello', is_final=True)])
    session = RealtimeSession(conn, _noop_runner, audio_retention='input_audio')
    await session.send_audio(b'\xaa\xbb')
    await session.clear_audio()  # discards the buffered chunk
    await session.send_audio(b'\xcc')  # only this survives into the finalized user turn
    _ = await collect_events(session)
    assert session.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SpeechPart(
                        speaker='user',
                        transcript='hello',
                        audio=_wav_content(b'\xcc'),
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
        [InputSpeechEndEvent(), OutputTranscript(text='Hi', is_final=True), ResponseCompleteEvent()],
        input_transcription_enabled=False,
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m', audio_retention='input_audio')
    await session.send_audio(b'\xaa\xbb')
    events = await collect_events(session)
    user_part = SpeechPart(speaker='user', audio=_wav_content(b'\xaa\xbb'))
    assert events[:3] == [
        PartStartEvent(index=0, part=user_part),
        PartEndEvent(index=0, part=user_part),
        InputSpeechEndEvent(),
    ]
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user', audio=_wav_content(b'\xaa\xbb'))]),
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='Hi')],
                model_name='m',
                timestamp=IsDatetime(),
                finish_reason='stop',
            ),
        ]
    )


async def test_audio_only_user_turn_finalized_on_turn_complete() -> None:
    # Providers without a speech-stopped signal (e.g. Gemini): the audio-only user turn is finalized at
    # the turn-complete boundary, before the assistant response, so history reads user-then-assistant.
    conn = FakeRealtimeConnection(
        [OutputTranscript(text='Hi', is_final=True), ResponseCompleteEvent()],
        input_transcription_enabled=False,
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m', audio_retention='input_audio')
    await session.send_audio(b'\xaa\xbb')
    _ = await collect_events(session)
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user', audio=_wav_content(b'\xaa\xbb'))]),
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='Hi')],
                model_name='m',
                timestamp=IsDatetime(),
                finish_reason='stop',
            ),
        ]
    )


async def test_audio_only_user_turn_finalized_on_each_manual_commit() -> None:
    conn = FakeRealtimeConnection([], input_transcription_enabled=False)
    session = RealtimeSession(conn, model_name='m', audio_retention='input_audio')

    await session.send_audio(b'\xaa')
    await session.commit_audio()
    session._translate_event(OutputTranscript(text='first', is_final=True))  # pyright: ignore[reportPrivateUsage]
    session._translate_event(ResponseCompleteEvent())  # pyright: ignore[reportPrivateUsage]
    await session.send_audio(b'\xbb')
    await session.commit_audio()
    session._translate_event(OutputTranscript(text='second', is_final=True))  # pyright: ignore[reportPrivateUsage]
    session._translate_event(ResponseCompleteEvent())  # pyright: ignore[reportPrivateUsage]
    events = await collect_events(session)

    first_user = SpeechPart(speaker='user', audio=_wav_content(b'\xaa'))
    second_user = SpeechPart(speaker='user', audio=_wav_content(b'\xbb'))
    # Each turn's part gets its own session-unique index, so a consumer can tell the second user turn
    # from the first (and from the assistant response that took index 1 in between).
    assert events == [
        PartStartEvent(index=0, part=first_user),
        PartEndEvent(index=0, part=first_user),
        PartStartEvent(index=2, part=second_user),
        PartEndEvent(index=2, part=second_user),
    ]
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user', audio=_wav_content(b'\xaa'))]),
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='first')],
                model_name='m',
                timestamp=IsDatetime(),
                finish_reason='stop',
            ),
            ModelRequest(parts=[SpeechPart(speaker='user', audio=_wav_content(b'\xbb'))]),
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='second')],
                model_name='m',
                timestamp=IsDatetime(),
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.parametrize(
    ('state_restored', 'expected'),
    [
        pytest.param(
            False,
            snapshot(
                [
                    ModelResponse(
                        parts=[SpeechPart(speaker='assistant', transcript='before')],
                        timestamp=IsDatetime(),
                        state='interrupted',
                    ),
                    ModelResponse(
                        parts=[SpeechPart(speaker='assistant', transcript='after')],
                        timestamp=IsDatetime(),
                        finish_reason='stop',
                    ),
                ]
            ),
            id='state-lost',
        ),
        pytest.param(
            True,
            snapshot(
                [
                    ModelResponse(
                        parts=[SpeechPart(speaker='assistant', transcript='beforeafter')],
                        timestamp=IsDatetime(),
                        finish_reason='stop',
                    )
                ]
            ),
            id='state-restored',
        ),
    ],
)
async def test_reconnect_response_state(state_restored: bool, expected: list[ModelMessage]) -> None:
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='before', is_final=False),
            SessionReconnectEvent(state_restored=state_restored),
            OutputTranscript(text='after', is_final=True),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn)
    await collect_events(session)
    assert session.new_messages() == expected


async def test_session_offers_the_conversation_for_replay_when_seeding_is_supported() -> None:
    """The session hands the connection a live view of the call, for a provider that must replay it.

    Only where seeding is the mechanism: a provider that resumes natively (Gemini, xAI) has nothing to
    replay, and one that cannot seed at all has nowhere to put it.
    """

    class _RecordsConversation(FakeRealtimeConnection):
        def __init__(self, events: list[RealtimeCodecEvent]) -> None:
            super().__init__(events)
            self.conversation: Callable[[], Sequence[ModelMessage]] | None = None

        def set_conversation(self, conversation: Callable[[], Sequence[ModelMessage]]) -> None:
            self.conversation = conversation

    seeding = _RecordsConversation([])
    async with RealtimeSession(seeding, profile=_profile(supports_session_seeding=True)) as session:
        await session.send('hello')
    assert seeding.conversation is not None
    # A live view, not a snapshot: the call grows after the connection is handed it.
    assert [type(m).__name__ for m in seeding.conversation()] == ['ModelRequest']

    no_seeding = _RecordsConversation([])
    async with RealtimeSession(no_seeding, profile=_profile(supports_session_seeding=False)):
        pass
    assert no_seeding.conversation is None


async def test_reconnect_while_idle_passes_through() -> None:
    # A lost-state reconnect with nothing in flight has no pre-drop response to finalize: the event
    # passes through and the next turn is recorded normally.
    conn = FakeRealtimeConnection(
        [
            SessionReconnectEvent(state_restored=False),
            OutputTranscript(text='after', is_final=True),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn)
    events = await collect_events(session)
    assert any(isinstance(e, SessionReconnectEvent) for e in events)
    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='after')],
                timestamp=IsDatetime(),
                finish_reason='stop',
            )
        ]
    )


async def test_reconnect_finalizes_multiple_in_flight_user_items() -> None:
    # Two overlapping user turns (partial transcripts routed by item id), with the second finalizing
    # out of order: `u2` leaves `_active_users_by_id` but stays queued in `_user_item_order` behind the
    # still-open `u1`. A lost-state reconnect must finalize the still-active `u1` and skip the
    # already-finalized `u2` (which the pending-flush emits in order), so neither turn is dropped or
    # double-recorded.
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='first turn', is_final=False, item_id='u1'),
            InputTranscript(text='second turn', is_final=False, item_id='u2'),
            InputTranscript(text='second turn', is_final=True, item_id='u2'),
            SessionReconnectEvent(state_restored=False),
            OutputTranscript(text='after', is_final=True),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn)
    await collect_events(session)
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user', transcript='first turn', id='u1')]),
            ModelRequest(parts=[SpeechPart(speaker='user', transcript='second turn', id='u2')]),
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='after')],
                timestamp=IsDatetime(),
                finish_reason='stop',
            ),
        ]
    )


async def test_audio_retained_with_transcription_enabled_waits_for_transcript() -> None:
    # With transcription enabled, a speech-stopped boundary does NOT emit an audio-only turn: the turn is
    # finalized from the (asynchronously delivered) transcript instead, so there's exactly one user turn —
    # never a duplicate audio-only one racing the transcript.
    conn = FakeRealtimeConnection(
        [InputSpeechEndEvent(), InputTranscript(text='hello', is_final=True), ResponseCompleteEvent()],
        input_transcription_enabled=True,
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m', audio_retention='input_audio')
    await session.send_audio(b'\xaa\xbb')
    _ = await collect_events(session)
    assert session.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SpeechPart(
                        speaker='user',
                        transcript='hello',
                        audio=_wav_content(b'\xaa\xbb'),
                    )
                ]
            )
        ]
    )


async def test_input_audio_segmented_by_item_id_across_overlapping_turns() -> None:
    # With input audio retained and transcription enabled, each speech-stopped boundary carries the input
    # item id, so its audio is cut into a per-item segment. When two turns overlap and their transcripts
    # finalize out of order (the second turn's `is_final` arrives before the first's), each user message
    # still carries its own audio. Without segmentation the whole rolling buffer would attach to whichever
    # transcript finalized first, giving that turn both turns' audio and the other turn none.
    gate_a = asyncio.Event()
    gate_b = asyncio.Event()

    class _Overlapping(FakeRealtimeConnection):
        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            await gate_a.wait()
            yield InputSpeechEndEvent(item_id='A')  # segments turn A's audio
            await gate_b.wait()
            yield InputSpeechEndEvent(item_id='B')  # segments turn B's audio
            yield InputTranscript(text='second', is_final=True, item_id='B')  # B finalizes first...
            yield InputTranscript(text='first', is_final=True, item_id='A')  # ...then A, out of order
            yield ResponseCompleteEvent()

    conn = _Overlapping([])
    session = RealtimeSession(conn, _noop_runner, audio_retention='input_audio')
    await session.send_audio(b'\xaa')  # turn A's audio, buffered before its boundary fires
    gate_a.set()
    async with session:
        async for event in session:
            # When A's boundary has passed (its segment is captured), queue B's audio and release B's.
            if isinstance(event, InputSpeechEndEvent) and event.item_id == 'A':
                await session.send_audio(b'\xbb')
                gate_b.set()

    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user', transcript='second', audio=_wav_content(b'\xbb'), id='B')]),
            ModelRequest(parts=[SpeechPart(speaker='user', transcript='first', audio=_wav_content(b'\xaa'), id='A')]),
        ]
    )


async def test_retained_input_audio_kept_when_transcription_fails() -> None:
    # A failed transcript must keep the retained audio on its placeholder without leaking it to a later turn.
    conn = FakeRealtimeConnection(
        [
            InputSpeechEndEvent(item_id='A'),
            InputTranscriptionErrorEvent(message='transcription failed', item_id='A'),
            InputTranscript(text='hi', is_final=True, item_id='B'),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, audio_retention='input_audio')
    await session.send_audio(b'\xaa')
    _ = await collect_events(session)
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user', audio=_wav_content(b'\xaa'), id='A')]),
            ModelRequest(parts=[SpeechPart(speaker='user', transcript='hi', id='B')]),
        ]
    )


async def test_no_transcription_and_no_input_retention_records_placeholder_once() -> None:
    conn = FakeRealtimeConnection(
        [InputSpeechStartEvent(), InputSpeechEndEvent(), ResponseCompleteEvent()], input_transcription_enabled=False
    )
    session = RealtimeSession(conn, _noop_runner)
    events = await collect_events(session)

    assert [type(event).__name__ for event in events] == [
        'InputSpeechStartEvent',
        'PartStartEvent',
        'PartEndEvent',
        'InputSpeechEndEvent',
        'ResponseCompleteEvent',
        'TurnCompleteEvent',
    ]
    assert session.new_messages() == [ModelRequest(parts=[SpeechPart(speaker='user')])]


async def test_no_transcription_with_input_retention_is_allowed() -> None:
    # Disabling transcription is fine as long as input audio is retained (audio-only user turns).
    conn = FakeRealtimeConnection([], input_transcription_enabled=False)
    RealtimeSession(conn, _noop_runner, audio_retention='input_audio')  # no error


async def test_manual_contentless_user_turn_without_transcription() -> None:
    conn = FakeRealtimeConnection([], input_transcription_enabled=False)
    session = RealtimeSession(conn, _noop_runner)

    await session.commit_audio()
    events = await collect_events(session)

    assert session.new_messages() == [ModelRequest(parts=[SpeechPart(speaker='user')])]
    assert [type(event).__name__ for event in events] == ['PartStartEvent', 'PartEndEvent']


async def test_text_output_modality_produces_text_part() -> None:
    # A text-output response (`OutputTranscript(output_text=True)`, from `output_modalities=('text',)`) must
    # be emitted and persisted as a `TextPart`, not a `SpeechPart` — it carries no speech.
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='hi', is_final=False, output_text=True),
            OutputTranscript(text='hi there', is_final=True, output_text=True),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = await collect_events(session)
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


async def test_empty_assistant_turn_is_recorded() -> None:
    # Audio with no transcript and no retention leaves a content-less assistant placeholder in history.
    conn = FakeRealtimeConnection([AudioDelta(data=b'\x00\x01'), ResponseCompleteEvent()])
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = await collect_events(session)
    assert [type(e).__name__ for e in events] == [
        'PartStartEvent',
        'PartDeltaEvent',
        'PartEndEvent',
        'ResponseCompleteEvent',
        'TurnCompleteEvent',
    ]
    end = next(e for e in events if isinstance(e, PartEndEvent))
    assert isinstance(end.part, SpeechPart) and end.part.transcript is None and end.part.audio is None
    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[SpeechPart(speaker='assistant')],
                model_name='m',
                timestamp=IsDatetime(),
                finish_reason='stop',
            )
        ]
    )


async def test_empty_input_transcript_produces_placeholder_request() -> None:
    conn = FakeRealtimeConnection([InputTranscript(text='', is_final=True)])
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = await collect_events(session)
    assert [type(e).__name__ for e in events] == ['PartStartEvent', 'PartEndEvent']
    end = next(e for e in events if isinstance(e, PartEndEvent))
    assert isinstance(end.part, SpeechPart) and end.part.transcript is None
    assert session.new_messages() == [ModelRequest(parts=[SpeechPart(speaker='user')])]


async def test_duplicate_final_input_transcript_is_idempotent() -> None:
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='hello', is_final=True, item_id='user-1'),
            InputTranscript(text='hello', is_final=True, item_id='user-1'),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    _ = await collect_events(session)

    assert session.all_messages() == [ModelRequest(parts=[SpeechPart(speaker='user', transcript='hello', id='user-1')])]


async def test_transcript_only_default_drops_audio() -> None:
    conn = FakeRealtimeConnection(
        [AudioDelta(data=b'\x00'), OutputTranscript(text='hi', is_final=True), ResponseCompleteEvent()]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    _ = await collect_events(session)
    response = session.new_messages()[0]
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts[0], SpeechPart)
    assert response.parts[0].audio is None


# --- seeding + handoff --------------------------------------------------------------------------


async def test_all_messages_includes_seed_new_messages_excludes_it() -> None:
    seed = [ModelRequest(parts=[UserPromptPart(content='earlier')])]
    conn = FakeRealtimeConnection([OutputTranscript(text='reply', is_final=True), ResponseCompleteEvent()])
    session = RealtimeSession(conn, _noop_runner, model_name='m', message_history=seed)
    _ = await collect_events(session)
    assert session.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='earlier', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='reply')],
                model_name='m',
                timestamp=IsDatetime(),
                finish_reason='stop',
            ),
        ]
    )
    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='reply')],
                model_name='m',
                timestamp=IsDatetime(),
                finish_reason='stop',
            )
        ]
    )


async def test_snapshot_is_a_copy() -> None:
    conn = FakeRealtimeConnection([OutputTranscript(text='one', is_final=True), ResponseCompleteEvent()])
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    _ = await collect_events(session)
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
            OutputTranscript(text='hi there', is_final=True),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='gpt-realtime')
    _ = await collect_events(session)

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
                finish_reason='stop',
            ),
        ]
    )


async def test_contentless_speech_parts_are_skipped_for_seeding_and_text_handoff() -> None:
    user = SpeechPart(speaker='user')
    assistant = SpeechPart(speaker='assistant')
    assert seed_speech_content(user, provider_name='test', supports_audio=False) == ''
    assert seed_speech_content(assistant, provider_name='test', supports_audio=False) == ''

    captured: list[ModelMessage] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured.extend(messages)
        return ModelResponse(parts=[TextPart(content='ok')])

    history: list[ModelMessage] = [ModelRequest(parts=[user]), ModelResponse(parts=[assistant])]
    result = await Agent(FunctionModel(respond)).run('continue', message_history=history)

    assert result.output == 'ok'
    assert len(captured) == 1
    assert isinstance(captured[0], ModelRequest)
    assert captured[0].parts == [UserPromptPart(content='continue', timestamp=IsDatetime())]


async def test_retained_audio_prepares_for_audio_capable_classic_model() -> None:
    conn = FakeRealtimeConnection([InputTranscript(text='hello', is_final=True)])
    session = RealtimeSession(conn, _noop_runner, audio_retention='input_audio')
    await session.send_audio(b'\xaa\xbb')
    _ = await collect_events(session)

    prepared = TestModel(profile=ModelProfile(supports_audio_input=True)).prepare_messages(session.all_messages())
    request = prepared[0]
    assert isinstance(request, ModelRequest)
    prompt = request.parts[0]
    assert isinstance(prompt, UserPromptPart) and isinstance(prompt.content, list)
    audio = prompt.content[0]
    assert isinstance(audio, BinaryContent)
    assert audio.media_type == 'audio/wav'
    assert audio.format == 'wav'


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


def _native_part_events(
    parts: list[NativeToolCallPart | NativeToolReturnPart],
) -> list[PartStartEvent | PartEndEvent]:
    return [
        event
        for index, part in enumerate(parts)
        for event in (PartStartEvent(index=index, part=part), PartEndEvent(index=index, part=part))
    ]


async def test_grounding_streams_and_folds_native_tool_parts() -> None:
    # Grounding parts stream to the consumer and fold into the assistant response ahead of speech,
    # mirroring the classic `GoogleModel`.
    grounding = _grounding_parts()
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='It is sunny in Rome', is_final=True),
            *_native_part_events(grounding),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='gemini-live-2.5-flash')
    events = await collect_events(session)

    assert events == [
        PartStartEvent(index=0, part=SpeechPart(speaker='assistant', transcript='')),
        PartDeltaEvent(
            index=0,
            delta=SpeechPartDelta(
                speaker='assistant', transcript_delta='It is sunny in Rome', transcript='It is sunny in Rome'
            ),
        ),
        *_native_part_events(grounding),
        PartEndEvent(index=0, part=SpeechPart(speaker='assistant', transcript='It is sunny in Rome')),
        ResponseCompleteEvent(),
        TurnCompleteEvent(),
    ]

    assert session.new_messages() == [
        ModelResponse(
            parts=[*grounding, SpeechPart(speaker='assistant', transcript='It is sunny in Rome')],
            model_name='gemini-live-2.5-flash',
            timestamp=IsDatetime(),
            finish_reason='stop',
        )
    ]


async def test_grounded_history_hands_off_with_native_parts_intact() -> None:
    # The native tool parts from a grounded voice turn survive the `all_messages()` → `agent.run` handoff:
    # `Model.prepare_messages` passes `NativeToolCallPart`/`NativeToolReturnPart` through untouched (only
    # the `SpeechPart`s are converted to the plain user-prompt / text shapes any model can consume).
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='weather in rome?', is_final=True),
            OutputTranscript(text='It is sunny in Rome', is_final=True),
            *_native_part_events(_grounding_parts()),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='gemini-live-2.5-flash')
    _ = await collect_events(session)

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
                finish_reason='stop',
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
            OutputTranscript(text='The answer is 2.', is_final=True),
            *_native_part_events(_code_execution_parts()),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='gemini-live-2.5-flash')
    _ = await collect_events(session)

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
                finish_reason='stop',
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
        [ToolCall(tool_call_id='tc_5', tool_name='greet', args='{"name": "Alice"}'), ResponseCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)

    async with agent.realtime(model).session() as session:
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
    conn = FakeRealtimeConnection([ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model, message_history=seed).session() as session:
        _ = [e async for e in session]
        assert session.all_messages() == seed  # seeded into the session's history
    assert model.last_messages == [
        *seed,
        ModelRequest(parts=[]),
    ]  # provider request view includes the current instruction-bearing request


async def test_agent_realtime_session_rejects_seeding_when_unsupported() -> None:
    # A model that can't seed a session rejects `message_history` up front, before dialing.
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn, profile=_profile(supports_session_seeding=False))
    seed = [ModelRequest(parts=[UserPromptPart(content='earlier question')])]
    with pytest.raises(UserError, match='does not support seeding a session'):
        async with agent.realtime(model, message_history=seed).session():
            pass  # pragma: no cover


async def test_agent_realtime_session_audio_retention_forwarded() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection(
        [AudioDelta(data=b'\x07'), OutputTranscript(text='hi', is_final=True), ResponseCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session(audio_retention='output_audio') as session:
        _ = [e async for e in session]
        response = session.new_messages()[0]
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts[0], SpeechPart)
    assert response.parts[0].audio == _wav_content(b'\x07')


async def test_agent_realtime_session_image_retention_forwarded() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([])
    model = FakeRealtimeModel(conn)
    images = [BinaryImage(data=bytes([index]), media_type='image/png') for index in range(3)]

    async with agent.realtime(model).session(retain_images_every_n=2) as session:
        for image in images:
            await session.send(image)
        retained = session.new_messages()

    assert retained == [
        ModelRequest(parts=[UserPromptPart(content=[images[0]], timestamp=IsDatetime())]),
        ModelRequest(parts=[UserPromptPart(content=[images[2]], timestamp=IsDatetime())]),
    ]


async def test_agent_realtime_session_additional_instructions() -> None:
    agent: Agent[None, str] = Agent(instructions='Default')
    conn = FakeRealtimeConnection([ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model, instructions='Custom').session() as session:
        _ = [e async for e in session]
    assert model.last_instructions == 'Default\nCustom'


async def test_agent_realtime_session_default_instructions_empty() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        _ = [e async for e in session]
    assert model.last_instructions == ''


async def test_agent_realtime_session_unknown_tool() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='tc_x', tool_name='nonexistent', args='{}'), ResponseCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        events = [e async for e in session]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert 'Unknown tool name' in str(result.part.content)
    assert 'nonexistent' in str(result.part.content)


async def test_agent_realtime_session_tool_exception() -> None:
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def explode() -> str:
        raise ValueError('nope')

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='tc', tool_name='explode', args='{}'), ResponseCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        with pytest.raises(ValueError, match='nope'):
            _ = [e async for e in session]
    assert conn.sent == []


async def test_agent_realtime_session_tool_failed_returns_error_result() -> None:
    """A tool raising `ToolFailed` yields a `failed`, error-key-wrapped result — not a crashed session.

    `tool_manager.handle_call` raises `ToolFailedError` for a `ToolFailed`; the session must answer with
    the failed result (like `run`/`iter`) rather than let it tear down the session. Realtime providers
    have no native failed-tool flag, so the failure is wrapped in an `{"error": ...}` object on the
    string-only tool channel.
    """
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def boom() -> str:
        raise ToolFailed('service down')

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='boom', args='{}'), ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        events = [e async for e in session]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, ToolReturnPart)
    assert result.part.outcome == 'failed'
    sent = next(s for s in conn.sent if isinstance(s, ToolResult))
    assert '"error"' in sent.output  # wrapped so the model sees the failure over the string-only channel


async def test_agent_realtime_session_validates_and_coerces_args() -> None:
    agent: Agent[None, str] = Agent()
    seen: int | None = None

    @agent.tool_plain
    def double(x: int) -> str:
        nonlocal seen
        seen = x
        return str(x * 2)

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='tc', tool_name='double', args='{"x": "21"}'), ResponseCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        events = [e async for e in session]

    assert seen == 21
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert result.part.content == '42'


async def test_agent_realtime_session_invalid_args_return_retry_message() -> None:
    agent: Agent[None, str] = Agent()

    # Never reached; validation fails first.
    @agent.tool_plain
    def double(x: int) -> str:  # pragma: no cover
        return str(x * 2)

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='tc', tool_name='double', args='{"x": "not a number"}'), ResponseCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        events = [e async for e in session]

    call = next(e for e in events if isinstance(e, FunctionToolCallEvent))
    assert call.args_valid is False
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, RetryPromptPart)
    assert 'validation error' in result.part.model_response()
    assert isinstance(session.new_messages()[1], ModelRequest)
    assert isinstance(session.new_messages()[1].parts[0], RetryPromptPart)


class _ToolRoundConnection(FakeRealtimeConnection):
    """Wait for the first retry result before yielding the next tool-call round."""

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        yield ToolCall(tool_call_id='tc1', tool_name='double', args='{"x": 1}')
        while len(self.sent) < 1:
            await asyncio.sleep(0)
        yield ToolCall(tool_call_id='tc2', tool_name='double', args='{"x": 2}')


class _EnqueueConnection(FakeRealtimeConnection):
    """Hold the turn boundary until the tool result has been sent."""

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        yield ToolCall(
            tool_call_id='tc',
            tool_name='queue_followup',
            args='{}',
            response_usage_follows=True,
        )
        while not any(isinstance(item, ToolResult) for item in self.sent):
            await asyncio.sleep(0)
        # Let the pending-message task observe that response usage is still outstanding. The queued
        # prompt must remain deferred until the usage event finalizes the tool-call response.
        await asyncio.sleep(0)
        yield SessionUsageEvent(usage=RequestUsage(input_tokens=1, output_tokens=1))
        yield ResponseCompleteEvent()


class _EnqueueDuringSpeechConnection(FakeRealtimeConnection):
    """Enqueue while assistant audio is active and expose whether delivery waits for its boundary."""

    def __init__(self) -> None:
        super().__init__([])
        self.audio_started = asyncio.Event()
        self.enqueued = asyncio.Event()
        self.sent_before_response_complete: list[RealtimeInput] = []

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        yield ToolCall(tool_call_id='tc', tool_name='queue_during_speech', args='{}')
        while not any(isinstance(item, ToolResult) for item in self.sent):
            await asyncio.sleep(0)
        self.audio_started.set()
        yield AudioDelta(data=b'audio')
        await self.enqueued.wait()
        await asyncio.sleep(0)
        self.sent_before_response_complete = list(self.sent)
        yield OutputTranscript(text='still speaking')
        yield ResponseCompleteEvent()


async def test_asap_enqueue_waits_for_active_response_to_complete() -> None:
    """`asap` is provider-agnostic: active assistant output finishes before queued text is sent."""
    agent: Agent[None, str] = Agent()
    conn = _EnqueueDuringSpeechConnection()

    @agent.tool
    async def queue_during_speech(ctx: RunContext[object]) -> str:
        async def enqueue_after_audio_starts() -> None:
            await conn.audio_started.wait()
            ctx.enqueue('follow-up context')
            conn.enqueued.set()

        asyncio.create_task(enqueue_after_audio_starts())
        return 'armed'

    async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
        _ = [event async for event in session]

    assert not any(isinstance(item, TextInput) for item in conn.sent_before_response_complete)
    assert [item.text for item in conn.sent if isinstance(item, TextInput)] == ['follow-up context']


@pytest.mark.parametrize(
    'priority',
    ['asap', 'when_idle'],
)
async def test_agent_realtime_session_delivers_enqueued_text(priority: Literal['asap', 'when_idle']) -> None:
    agent: Agent[None, str] = Agent()

    @agent.tool
    def queue_followup(ctx: RunContext[object]) -> str:
        assert ctx.enqueue('follow-up context', priority=priority) is not None
        return 'queued'

    conn = _EnqueueConnection([])
    async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
        _ = [event async for event in session]

    assert [type(item).__name__ for item in conn.sent] == ['ToolResult', 'TextInput']
    call_response, tool_return, followup = session.new_messages()
    assert isinstance(call_response, ModelResponse) and isinstance(call_response.parts[0], ToolCallPart)
    assert isinstance(tool_return, ModelRequest) and isinstance(tool_return.parts[0], ToolReturnPart)
    assert followup == ModelRequest(parts=[UserPromptPart(content='follow-up context', timestamp=IsDatetime())])


class _ConcurrentEnqueueConnection(FakeRealtimeConnection):
    """Block the first pending-message send while a second sync tool appends to the queue."""

    def __init__(self) -> None:
        super().__init__([])
        self.first_send_started = ThreadEvent()
        self.second_enqueued = ThreadEvent()

    async def send(self, content: RealtimeInput) -> None:
        self.sent.append(content)
        if isinstance(content, TextInput) and content.text == 'first':
            self.first_send_started.set()
            while not self.second_enqueued.is_set():
                await asyncio.sleep(0)

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        yield ToolCall(tool_call_id='tc-1', tool_name='queue_concurrently', args='{"text": "first"}')
        yield ToolCall(tool_call_id='tc-2', tool_name='queue_concurrently', args='{"text": "second"}')
        while sum(isinstance(item, ToolResult) for item in self.sent) < 2:
            await asyncio.sleep(0)
        yield ResponseCompleteEvent()


async def test_sync_tool_enqueue_during_drain_is_not_lost() -> None:
    agent: Agent[None, str] = Agent()
    conn = _ConcurrentEnqueueConnection()

    @agent.tool
    def queue_concurrently(ctx: RunContext[object], text: str) -> str:
        if text == 'second':
            assert conn.first_send_started.wait(timeout=5)
        assert ctx.enqueue(text) is not None
        if text == 'second':
            conn.second_enqueued.set()
        return text

    async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
        _ = [event async for event in session]

    assert [item.text for item in conn.sent if isinstance(item, TextInput)] == ['first', 'second']
    prompts = [
        part.content
        for message in session.new_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    assert prompts == ['first', 'second']


async def test_agent_realtime_session_rejects_non_text_enqueue() -> None:
    agent: Agent[object, str] = Agent()

    @agent.tool
    def queue_image(ctx: RunContext[object]) -> str:
        ctx.enqueue(BinaryImage(data=b'image', media_type='image/png'))
        return 'unreachable'  # pragma: no cover

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='queue_image', args='{}')])
    async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
        with pytest.raises(UserError, match='currently supports one plain-text prompt per call'):
            _ = [event async for event in session]


@pytest.mark.parametrize(
    'messages',
    [
        [ModelResponse(parts=[TextPart(content='not a request')])],
        [ModelRequest(parts=[UserPromptPart(content='one'), UserPromptPart(content='two')])],
    ],
)
async def test_realtime_pending_messages_reject_unsupported_message_shapes(messages: list[ModelMessage]) -> None:
    session = RealtimeSession(FakeRealtimeConnection([]))
    manager = session._tool_manager  # pyright: ignore[reportPrivateUsage]
    assert manager.ctx is not None
    assert manager.ctx.pending_messages is not None
    with pytest.raises(UserError, match='one plain-text prompt per call'):
        manager.ctx.pending_messages.append(PendingMessage(messages=messages))


async def test_session_exit_is_idempotent_and_flushes_unfinalized_user() -> None:
    session = RealtimeSession(FakeRealtimeConnection([InputTranscript(text='partial')]))
    await session.__aexit__(None, None, None)
    async with session:
        _ = [event async for event in session]
    await session.__aexit__(None, None, None)

    assert session.new_messages() == [ModelRequest(parts=[SpeechPart(speaker='user', transcript='partial')])]


def test_session_accepts_unprepared_tool_manager_without_pending_context() -> None:
    manager = ToolManager(FunctionToolset())
    session = _RealtimeSession(FakeRealtimeConnection([]), manager)
    assert session._tool_manager.ctx is None  # pyright: ignore[reportPrivateUsage]


async def test_replayed_items_are_suppressed_by_item_and_tool_call_id() -> None:
    conn = FakeRealtimeConnection(
        [
            ConversationCreated('conversation-1'),
            ConversationItemCreated(item_id='replayed-item', tool_call_id='replayed-call', replayed=True),
            AudioDelta(data=b'audio', item_id='replayed-item'),
            OutputTranscript(text='assistant', item_id='replayed-item'),
            InputTranscript(text='user', item_id='replayed-item'),
            ToolCall(
                tool_call_id='replayed-call',
                tool_name='noop',
                args='{}',
                item_id='new-item',
            ),
            ToolCallCancelled(tool_call_ids=['unknown-call']),
        ]
    )
    session = RealtimeSession(conn)

    assert await collect_events(session) == []
    assert session.new_messages() == []


async def test_existing_assistant_speech_adopts_late_item_id() -> None:
    session = RealtimeSession(
        FakeRealtimeConnection(
            [
                AudioDelta(data=b'first'),
                AudioDelta(data=b'second', item_id='assistant-item'),
                OutputTranscript(text='spoken', item_id='assistant-item'),
                ResponseCompleteEvent(),
            ]
        ),
        provider_name='openai',
    )
    _ = await collect_events(session)

    response = session.new_messages()[0]
    assert isinstance(response, ModelResponse)
    part = response.parts[0]
    assert isinstance(part, SpeechPart)
    assert part.id == 'assistant-item'


async def test_empty_finalized_user_precedes_later_item() -> None:
    session = RealtimeSession(
        FakeRealtimeConnection(
            [
                InputTranscript(text='', is_final=True, item_id='empty'),
                InputTranscript(text='kept', is_final=True, item_id='kept'),
            ]
        )
    )
    _ = await collect_events(session)

    assert session.new_messages() == [
        ModelRequest(parts=[SpeechPart(speaker='user', id='empty')]),
        ModelRequest(parts=[SpeechPart(speaker='user', transcript='kept', id='kept')]),
    ]


async def test_session_close_recovers_finalized_user_orphaned_from_ordered_stream() -> None:
    session = RealtimeSession(FakeRealtimeConnection([]))
    session._user_item_order.append('missing')  # pyright: ignore[reportPrivateUsage]
    session._user_item_order.append('empty')  # pyright: ignore[reportPrivateUsage]
    session._finalized_users_by_id['empty'] = SpeechPart(speaker='user')  # pyright: ignore[reportPrivateUsage]
    session._user_item_order.append('orphan')  # pyright: ignore[reportPrivateUsage]
    session._finalized_users_by_id['orphan'] = SpeechPart(  # pyright: ignore[reportPrivateUsage]
        speaker='user', transcript='recovered'
    )

    async with session:
        pass

    assert session.new_messages() == [
        ModelRequest(parts=[SpeechPart(speaker='user')]),
        ModelRequest(parts=[SpeechPart(speaker='user', transcript='recovered')]),
    ]


async def test_replayed_item_tracking_accepts_each_identifier_independently() -> None:
    session = RealtimeSession(FakeRealtimeConnection([]))
    session._handle_conversation_item(  # pyright: ignore[reportPrivateUsage]
        ConversationItemCreated(item_id='item-only', replayed=True)
    )
    session._handle_conversation_item(  # pyright: ignore[reportPrivateUsage]
        ConversationItemCreated(tool_call_id='call-only', replayed=True)
    )

    assert not session._accept_item('item-only')  # pyright: ignore[reportPrivateUsage]
    assert not session._accept_item(None, 'call-only')  # pyright: ignore[reportPrivateUsage]

    # A normal (non-replayed) conversation item is not resumption traffic, so it records nothing and
    # every later event for it is accepted. Pinned directly rather than relying on incidental coverage
    # from a provider WS test.
    session._handle_conversation_item(  # pyright: ignore[reportPrivateUsage]
        ConversationItemCreated(item_id='live-item', tool_call_id='live-call', replayed=False)
    )
    assert session._accept_item('live-item')  # pyright: ignore[reportPrivateUsage]
    assert session._accept_item(None, 'live-call')  # pyright: ignore[reportPrivateUsage]


def test_asap_notifications_without_live_loop_and_after_close_are_ignored() -> None:
    session = RealtimeSession(FakeRealtimeConnection([]))
    session._notify_asap_pending_messages()  # pyright: ignore[reportPrivateUsage]
    session._closed = True  # pyright: ignore[reportPrivateUsage]
    session._start_asap_pending_message_drain()  # pyright: ignore[reportPrivateUsage]


async def test_failed_asap_drain_is_forwarded_to_session_iterator() -> None:
    class _FailingSend(FakeRealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            raise RuntimeError('send failed')

    session = RealtimeSession(_FailingSend([]))
    manager = session._tool_manager  # pyright: ignore[reportPrivateUsage]
    assert manager.ctx is not None
    assert manager.ctx.pending_messages is not None
    async with session:
        manager.ctx.pending_messages.append(
            PendingMessage(messages=[ModelRequest(parts=[UserPromptPart(content='queued')])], priority='asap')
        )
        with pytest.raises(RuntimeError, match='send failed'):
            _ = [event async for event in session]


async def test_tool_completion_drains_messages_deferred_until_usage_arrives(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn)
    session._asap_drain_deferred = True  # pyright: ignore[reportPrivateUsage]
    session._pending_messages.append(  # pyright: ignore[reportPrivateUsage]
        PendingMessage(
            messages=[ModelRequest(parts=[UserPromptPart(content='after tool')])],
            priority='asap',
        )
    )
    validation_done = asyncio.Event()

    async def complete_after_usage(
        call: ToolCall, call_part: ToolCallPart, validation_done: asyncio.Event
    ) -> tuple[ToolReturnPart, None]:
        del call, validation_done
        session._tool_calls_awaiting_usage.clear()  # pyright: ignore[reportPrivateUsage]
        return ToolReturnPart(tool_name=call_part.tool_name, content='done', tool_call_id=call_part.tool_call_id), None

    session._tool_calls_awaiting_usage.add('call')  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(session, '_execute_tool', complete_after_usage)
    await session._run_tool(  # pyright: ignore[reportPrivateUsage]
        ToolCall(tool_call_id='call', tool_name='noop', args='{}'),
        ToolCallPart(tool_name='noop', args={}, tool_call_id='call'),
        validation_done,
    )

    assert TextInput('after tool') in conn.sent


async def test_deferred_asap_drain_failure_after_tool_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    # The post-`finally` deferred `asap` drain in `_run_tool` runs OUTSIDE its try/except. If its
    # `connection.send` fails (e.g. the socket just dropped), `_tool_task_done` must forward the error to
    # the consumer — mirroring `_pending_message_task_done` — instead of letting it vanish as an
    # unretrieved-task-exception warning at GC, silently losing the enqueued follow-up.
    class _FailingDrain(FakeRealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            if isinstance(content, TextInput):
                raise RuntimeError('drain send failed')
            # This test only drives the drain's `TextInput` send.
            await super().send(content)  # pragma: no cover

    conn = _FailingDrain([])
    session = RealtimeSession(conn)
    session._asap_drain_deferred = True  # pyright: ignore[reportPrivateUsage]
    session._pending_messages.append(  # pyright: ignore[reportPrivateUsage]
        PendingMessage(messages=[ModelRequest(parts=[UserPromptPart(content='after tool')])], priority='asap')
    )

    async def complete_after_usage(
        call: ToolCall, call_part: ToolCallPart, validation_done: asyncio.Event
    ) -> tuple[ToolReturnPart, None]:
        del call, validation_done
        session._tool_calls_awaiting_usage.clear()  # pyright: ignore[reportPrivateUsage]
        return ToolReturnPart(tool_name=call_part.tool_name, content='done', tool_call_id=call_part.tool_call_id), None

    session._tool_calls_awaiting_usage.add('call')  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(session, '_execute_tool', complete_after_usage)

    task = asyncio.create_task(
        session._run_tool(  # pyright: ignore[reportPrivateUsage]
            ToolCall(tool_call_id='call', tool_name='noop', args='{}'),
            ToolCallPart(tool_name='noop', args={}, tool_call_id='call'),
            asyncio.Event(),
        )
    )
    task.add_done_callback(session._tool_task_done)  # pyright: ignore[reportPrivateUsage]
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)  # let the done-callback run

    queued: list[Any] = []
    while not session._queue.empty():  # pyright: ignore[reportPrivateUsage]
        queued.append(session._queue.get_nowait())  # pyright: ignore[reportPrivateUsage]
    assert any(isinstance(item, RuntimeError) and str(item) == 'drain send failed' for item in queued)


async def test_tool_call_limit_stops_pump_before_later_events() -> None:
    async def runner(*args: Any) -> str:
        return 'ok'

    session = RealtimeSession(
        FakeRealtimeConnection(
            [
                ToolCall(tool_call_id='first', tool_name='noop', args='{}'),
                ToolCall(tool_call_id='second', tool_name='noop', args='{}'),
                ResponseCompleteEvent(),
            ]
        ),
        runner=runner,
        usage_limits=UsageLimits(tool_calls_limit=1),
    )

    async with session:
        with pytest.raises(UsageLimitExceeded, match='tool_calls_limit'):
            _ = [event async for event in session]
    assert session.usage.tool_calls == 1


async def test_iterator_reuses_receive_pump_started_by_session_owner() -> None:
    session = RealtimeSession(FakeRealtimeConnection([ResponseCompleteEvent()]))
    async with session:
        session._pump_task = asyncio.create_task(  # pyright: ignore[reportPrivateUsage]
            session._pump(session._session_span_context)  # pyright: ignore[reportPrivateUsage]
        )
        assert [event async for event in session] == [ResponseCompleteEvent(), TurnCompleteEvent()]


async def test_receive_pump_stops_when_event_handler_trips_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    session = RealtimeSession(FakeRealtimeConnection([ResponseCompleteEvent(), ResponseCompleteEvent()]))
    handled = 0

    async def stop_after_first(event: RealtimeCodecEvent) -> bool:
        nonlocal handled
        handled += 1
        return True

    monkeypatch.setattr(session, '_handle_pump_event', stop_after_first)
    await session._pump(None)  # pyright: ignore[reportPrivateUsage]

    assert handled == 1


async def test_tool_manager_reports_validation_failure_when_retry_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = make_tool_manager()
    outcomes: list[bool] = []

    async def exhausted(*args: Any, **kwargs: Any) -> Any:
        raise UnexpectedModelBehavior('retry budget exhausted')

    async def record_validation(valid: bool) -> None:
        outcomes.append(valid)

    monkeypatch.setattr(manager, 'validate_tool_call', exhausted)
    with pytest.raises(UnexpectedModelBehavior, match='retry budget exhausted'):
        await manager.handle_call(
            ToolCallPart(tool_name='noop', args={}, tool_call_id='call'),
            on_validate=record_validation,
        )
    assert outcomes == [False]


async def test_agent_realtime_session_retry_limit_advances_across_tool_rounds() -> None:
    agent: Agent[None, str] = Agent(retries=1)

    @agent.tool_plain
    def double(x: int) -> str:
        raise ModelRetry(f'{x} is not allowed')

    conn = _ToolRoundConnection([])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        events: list[RealtimeEvent] = []
        with pytest.raises(UnexpectedModelBehavior, match="Tool 'double' exceeded max retries count of 1"):
            async for event in session:
                events.append(event)

    results = [e.part for e in events if isinstance(e, FunctionToolResultEvent)]
    assert len(results) == 1 and isinstance(results[0], RetryPromptPart)
    assert str(results[0].content).startswith('1 is not allowed')
    assert conn.sent == [ToolResult(tool_call_id='tc1', output=results[0].model_response())]


async def test_agent_realtime_session_runs_args_validator() -> None:
    agent: Agent[None, str] = Agent()

    def guard(ctx: RunContext[Any], city: str) -> None:
        raise ModelRetry('not allowed')

    # Never reached; the validator rejects first.
    @agent.tool_plain(args_validator=guard)
    def weather(city: str) -> str:  # pragma: no cover
        return f'sunny in {city}'

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='tc', tool_name='weather', args='{"city": "forbidden"}'), ResponseCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        events = [e async for e in session]

    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert 'not allowed' in str(result.part.content)


@pytest.mark.parametrize(
    ('follow_up_content', 'expected_wire_content'),
    [
        ('extra context', ['extra context']),
        (
            ['extra context', BinaryContent(data=b'image', media_type='image/png')],
            ['extra context', BinaryContent(data=b'image', media_type='image/png')],
        ),
    ],
)
async def test_agent_realtime_session_tool_return_is_unwrapped(
    follow_up_content: str | list[str | BinaryContent], expected_wire_content: list[str | BinaryContent]
) -> None:
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def info() -> ToolReturn:
        return ToolReturn(
            return_value={'value': 42},
            content=follow_up_content,
            metadata={'source': 'tool'},
        )

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='info', args='{}'), ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        events = [e async for e in session]

    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, ToolReturnPart)
    assert result.part.content == {'value': 42}
    assert result.part.metadata == {'source': 'tool'}
    assert result.content == follow_up_content
    assert conn.sent == [
        ToolResult(
            tool_call_id='tc',
            output='{"value":42}',
            content=expected_wire_content,
        )
    ]
    request = next(message for message in session.new_messages() if isinstance(message, ModelRequest))
    assert request.parts == [
        result.part,
        UserPromptPart(content=follow_up_content, timestamp=IsDatetime()),
    ]


async def test_agent_realtime_session_denied_tool_returns_denial_message() -> None:
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def danger() -> str:
        raise ApprovalRequired()

    def deny(ctx: RunContext[Any], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: False for call in requests.approvals})

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='danger', args='{}'), ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model, capabilities=[HandleDeferredToolCalls(handler=deny)]).session() as session:
        events = [e async for e in session]

    lifecycle = [event for event in events if isinstance(event, (DeferredToolRequestsEvent, DeferredToolResultsEvent))]
    assert lifecycle == [
        DeferredToolRequestsEvent(
            DeferredToolRequests(approvals=[ToolCallPart(tool_name='danger', args={}, tool_call_id='tc')])
        ),
        DeferredToolResultsEvent(DeferredToolResults(approvals={'tc': False})),
    ]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, ToolReturnPart)
    assert result.part.outcome == 'denied'
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

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='whoami', args='{}'), ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model, deps='alice').session() as session:
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
        [ToolCall(tool_call_id='tc', tool_name='inspect_model', args='{}'), ResponseCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
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
        [ToolCall(tool_call_id='tc', tool_name='inspect_model', args='{}'), ResponseCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        _ = [e async for e in session]
    assert seen_system == 'fake'


async def test_agent_realtime_session_concurrent_tools_end_to_end() -> None:
    agent: Agent[None, str] = Agent()
    release = asyncio.Event()

    @agent.tool_plain
    async def slow_lookup() -> str:
        await release.wait()
        return 'background result'

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='bg', tool_name='slow_lookup', args='{}'), ResponseCompleteEvent()],
        release=release,
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        events = [e async for e in session]

    assert [type(e).__name__ for e in events] == snapshot(
        ['PartStartEvent', 'PartEndEvent', 'FunctionToolCallEvent', 'ResponseCompleteEvent', 'FunctionToolResultEvent']
    )
    result = events[-1]
    assert isinstance(result, FunctionToolResultEvent)
    assert result.part.content == 'background result'


async def test_agent_realtime_session_forwards_model_settings() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    settings = RealtimeModelSettings(max_tokens=50)
    async with agent.realtime(model, model_settings=settings).session() as session:
        _ = [e async for e in session]
    assert model.last_model_settings == settings


async def test_agent_realtime_session_merges_model_and_call_settings() -> None:
    """Call-time realtime settings override the model defaults key by key."""
    agent: Agent[None, str] = Agent(model_settings=ModelSettings(temperature=0.1, max_tokens=100))
    conn = FakeRealtimeConnection([ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn, settings=RealtimeModelSettings(max_tokens=100, parallel_tool_calls=False))
    async with agent.realtime(
        model, model_settings=RealtimeModelSettings(parallel_tool_calls=True)
    ).session() as session:
        _ = [e async for e in session]
    assert model.last_model_settings == RealtimeModelSettings(max_tokens=100, parallel_tool_calls=True)


async def test_agent_realtime_session_ignores_regular_model_settings_override() -> None:
    """`Agent.override(model_settings=...)` does not affect realtime settings."""
    agent: Agent[None, str] = Agent(model_settings=ModelSettings(temperature=0.1))
    conn = FakeRealtimeConnection([ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    with agent.override(model_settings=ModelSettings(temperature=0.9)):
        async with agent.realtime(model, model_settings=RealtimeModelSettings(max_tokens=50)).session() as session:
            _ = [e async for e in session]
    assert model.last_model_settings == RealtimeModelSettings(max_tokens=50)


async def test_agent_realtime_session_send_audio() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        await session.send_audio(b'\xab\xcd')
    assert conn.sent == [AudioInput(data=b'\xab\xcd')]


# --- parity with run/iter: instructions, toolsets, usage, usage_limits, capabilities, metadata ---


async def test_agent_realtime_session_dynamic_instructions() -> None:
    agent: Agent[None, str] = Agent(instructions='Base')

    @agent.instructions
    def extra() -> str:
        return 'Dynamic'

    @agent.instructions
    def skipped() -> str | None:
        return None  # a dynamic instruction returning None contributes nothing

    conn = FakeRealtimeConnection([ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        _ = [e async for e in session]
    # Static literal then dynamic function, double-newline separated — same as `run`/`iter`.
    assert model.last_instructions == 'Base\n\nDynamic'


async def test_agent_realtime_session_dynamic_instructions_see_message_history() -> None:
    """A dynamic instruction function sees `message_history` via `ctx.messages`, like `run`/`iter`.

    Regression: `realtime_session` used to leave `RunContext.messages` empty, so a dynamic instruction
    (or a capability `for_run` hook) that read `ctx.messages` saw `[]` even when the caller passed a
    `message_history`.
    """
    agent: Agent[None, str] = Agent()

    @agent.instructions
    def prior_count(ctx: RunContext) -> str:
        return f'{len(ctx.messages)} prior messages'

    seed = [
        ModelRequest(parts=[UserPromptPart(content='earlier question')]),
        ModelResponse(parts=[TextPart(content='earlier answer')]),
    ]
    conn = FakeRealtimeConnection([ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model, message_history=seed).session() as session:
        _ = [e async for e in session]
    assert model.last_instructions == '2 prior messages'


async def test_agent_realtime_session_additional_toolsets() -> None:
    agent: Agent[None, str] = Agent()
    extra_toolset: FunctionToolset[object] = FunctionToolset()

    @extra_toolset.tool_plain
    def extra_tool() -> str:
        return 'x'

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='t1', tool_name='extra_tool', args='{}'), ResponseCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model, toolsets=[extra_toolset]).session() as session:
        events = [e async for e in session]
    assert 'extra_tool' in [t.name for t in model.last_tools or []]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert result.part.content == 'x'  # the extra toolset's tool is offered AND callable


async def test_agent_realtime_session_external_usage_accumulates() -> None:
    usage = RunUsage()
    conn = FakeRealtimeConnection(
        [SessionUsageEvent(usage=RequestUsage(input_tokens=7, output_tokens=3)), ResponseCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    agent: Agent[None, str] = Agent()
    async with agent.realtime(model, usage=usage).session() as session:
        assert session.usage is usage  # the provided accumulator is used
        _ = [e async for e in session]
    assert usage.input_tokens == 7
    assert usage.output_tokens == 3


async def test_run_level_usage_is_not_attributed_to_or_finalize_response() -> None:
    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'done'

    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='first response', is_final=True),
            ToolCall(tool_call_id='pending', tool_name='noop', args='{}', response_usage_follows=True),
            SessionUsageEvent(usage=RequestUsage(details={'input_transcription_seconds': 3}), response_scoped=False),
            OutputTranscript(text='second response', is_final=True),
            ResponseCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, runner)

    _ = await collect_events(session)

    assert session.usage.details == {'input_transcription_seconds': 3}
    responses = [message for message in session.new_messages() if isinstance(message, ModelResponse)]
    assert len(responses) == 1
    assert responses[0].usage.details == {}


async def test_agent_realtime_session_token_limit_raises() -> None:
    conn = FakeRealtimeConnection(
        [SessionUsageEvent(usage=RequestUsage(input_tokens=100, output_tokens=100)), ResponseCompleteEvent()]
    )
    model = FakeRealtimeModel(conn)
    agent: Agent[None, str] = Agent()
    async with agent.realtime(model, usage_limits=UsageLimits(total_tokens_limit=50)).session() as session:
        with pytest.raises(UsageLimitExceeded, match='Exceeded the total_tokens_limit of 50'):
            _ = [e async for e in session]


async def test_agent_realtime_session_request_limit_raises() -> None:
    conn = FakeRealtimeConnection(
        [
            SessionUsageEvent(usage=RequestUsage(input_tokens=1, output_tokens=1)),
            OutputTranscript(text='first', is_final=True),
            ResponseCompleteEvent(),
            SessionUsageEvent(usage=RequestUsage(input_tokens=1, output_tokens=1)),
            OutputTranscript(text='second', is_final=True),
            ResponseCompleteEvent(),
        ]
    )
    model = FakeRealtimeModel(conn)
    agent: Agent[None, str] = Agent()
    async with agent.realtime(model, usage_limits=UsageLimits(request_limit=1)).session() as session:
        events: list[RealtimeEvent] = []
        with pytest.raises(UsageLimitExceeded, match='next request would exceed the request_limit of 1'):
            async for event in session:
                events.append(event)
    assert sum(isinstance(event, ResponseCompleteEvent) for event in events) == 1
    assert session.usage.requests == 2


async def test_agent_realtime_session_response_without_usage_counts_toward_request_limit() -> None:
    conn = FakeRealtimeConnection([OutputTranscript(text='response', is_final=True), ResponseCompleteEvent()])
    agent: Agent[None, str] = Agent()
    async with agent.realtime(FakeRealtimeModel(conn), usage_limits=UsageLimits(request_limit=1)).session() as session:
        _ = [event async for event in session]
    assert session.usage.requests == 1


async def test_agent_realtime_session_tool_call_limit_raises() -> None:
    agent: Agent[None, str] = Agent()

    # Never runs: the limit trips first.
    @agent.tool_plain
    def greet() -> str:  # pragma: no cover
        return 'hi'

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t1', tool_name='greet', args='{}'), ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model, usage_limits=UsageLimits(tool_calls_limit=0)).session() as session:
        with pytest.raises(UsageLimitExceeded, match='exceed the tool_calls_limit of 0'):
            _ = [e async for e in session]


async def test_agent_realtime_session_usage_limits_within_budget() -> None:
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def greet() -> str:
        return 'hi'

    conn = FakeRealtimeConnection(
        [
            SessionUsageEvent(usage=RequestUsage(input_tokens=1, output_tokens=1)),
            ToolCall(tool_call_id='t1', tool_name='greet', args='{}'),
            ResponseCompleteEvent(),
        ]
    )
    model = FakeRealtimeModel(conn)
    limits = UsageLimits(total_tokens_limit=1000, tool_calls_limit=10)
    async with agent.realtime(model, usage_limits=limits).session() as session:
        events = [e async for e in session]
    assert not any(isinstance(e, SessionErrorEvent) for e in events)
    assert any(isinstance(e, FunctionToolResultEvent) for e in events)


async def test_agent_realtime_session_native_tools_from_capability() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model, capabilities=[NativeTool(WebSearchTool())]).session() as session:
        _ = [e async for e in session]
    assert model.last_native_tools is not None
    assert any(isinstance(t, WebSearchTool) for t in model.last_native_tools)


async def test_agent_realtime_session_rejects_unsupported_native_tool() -> None:
    # A native tool the model doesn't support (per its `supported_native_tools` profile) with no local
    # fallback fails up front, before connecting — even when contributed by a capability. This runs the
    # same native ↔ local-tool swap the classic agent-run path applies, so the error points at `local=`.
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([])
    model = FakeRealtimeModel(conn, profile=_profile(supported_native_tools=frozenset()))
    with pytest.raises(
        UserError,
        match=r"not supported by this model.*WebSearch\(local='duckduckgo'\)",
    ):
        async with agent.realtime(model, capabilities=[NativeTool(WebSearchTool())]).session():
            pass  # pragma: no cover


async def test_agent_realtime_session_local_capability_tool_declared() -> None:
    def fetch(url: str) -> str:
        # Not executed in this wiring test.
        return f'content of {url}'  # pragma: no cover

    agent: Agent[None, str] = Agent(capabilities=[WebFetch(native=False, local=fetch)])
    conn = FakeRealtimeConnection([ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
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

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t1', tool_name='greet', args='{}'), ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    cap = _HookCapability()
    async with agent.realtime(model, capabilities=[cap]).session() as session:
        events = [e async for e in session]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert result.part.content == '[hooked] hi'
    assert cap.events == ['before:greet', 'after:hi']


async def test_agent_realtime_session_metadata_and_conversation_id() -> None:
    agent: Agent[None, str] = Agent()

    @agent.tool
    def whoami(ctx: RunContext) -> str:
        return f'{ctx.conversation_id}|{ctx.metadata}'

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t1', tool_name='whoami', args='{}'), ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model, conversation_id='conv-1', metadata={'tier': 'gold'}).session() as session:
        events = [e async for e in session]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert 'conv-1' in str(result.part.content)
    assert 'gold' in str(result.part.content)


async def test_session_stamps_conversation_id_and_classic_resume_resolves_it() -> None:
    seeded = [ModelRequest(parts=[UserPromptPart(content='seed')])]
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='spoken', is_final=True),
            ToolCall(tool_call_id='t1', tool_name='f', args='{}'),
            OutputTranscript(text='answer', is_final=True),
            ResponseCompleteEvent(),
        ]
    )

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'done'

    session = RealtimeSession(
        conn,
        runner,
        model_name='m',
        conversation_id='c1',
        message_history=seeded,
    )
    await session.send('typed')
    _ = await collect_events(session)

    assert session.all_messages()[0].conversation_id is None
    assert all(message.conversation_id == 'c1' for message in session.new_messages())
    assert resolve_conversation_id(None, session.all_messages()) == 'c1'


async def test_agent_realtime_session_native_tools_override_honored() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    with agent.override(native_tools=[WebSearchTool()]):
        async with agent.realtime(model).session() as session:
            _ = [e async for e in session]
    assert model.last_native_tools is not None
    assert any(isinstance(t, WebSearchTool) for t in model.last_native_tools)


async def test_wrapper_agent_realtime_session_proxies() -> None:
    from pydantic_ai.agent import WrapperAgent

    inner: Agent[None, str] = Agent(instructions='Inner')
    wrapper = WrapperAgent(inner)
    conn = FakeRealtimeConnection([])
    model = FakeRealtimeModel(conn)
    images = [BinaryImage(data=bytes([index]), media_type='image/png') for index in range(3)]
    # The wrapped agent's session is used, and per-session options like `retain_images_every_n` forward
    # through the wrapper (and the durable-exec subclasses that extend it) rather than being dropped.
    async with wrapper.realtime(model).session(retain_images_every_n=2) as session:
        for image in images:
            await session.send(image)
        retained = session.new_messages()
    assert model.last_instructions == 'Inner'  # the wrapped agent's session was used
    assert retained == [
        ModelRequest(parts=[UserPromptPart(content=[images[0]], timestamp=IsDatetime())]),
        ModelRequest(parts=[UserPromptPart(content=[images[2]], timestamp=IsDatetime())]),
    ]


async def test_agent_realtime_session_drops_auto_injected_tool_search() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([ResponseCompleteEvent()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        _ = [e async for e in session]
    assert model.last_native_tools == []
