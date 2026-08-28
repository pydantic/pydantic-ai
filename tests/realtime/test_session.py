"""Tests for `RealtimeSession`: event translation, history assembly, tool dispatch, and `Agent.realtime_session`."""

from __future__ import annotations as _annotations

import asyncio
import io
import wave
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from decimal import Decimal
from threading import Event as ThreadEvent
from typing import Any, Literal, TypeVar, cast

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
    RunCancelled,
    ToolFailed,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import (
    BinaryAudio,
    BinaryContent,
    BinaryImage,
    DeferredToolRequestsEvent,
    DeferredToolResultsEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    InstructionPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RealtimeSessionErrorEvent,
    RetryPromptPart,
    SpeechPart,
    SpeechPartDelta,
    SystemPromptPart,
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
    RealtimeError,
    RealtimeEvent,
    RealtimeInputSpeechEndEvent,
    RealtimeInputSpeechStartEvent,
    RealtimeInputTranscriptionErrorEvent,
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
    RealtimeSession as _RealtimeSession,
    RealtimeSessionReconnectEvent,
    RealtimeTurnCompleteEvent,
    TranscriptUpdate,
)
from pydantic_ai.realtime._session import _pending_message_text  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.realtime._utils import resolve_advertised_tools, seed_pcm_audio, seed_speech_content
from pydantic_ai.realtime.codec import (
    AudioDelta,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    ConversationCreated,
    ConversationItemCreated,
    CreateResponse,
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    ResponseDone,
    SessionUsage,
    ToolCall,
    ToolCallCancelled,
    ToolResult,
    TruncateOutput,
)
from pydantic_ai.settings import ModelSettings, ToolOrOutput
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ExternalToolset, FunctionToolset
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


def test_seed_pcm_audio_rejects_truncated_wav() -> None:
    audio = _wav_content(b'\x00\x01')
    truncated = BinaryContent(data=audio.data[:-2], media_type='audio/wav')

    with pytest.raises(UserError, match='not valid WAV audio'):
        seed_pcm_audio(audio=truncated, provider_name='test', sample_rate=24000)


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
    if any(name in kwargs for name in ('model_name', 'provider_name', 'provider_url')):
        kwargs['model'] = FakeRealtimeModel(
            connection,
            model_name=kwargs.pop('model_name', None),
            system=kwargs.pop('provider_name', None),
            base_url=kwargs.pop('provider_url', None),
        )
    return _RealtimeSession(connection, tool_manager=make_tool_manager(runner), **kwargs)


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
    supports_text_output: bool = True,
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
        supports_text_output=supports_text_output,
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
        reconnect_restores_in_flight_state: bool = True,
    ) -> None:
        self._events = events
        self._release = release
        self._input_transcription_enabled = input_transcription_enabled
        self._model_name = model_name
        self._reconnect_restores_in_flight_state = reconnect_restores_in_flight_state
        self.sent: list[RealtimeInput] = []

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def input_transcription_enabled(self) -> bool:
        return self._input_transcription_enabled

    @property
    def reconnect_restores_in_flight_state(self) -> bool:
        return self._reconnect_restores_in_flight_state

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
        model_name: str | None = 'fake-realtime',
        system: str | None = 'fake',
        base_url: str | None = None,
    ) -> None:
        self._connection = connection
        self.settings = settings
        self._fixed_profile = profile or _profile()
        self._model_name = model_name
        self._system = system
        self._base_url = base_url
        self.last_instructions: str | None = None
        self.last_tools: list[ToolDefinition] | None = None
        self.last_native_tools: list[AbstractNativeTool] | None = None
        self.last_model_settings: RealtimeModelSettings | None = None
        self.last_messages: Sequence[ModelMessage] | None = None

    @property
    def model_name(self) -> str:
        return cast('str', self._model_name)

    @property
    def system(self) -> str:
        return cast('str', self._system)

    @property
    def base_url(self) -> str | None:
        return self._base_url

    @property
    def profile(self) -> RealtimeModelProfile:
        # A fixed, already-resolved profile: these tests pin session behavior per flag, not the
        # default/provider/user layering the base class does (covered in the provider tests).
        return self._fixed_profile

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
            ResponseDone(),
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
            RealtimeTurnCompleteEvent(),
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
            ResponseDone(),
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


async def test_cumulative_transcript_extending_across_padding_is_still_an_append() -> None:
    """A snapshot that only extends its predecessor once padding is ignored is an append, not a revision.

    Providers pad transcripts unevenly (`' Hello'` then `'Hello there'`), and reporting that as a
    revision would make a caption redraw the whole turn for text the user already saw.
    """
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text=' Hello', cumulative=True),
            InputTranscript(text='Hello there', cumulative=True),
            ResponseDone(),
        ]
    )
    session = RealtimeSession(conn)

    async with session:
        deltas = (await asyncio.gather(drain_events(session), aiter_to_list(session.stream_transcripts(delta=True))))[1]

    assert deltas == [
        TranscriptUpdate(index=0, speaker='user', delta=' Hello', transcript=' Hello'),
        TranscriptUpdate(index=0, speaker='user', delta=' there', transcript='Hello there'),
    ]


async def test_cumulative_transcript_repeating_itself_emits_nothing() -> None:
    """An unchanged snapshot is not news; re-emitting it would flicker a caption for no reason."""
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='Hello', cumulative=True),
            InputTranscript(text='Hello', cumulative=True),
            ResponseDone(),
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
                ResponseDone(),
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
    assert [type(sent).__name__ for sent in conn.sent] == ['str', 'ToolResult']


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
        # The sample-rate properties are the supported read surface: they resolve the profile keys and
        # supply the 24 kHz default for keys a profile omits, so callers never index the profile dict.
        assert session.audio_input_sample_rate == 16000
        assert session.audio_output_sample_rate == 24000
    # The same properties exist on the model, for configuring capture/playback before a session exists.
    assert model.audio_input_sample_rate == 16000
    assert model.audio_output_sample_rate == 24000


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
            ResponseDone(),
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
            RealtimeTurnCompleteEvent(),
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
    conn = FakeRealtimeConnection([OutputTranscript(text='Hello world', is_final=True), ResponseDone()])
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
            RealtimeTurnCompleteEvent(),
        ]
    )


async def test_multiple_assistant_items_fold_into_one_response() -> None:
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='first', is_final=True, item_id='item-1'),
            OutputTranscript(text='second', is_final=True, item_id='item-2', output_text=True),
            ResponseDone(provider_response_id='response-1', finish_reason='stop'),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, provider_name='openai')

    _ = await collect_events(session)

    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[
                    SpeechPart(speaker='assistant', transcript='first'),
                    TextPart(content='second'),
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
            ResponseDone(
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
            ResponseDone(
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
    session = RealtimeSession(FakeRealtimeConnection([ResponseDone()]), _noop_runner)

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
        [ModelRequest(parts=[SpeechPart(speaker='user', transcript='what is the weather')], timestamp=IsDatetime())]
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
        SpeechPart(speaker='user', transcript='first turn'),
        SpeechPart(speaker='user', transcript='second turn'),
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
            parts=[SpeechPart(speaker='user', transcript='first partial')],
            timestamp=IsDatetime(),
        ),
        ModelRequest(
            parts=[SpeechPart(speaker='user', transcript='second final')],
            timestamp=IsDatetime(),
        ),
    ]


async def test_partial_only_user_transcript_finalized_on_turn_complete() -> None:
    # Gemini streams only partial input transcripts (never `is_final`) and no `RealtimeInputSpeechEndEvent`, so
    # the user turn is finalized at the turn boundary. Without that, the transcribed user turn is
    # dropped from history entirely (only the assistant response would remain).
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='what is ', is_final=False),
            InputTranscript(text='the weather', is_final=False),
            ResponseDone(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner)
    _ = await collect_events(session)
    assert session.new_messages() == snapshot(
        [ModelRequest(parts=[SpeechPart(speaker='user', transcript='what is the weather')], timestamp=IsDatetime())]
    )


async def test_partial_only_user_transcript_strips_leading_space() -> None:
    # Gemini streams partial-only transcripts whose first delta carries a leading space; with no final
    # snapshot to reconcile against (unlike OpenAI's `.completed`), the finalized turn would keep the
    # space. Finalization strips it so the result matches the OpenAI transcription of the same utterance.
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text=' Hello, my name', is_final=False),
            InputTranscript(text=' is Marcelo.', is_final=False),
            ResponseDone(),
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
        [
            ModelRequest(
                parts=[SpeechPart(speaker='user', transcript='Hello, my name is Marcelo.')], timestamp=IsDatetime()
            )
        ]
    )


async def test_audio_delta_streams_and_transcript_pairs() -> None:
    conn = FakeRealtimeConnection(
        [AudioDelta(data=b'\x00\x01'), OutputTranscript(text='hi', is_final=True), ResponseDone()]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = await collect_events(session)
    assert [type(e).__name__ for e in events] == snapshot(
        [
            'PartStartEvent',
            'PartDeltaEvent',
            'PartDeltaEvent',
            'PartEndEvent',
            'RealtimeTurnCompleteEvent',
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
    conn = FakeRealtimeConnection([ResponseDone(interrupted=True), RealtimeSessionErrorEvent(message='oops')])
    session = RealtimeSession(conn, _noop_runner)
    events = await collect_events(session)
    # A recoverable error is mid-stream: the session keeps running and surfaces the event to the
    # consumer (rather than swallowing it) so a quiet failure is observable.
    assert events == [
        RealtimeTurnCompleteEvent(),
        RealtimeSessionErrorEvent(message='oops'),
    ]


async def test_input_transcription_failure_passes_through_and_session_continues() -> None:
    # Failures finalize placeholder turns whether or not they identify their turn (`item_id` may be absent).
    identified = RealtimeInputTranscriptionErrorEvent(message='audio unintelligible', item_id='user-1', content_index=0)
    anonymous = RealtimeInputTranscriptionErrorEvent(message='transcription unavailable')
    conn = FakeRealtimeConnection([identified, anonymous, ResponseDone()])
    session = RealtimeSession(conn, _noop_runner)

    events = await collect_events(session)
    assert [type(event).__name__ for event in events] == [
        'PartStartEvent',
        'PartEndEvent',
        'RealtimeInputTranscriptionErrorEvent',
        'PartStartEvent',
        'PartEndEvent',
        'RealtimeInputTranscriptionErrorEvent',
        'RealtimeTurnCompleteEvent',
    ]
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user')], timestamp=IsDatetime()),
            ModelRequest(parts=[SpeechPart(speaker='user')], timestamp=IsDatetime()),
        ]
    )


async def test_input_transcription_failure_after_partial_does_not_block_later_turns() -> None:
    # Item A streams a partial transcript, item B finalizes out of order, then A's transcription fails.
    # A's placeholder must unblock the ordered prefix so both turns reach history in provider order.
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='partial A', is_final=False, item_id='A'),
            InputTranscript(text='hello from B', is_final=True, item_id='B'),
            RealtimeInputTranscriptionErrorEvent(message='transcription failed', item_id='A'),
            ResponseDone(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner)
    _ = await collect_events(session)
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user')], timestamp=IsDatetime()),
            ModelRequest(parts=[SpeechPart(speaker='user', transcript='hello from B')], timestamp=IsDatetime()),
        ]
    )


async def test_input_transcription_failure_ends_the_part_it_opened() -> None:
    """A failed transcription closes the failed turn's own part index and keeps overlapping turns active.

    Regression: `_finalize_failed_user_item` hardcoded `PartEndEvent(index=0, ...)` — with any earlier
    part in the session, a live caption UI saw the failed turn left open and part 0 closed again — and
    cleared `_user_turn_active` outright, marking the whole user side idle while another overlapping
    item was still streaming.
    """
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='hello from A', is_final=True, item_id='A'),  # claims part index 0
            InputTranscript(text='partial B', is_final=False, item_id='B'),  # opens part index 1
            InputTranscript(text='partial C', is_final=False, item_id='C'),  # opens part index 2, overlapping
            RealtimeInputTranscriptionErrorEvent(message='failed', item_id='B'),
            ResponseDone(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner)

    events: list[RealtimeEvent] = []
    failed_index: int | None = None
    async with session:
        async for event in session:  # pragma: no branch
            events.append(event)
            if (
                isinstance(event, PartDeltaEvent)
                and isinstance(event.delta, SpeechPartDelta)
                and event.delta.transcript == 'partial B'
            ):
                failed_index = event.index
            if isinstance(event, PartEndEvent) and event.index == failed_index:
                # C is still streaming: B's failure must not mark the whole user side idle.
                assert session._user_turn_active  # pyright: ignore[reportPrivateUsage]

    assert failed_index is not None and failed_index != 0
    assert any(isinstance(event, PartStartEvent) and event.index == failed_index for event in events)
    assert any(isinstance(event, PartEndEvent) and event.index == failed_index for event in events)


async def test_input_transcription_failure_ignores_already_closed_items() -> None:
    """A duplicate or late transcription-error event for a closed item must not re-open it.

    Regression: `_finalize_failed_user_item` lacked `_handle_input_transcript`'s closed-item guard, so a
    stray error for an already finalized (or already failed) item minted a fresh blank `SpeechPart` with
    a new part index and recorded a second user turn in history.
    """
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='hello from A', is_final=True, item_id='A'),
            RealtimeInputTranscriptionErrorEvent(message='late error for finalized item', item_id='A'),
            InputTranscript(text='partial B', is_final=False, item_id='B'),
            RealtimeInputTranscriptionErrorEvent(message='failed', item_id='B'),
            RealtimeInputTranscriptionErrorEvent(message='duplicate failure', item_id='B'),
            ResponseDone(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner)

    events = await collect_events(session)

    ended = [e.part for e in events if isinstance(e, PartEndEvent) and isinstance(e.part, SpeechPart)]
    assert ended == [SpeechPart(speaker='user', transcript='hello from A'), SpeechPart(speaker='user')]
    user_parts = [
        part for message in session.new_messages() if isinstance(message, ModelRequest) for part in message.parts
    ]
    assert user_parts == [
        SpeechPart(speaker='user', transcript='hello from A'),
        SpeechPart(speaker='user'),
    ]


@pytest.mark.parametrize('item_id', [None, 'user-1'])
async def test_input_transcription_failure_retained_audio_fallback(item_id: str | None) -> None:
    conn = FakeRealtimeConnection([RealtimeInputTranscriptionErrorEvent(message='failed', item_id=item_id)])
    session = RealtimeSession(conn, _noop_runner, audio_retention='input_audio')
    if item_id is None:
        await session.send_audio(b'\xaa')

    _ = await collect_events(session)

    expected_audio = _wav_content(b'\xaa') if item_id is None else None
    assert session.new_messages() == [
        ModelRequest(parts=[SpeechPart(speaker='user', audio=expected_audio)], timestamp=IsDatetime())
    ]


async def test_fatal_session_error_raises() -> None:
    conn = FakeRealtimeConnection([RealtimeSessionErrorEvent(message='provider failed', recoverable=False)])
    session = RealtimeSession(conn, _noop_runner)
    with pytest.raises(RealtimeError, match='provider failed'):
        _ = await collect_events(session)


async def test_interrupted_turn_keeps_partial_transcript() -> None:
    # A barge-in cancels the turn; the completed part reflects the partial transcript seen so far.
    conn = FakeRealtimeConnection(
        [OutputTranscript(text='the answer is ', is_final=False), ResponseDone(interrupted=True)]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = await collect_events(session)
    # A barge-in ends the exchange too: nothing more is coming, so the turn completes with it.
    assert events[-1] == RealtimeTurnCompleteEvent()
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
            ResponseDone(interrupted=True),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')

    await session.interrupt(played_ms=640)
    _ = await collect_events(session)

    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[
                    SpeechPart(speaker='assistant', transcript='first'),
                    SpeechPart(speaker='assistant', transcript='second', interrupted_at_ms=640),
                ],
                model_name='m',
                timestamp=IsDatetime(),
                state='interrupted',
            )
        ]
    )


async def test_deltas_still_in_flight_when_a_cancel_lands_stay_in_the_interrupted_turn() -> None:
    """Audio already on the wire when `interrupt()` is sent belongs to the turn it was generated for.

    Cancelling is a round trip, so a provider keeps emitting for the few milliseconds it takes to
    arrive. Those deltas must land in the response being cancelled — starting a second response for
    them would show the user a turn the model never took.
    """
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='I was saying', is_final=False, item_id='item-1'),
            OutputTranscript(text=' something', is_final=True, item_id='item-1'),
            ResponseDone(interrupted=True),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')

    await session.interrupt(played_ms=120)
    _ = await collect_events(session)

    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[
                    SpeechPart(
                        speaker='assistant',
                        transcript='I was saying something',
                        interrupted_at_ms=120,
                    )
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
            ResponseDone(interrupted=True),
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


async def test_speech_parts_do_not_persist_provider_item_ids() -> None:
    openai = RealtimeSession(
        FakeRealtimeConnection([OutputTranscript(text='hello', is_final=True, item_id='item-a'), ResponseDone()]),
        _noop_runner,
        provider_name='openai',
    )
    gemini = RealtimeSession(
        FakeRealtimeConnection([OutputTranscript(text='hello', is_final=True), ResponseDone()]),
        _noop_runner,
        provider_name='google',
    )
    _ = await collect_events(openai)
    _ = await collect_events(gemini)

    openai_part = openai.new_messages()[0].parts[0]
    gemini_part = gemini.new_messages()[0].parts[0]
    assert isinstance(openai_part, SpeechPart) and isinstance(gemini_part, SpeechPart)
    assert (openai_part.id, openai_part.provider_name) == (None, None)
    assert (gemini_part.id, gemini_part.provider_name) == (None, None)


# --- tool calls: history + events --------------------------------------------------------------


async def test_turn_completes_once_the_tool_round_is_over_not_before() -> None:
    """`RealtimeTurnCompleteEvent` waits for the response *after* the tool, which is the whole point of it.

    The provider's own turn boundary arrives once per response, so a tool round produces several and
    which one is last differs by model — `gpt-realtime-2.1` and xAI Grok Voice speak *before* calling the
    tool, so stopping on their first boundary truncates the exchange. This connection answers the tool
    result the way a provider does, so the ordering is the real one rather than the fixture's.
    """
    tool_answered = asyncio.Event()

    class _AnswersTheTool(FakeRealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            # The tool result is the only thing the session sends in this scenario.
            if isinstance(content, ToolResult):  # pragma: no branch
                tool_answered.set()
            self.sent.append(content)

        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            yield ToolCall(tool_call_id='tc_1', tool_name='get_weather', args='{"city": "Paris"}')
            yield ResponseDone()  # the tool-call response; the exchange is not over
            await tool_answered.wait()
            yield OutputTranscript(text='It is sunny in Paris', is_final=True)
            yield ResponseDone()  # the answer; now it is

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'Sunny, 22C'

    session = RealtimeSession(_AnswersTheTool([]), runner, model_name='m')
    events = await collect_events(session)

    assert [type(event).__name__ for event in events] == snapshot(
        [
            'PartStartEvent',
            'PartEndEvent',
            'FunctionToolCallEvent',
            'FunctionToolResultEvent',
            'PartStartEvent',
            'PartDeltaEvent',
            'PartEndEvent',
            'RealtimeTurnCompleteEvent',
        ]
    )
    # Exactly one, and it lands after the model has actually answered.
    assert [i for i, e in enumerate(events) if isinstance(e, RealtimeTurnCompleteEvent)] == [len(events) - 1]


async def test_tool_call_round_builds_classic_history() -> None:
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text="what's the weather in Paris", is_final=True),
            ToolCall(tool_call_id='tc_1', tool_name='get_weather', args='{"city": "Paris"}'),
            OutputTranscript(text="It's sunny in Paris", is_final=True),
            ResponseDone(),
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
            'PartEndEvent',
            'FunctionToolResultEvent',
        ]
    )

    assert conn.sent == [ToolResult(tool_call_id='tc_1', output='Sunny, 22C')]
    # History mirrors a classic tool-call round: user request, tool-call response, tool result, answer.
    assert session.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[SpeechPart(speaker='user', transcript="what's the weather in Paris")], timestamp=IsDatetime()
            ),
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
                ],
                timestamp=IsDatetime(),
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
            RealtimeInputSpeechStartEvent(),
            InputTranscript(text="what's the weather in Paris", item_id='item-1'),
            ToolCall(tool_call_id='tc_1', tool_name='get_weather', args='{"city": "Paris"}'),
            InputTranscript(text="what's the weather in Paris", is_final=True, item_id='item-1'),
            OutputTranscript(text="It's sunny in Paris", is_final=True),
            ResponseDone(),
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
            ResponseDone(),
            RealtimeInputSpeechStartEvent(),
            InputTranscript(text='and tomorrow', item_id='item-2'),
            ToolCall(tool_call_id='tc_1', tool_name='get_weather', args='{"city": "Paris"}'),
            InputTranscript(text='and tomorrow', is_final=True, item_id='item-2'),
            OutputTranscript(text='Rain.', is_final=True),
            ResponseDone(),
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

    Gemini Live sends neither `RealtimeInputSpeechStartEvent` nor a final input transcript, so the turn is only
    finalized at `ResponseDone` — by which time its tool round is long recorded. Audio starting is
    then the only signal that a user turn began, so that is where its place in history comes from.
    """
    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='tc_1', tool_name='get_weather', args='{"city": "Paris"}'),
            InputTranscript(text="what's the weather in Paris"),
            OutputTranscript(text="It's sunny in Paris", is_final=True),
            ResponseDone(),
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
            SessionUsage(
                usage=RequestUsage(output_tokens=1),
                provider_response_id='response-tool',
                finish_reason='tool_call',
            ),
            ResponseDone(
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
    assert 'Invalid JSON' in str(result.part.content)
    assert isinstance(conn.sent[0], ToolResult)
    assert conn.sent[0].output == result.part.model_response()


async def test_non_object_json_args_reported() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='noop', args='[1, 2]')])
    session = RealtimeSession(conn, _noop_runner)
    events = await collect_events(session)
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, RetryPromptPart)
    assert 'Input should be an object' in str(result.part.content)


class _RepeatedMalformedToolArgsConnection(FakeRealtimeConnection):
    """Start a new response after the first malformed call is returned for retry."""

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        yield ToolCall(tool_call_id='tc1', tool_name='noop', args='not json')
        while not self.sent:
            await asyncio.sleep(0)
        yield ResponseDone()
        yield ToolCall(tool_call_id='tc2', tool_name='noop', args='still not json')


async def test_repeated_malformed_json_args_exceed_tool_retry_budget() -> None:
    session = RealtimeSession(_RepeatedMalformedToolArgsConnection([]), _noop_runner)

    with pytest.raises(UnexpectedModelBehavior, match="Tool 'noop' exceeded max retries count of 1"):
        await collect_events(session)

    assert session._tool_manager.failed_tools == set()  # pyright: ignore[reportPrivateUsage]
    assert session._tool_manager.ctx is not None  # pyright: ignore[reportPrivateUsage]
    assert session._tool_manager.ctx.retries == {'noop': 1}  # pyright: ignore[reportPrivateUsage]


async def test_tool_runner_exception_ends_session() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='boom', args='{}')])

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        raise RuntimeError('kaboom')

    session = RealtimeSession(conn, runner)
    with pytest.raises(RuntimeError, match='kaboom'):
        await collect_events(session)
    assert conn.sent == []


async def test_tool_runner_base_exception_ends_session() -> None:
    class ToolFailure(BaseException):
        pass

    conn = BlockingRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='boom', args='{}')])

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        raise ToolFailure('kaboom')

    session = RealtimeSession(conn, runner)
    with pytest.raises(ToolFailure, match='kaboom'):
        await collect_events(session)
    assert conn.sent == []


async def test_tool_runner_base_exception_wins_connection_completion_race() -> None:
    class ToolFailure(BaseException):
        pass

    connection_finished = asyncio.Event()

    class _ConnectionEndsWithTool(FakeRealtimeConnection):
        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            try:
                yield ToolCall(tool_call_id='tc', tool_name='boom', args='{}')
            finally:
                connection_finished.set()

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        raise ToolFailure('kaboom')

    conn = _ConnectionEndsWithTool([])
    session = RealtimeSession(conn, runner)
    with pytest.raises(ToolFailure, match='kaboom'):
        await collect_events(session)
    assert connection_finished.is_set()
    assert conn.sent == []


async def test_tool_runner_cancelled_call_ends_cleanly() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class _CancelAfterToolStarts(FakeRealtimeConnection):
        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            yield ToolCall(tool_call_id='tc', tool_name='slow', args='{}')
            await started.wait()
            yield ToolCallCancelled(tool_call_ids=['tc'])

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return 'never'  # pragma: no cover

    conn = _CancelAfterToolStarts([])
    events = await collect_events(RealtimeSession(conn, runner))

    assert cancelled.is_set()
    assert conn.sent == []
    results = [event for event in events if isinstance(event, FunctionToolResultEvent)]
    assert len(results) == 1 and isinstance(results[0].part, ToolReturnPart)
    assert results[0].part.content == 'The tool call was interrupted before a result was produced.'


async def test_validation_hook_exception_reports_failed_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = make_tool_manager()
    outcomes: list[bool] = []

    async def fail_validation(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('validation hook failed')

    async def record_validation(valid: bool) -> None:
        outcomes.append(valid)

    monkeypatch.setattr(manager, 'validate_tool_call', fail_validation)
    with pytest.raises(RuntimeError, match='validation hook failed'):
        await manager.handle_call(
            ToolCallPart(tool_name='noop', args={}, tool_call_id='call'),
            on_validate=record_validation,
        )
    assert outcomes == [False]


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
        [OutputTranscript(text='hi', is_final=True), ResponseDone()], model_name='grok-voice-latest'
    )
    session = RealtimeSession(conn, _noop_runner, model_name='grok-voice-4-turbo')
    _ = await collect_events(session)
    response = next(m for m in session.all_messages() if isinstance(m, ModelResponse))
    assert response.model_name == 'grok-voice-latest'


async def test_agent_realtime_session_threads_provider_name() -> None:
    agent: Agent[object, str] = Agent()
    model = FakeRealtimeModel(FakeRealtimeConnection([OutputTranscript(text='hi', is_final=True), ResponseDone()]))
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
            ResponseDone(),
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
            ResponseDone(),
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
            ResponseDone(),
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


def test_parallel_tool_returns_are_inserted_in_call_order() -> None:
    session = RealtimeSession(FakeRealtimeConnection([]))
    call_one = ToolCallPart(tool_name='one', args={}, tool_call_id='one')
    call_two = ToolCallPart(tool_name='two', args={}, tool_call_id='two')
    response = ModelResponse(parts=[call_one, call_two])
    first_return = ModelRequest(parts=[ToolReturnPart(tool_name='one', content='1', tool_call_id='one')])
    second_return = ModelRequest(parts=[ToolReturnPart(tool_name='two', content='2', tool_call_id='two')])
    session._history.append(response)  # pyright: ignore[reportPrivateUsage]

    session._insert_tool_return(call_two, second_return)  # pyright: ignore[reportPrivateUsage]
    session._insert_tool_return(call_one, first_return)  # pyright: ignore[reportPrivateUsage]

    assert session.new_messages() == [response, first_return, second_return]


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


async def test_tool_error_surfaces_at_close_when_the_stream_was_never_iterated() -> None:
    """A failed tool reaches the consumer through the queue, which has no reader here.

    Like the pump-error case above, `close()` is the only place left for it to surface: without this
    a caller using only `send()` and the audio/transcript views would exit cleanly from a tool that
    actually crashed.
    """

    async def runner(*args: Any) -> str:
        raise RuntimeError('tool exploded')

    session = RealtimeSession(
        FakeRealtimeConnection([ToolCall(tool_call_id='tc1', tool_name='noop', args='{}')]),
        runner=runner,
    )
    with pytest.raises(RuntimeError, match='tool exploded'):
        async with session:
            # Drive the pump through an audio view alone, like the quickstart's playback task.
            assert [chunk async for chunk in session.stream_audio()] == []
            await asyncio.sleep(0.05)  # let the failed tool's error reach the queue


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
        ('c1', 'The tool call was interrupted before a result was produced.')
    ]


async def test_sequential_tool_is_execution_barrier() -> None:
    before_started = asyncio.Event()
    barrier_started = asyncio.Event()
    before_release = asyncio.Event()
    barrier_release = asyncio.Event()
    order: list[str] = []
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    async def before() -> str:
        order.append('before start')
        before_started.set()
        await before_release.wait()
        order.append('before finish')
        return 'before'

    @agent.tool_plain(sequential=True)
    async def barrier() -> str:
        order.append('barrier start')
        barrier_started.set()
        await barrier_release.wait()
        order.append('barrier finish')
        return 'barrier'

    @agent.tool_plain
    async def after() -> str:
        order.append('after start')
        order.append('after finish')
        return 'after'

    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='before', tool_name='before', args='{}'),
            ToolCall(tool_call_id='barrier', tool_name='barrier', args='{}'),
            ToolCall(tool_call_id='after', tool_name='after', args='{}'),
        ]
    )
    async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
        events_task = asyncio.create_task(drain_events(session))
        await before_started.wait()
        assert order == ['before start']
        before_release.set()
        await barrier_started.wait()
        assert order == ['before start', 'before finish', 'barrier start']
        barrier_release.set()
        await events_task

    assert order == [
        'before start',
        'before finish',
        'barrier start',
        'barrier finish',
        'after start',
        'after finish',
    ]


async def test_parallel_execution_mode_sequential_serializes_realtime_tools() -> None:
    release = [asyncio.Event(), asyncio.Event(), asyncio.Event()]
    started = [asyncio.Event(), asyncio.Event(), asyncio.Event()]
    order: list[str] = []
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    async def ordered(index: int) -> int:
        order.append(f'{index} start')
        started[index].set()
        await release[index].wait()
        order.append(f'{index} finish')
        return index

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id=str(i), tool_name='ordered', args=f'{{"index": {i}}}') for i in range(3)]
    )
    with ToolManager.parallel_execution_mode('sequential'):
        async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
            events_task = asyncio.create_task(drain_events(session))
            for i in range(3):
                await started[i].wait()
                assert order == [item for j in range(i) for item in (f'{j} start', f'{j} finish')] + [f'{i} start']
                release[i].set()
            await events_task

    assert order == [item for i in range(3) for item in (f'{i} start', f'{i} finish')]


async def test_cancelled_realtime_barrier_releases_following_tool() -> None:
    barrier_started = asyncio.Event()
    after_started = asyncio.Event()
    barrier_cancelled = asyncio.Event()
    agent: Agent[None, str] = Agent()

    @agent.tool_plain(sequential=True)
    async def barrier() -> str:
        barrier_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            barrier_cancelled.set()
            raise
        return 'never'  # pragma: no cover

    @agent.tool_plain
    async def after() -> str:
        after_started.set()
        return 'after'

    class _CancelBarrier(FakeRealtimeConnection):
        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            yield ToolCall(tool_call_id='barrier', tool_name='barrier', args='{}')
            yield ToolCall(tool_call_id='after', tool_name='after', args='{}')
            await barrier_started.wait()
            yield ToolCallCancelled(tool_call_ids=['barrier'])

    conn = _CancelBarrier([])
    async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
        await drain_events(session)

    assert barrier_cancelled.is_set()
    assert after_started.is_set()


async def test_realtime_barrier_exception_releases_following_tool() -> None:
    after_started = asyncio.Event()
    agent: Agent[None, str] = Agent()

    @agent.tool_plain(sequential=True)
    async def barrier() -> str:
        raise RuntimeError('barrier failed')

    @agent.tool_plain
    async def after() -> str:
        after_started.set()
        return 'after'

    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='barrier', tool_name='barrier', args='{}'),
            ToolCall(tool_call_id='after', tool_name='after', args='{}'),
        ]
    )
    async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
        events = session.__aiter__()
        await anext(events)
        await asyncio.wait_for(after_started.wait(), timeout=1)
        with pytest.raises(RuntimeError, match='barrier failed'):
            await aiter_to_list(events)

    assert after_started.is_set()


async def test_realtime_tools_run_concurrently_by_default() -> None:
    both_started = asyncio.Event()
    started: set[str] = set()
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    async def parallel(name: str) -> str:
        started.add(name)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        return name

    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='one', tool_name='parallel', args='{"name": "one"}'),
            ToolCall(tool_call_id='two', tool_name='parallel', args='{"name": "two"}'),
        ]
    )
    async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
        await drain_events(session)

    assert started == {'one', 'two'}


async def test_tool_call_cancellation_unknown_id_is_ignored() -> None:
    # A cancellation for an id with no matching in-flight call (already finished, or never started) must
    # be a no-op: no crash, no spurious result event, nothing sent. Covers the race where a tool finishes
    # in the window before its cancellation arrives (the `finally`-pop makes that atomic).
    conn = FakeRealtimeConnection(
        [
            ToolCallCancelled(tool_call_ids=['never-started']),
            OutputTranscript(text='hi', is_final=True),
            ResponseDone(),
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
    await session.send(BinaryAudio(data=b'\x03', media_type='audio/pcm'))
    assert conn.sent == [
        BinaryAudio(data=b'\x01\x02', media_type='audio/pcm'),
        'hello',
        BinaryImage(data=b'\xff\xd8', media_type='image/jpeg'),
        BinaryAudio(data=b'\x03', media_type='audio/pcm'),
    ]


async def test_send_audio_accepts_async_iterable() -> None:
    """`send_audio` drains an async iterable chunk by chunk, so a capture loop can be one call.

    Unit test: the per-chunk forwarding of a caller-supplied iterator has no wire signature of its
    own — a cassette can't distinguish it from a caller looping over `send_audio(chunk)`.
    """
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)

    async def microphone() -> AsyncIterator[bytes]:
        yield b'\x01\x02'
        yield b'\x03\x04'

    await session.send_audio(microphone())

    assert conn.sent == [
        BinaryAudio(data=b'\x01\x02', media_type='audio/pcm'),
        BinaryAudio(data=b'\x03\x04', media_type='audio/pcm'),
    ]


async def test_send_dispatches_through_bookkeeping_helpers() -> None:
    # `send(<str>)` must route through the text-turn path, so the user turn lands in history rather
    # than bypassing it (the raw pass-through used to skip all session bookkeeping).
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)
    await session.send('hello')
    assert conn.sent == ['hello']
    assert session.new_messages() == snapshot(
        [ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsDatetime())], timestamp=IsDatetime())]
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
            'hello',
            BinaryImage(data=b'image', media_type='image/png'),
            BinaryAudio(data=b'\x01\x02\x03\x04', media_type='audio/pcm'),
            BinaryAudio(data=b'\xaa\xbb', media_type='audio/pcm'),
        ]
    )
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsDatetime())], timestamp=IsDatetime()),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[BinaryImage(data=b'image', media_type='image/png')],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_send_accepts_sequence() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner)

    await session.send(['look at this', BinaryImage(data=b'image', media_type='image/png')])

    assert conn.sent == ['look at this', BinaryImage(data=b'image', media_type='image/png')]


async def test_image_history_retention_samples_and_round_trips() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, retain_images_every_n=2)
    images = [BinaryImage(data=f'image-{index}'.encode(), media_type='image/png') for index in range(4)]

    for image in images:
        await session.send(image)

    assert conn.sent == [BinaryImage(data=image.data, media_type='image/png') for image in images]
    assert session.all_messages() == [
        ModelRequest(parts=[UserPromptPart(content=[images[0]], timestamp=IsDatetime())], timestamp=IsDatetime()),
        ModelRequest(parts=[UserPromptPart(content=[images[2]], timestamp=IsDatetime())], timestamp=IsDatetime()),
    ]
    serialized = ModelMessagesTypeAdapter.dump_json(session.all_messages())
    assert ModelMessagesTypeAdapter.validate_json(serialized) == session.all_messages()


async def test_failed_unretained_image_send_has_nothing_to_take_back() -> None:
    # With `retain_images_every_n=2` the second image is never recorded, so when its send fails
    # there is no phantom history to roll back; the failure still propagates.
    class _SecondImageSendFails(FakeRealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            if isinstance(content, BinaryImage) and self.sent:
                raise RuntimeError('send failed')
            await super().send(content)

    conn = _SecondImageSendFails([])
    session = RealtimeSession(conn, _noop_runner, retain_images_every_n=2)
    images = [
        BinaryImage(data=b'image-0', media_type='image/png'),
        BinaryImage(data=b'image-1', media_type='image/png'),
    ]

    await session.send(images[0])
    with pytest.raises(RuntimeError, match='send failed'):
        await session.send(images[1])

    assert session.all_messages() == [
        ModelRequest(parts=[UserPromptPart(content=[images[0]], timestamp=IsDatetime())], timestamp=IsDatetime()),
    ]


async def test_image_history_retention_must_be_positive() -> None:
    with pytest.raises(UserError, match='`retain_images_every_n` must be at least 1'):
        RealtimeSession(FakeRealtimeConnection([]), _noop_runner, retain_images_every_n=0)


async def test_image_history_cap_evicts_oldest() -> None:
    """`retain_images_max` keeps history a sliding window: sampling slows growth, the cap bounds it."""
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, retain_images_max=2)
    images = [BinaryImage(data=f'image-{index}'.encode(), media_type='image/png') for index in range(4)]

    for image in images:
        await session.send(image)

    # Every frame still reached the provider; only the local record is bounded.
    assert conn.sent == [BinaryImage(data=image.data, media_type='image/png') for image in images]
    assert session.all_messages() == [
        ModelRequest(parts=[UserPromptPart(content=[images[2]], timestamp=IsDatetime())], timestamp=IsDatetime()),
        ModelRequest(parts=[UserPromptPart(content=[images[3]], timestamp=IsDatetime())], timestamp=IsDatetime()),
    ]


async def test_image_history_cap_composes_with_sampling() -> None:
    """With `retain_images_every_n=2` only frames 0 and 2 are retained; a cap of 1 keeps the newest."""
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, retain_images_every_n=2, retain_images_max=1)
    images = [BinaryImage(data=f'image-{index}'.encode(), media_type='image/png') for index in range(4)]

    for image in images:
        await session.send(image)

    assert session.all_messages() == [
        ModelRequest(parts=[UserPromptPart(content=[images[2]], timestamp=IsDatetime())], timestamp=IsDatetime()),
    ]


async def test_image_history_cap_zero_retains_nothing() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, retain_images_max=0)
    image = BinaryImage(data=b'image-0', media_type='image/png')

    await session.send(image)

    assert conn.sent == [BinaryImage(data=image.data, media_type='image/png')]
    assert session.all_messages() == []


async def test_image_history_cap_must_be_non_negative() -> None:
    with pytest.raises(UserError, match='`retain_images_max` must be at least 0'):
        RealtimeSession(FakeRealtimeConnection([]), _noop_runner, retain_images_max=-1)


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
    # `send(BinaryImage(...))` must enforce the same `supports_image_input` guard as `send_image`.
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, profile=_profile(supports_image_input=False))
    with pytest.raises(UserError, match='does not support image input'):
        await session.send(BinaryImage(data=b'\xff', media_type='image/jpeg'))
    assert conn.sent == []


async def test_send_text_adds_user_prompt_to_history() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, conversation_id='c1')
    await session.send('turn it up')
    assert session.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='turn it up', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                conversation_id='c1',
            )
        ]
    )


async def test_send_during_response_is_recorded_after_response() -> None:
    response_started = asyncio.Event()
    continue_response = asyncio.Event()

    class MidResponseConnection(RealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            assert content == 'next turn'

        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            yield OutputTranscript(text='first ', item_id='assistant-1')
            response_started.set()
            await continue_response.wait()
            yield OutputTranscript(text='response', is_final=True, item_id='assistant-1')
            yield ResponseDone(provider_response_id='response-1', finish_reason='stop')

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
    assert isinstance(remaining_events[-1], RealtimeTurnCompleteEvent)
    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='first response')],
                provider_response_id='response-1',
                timestamp=IsDatetime(),
                finish_reason='stop',
            ),
            ModelRequest(parts=[UserPromptPart(content='next turn', timestamp=IsDatetime())], timestamp=IsDatetime()),
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
            session._translate_event(ResponseDone())  # pyright: ignore[reportPrivateUsage]

    conn = _FinishingSend([])
    session = RealtimeSession(conn)
    await session.send('question')
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='question', timestamp=IsDatetime())], timestamp=IsDatetime()),
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
            session._translate_event(RealtimeInputSpeechEndEvent())  # pyright: ignore[reportPrivateUsage]

    conn = _BoundarySend([], input_transcription_enabled=False)
    session = RealtimeSession(conn, audio_retention='input_audio')
    await session.send_audio(b'\xaa\xbb')
    assert session.new_messages() == snapshot(
        [ModelRequest(parts=[SpeechPart(speaker='user', audio=_wav_content(b'\xaa\xbb'))], timestamp=IsDatetime())]
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
        await session.send('typed question')
    with pytest.raises(RuntimeError, match='send failed'):
        await session.send_audio(b'\xaa\xbb')
    conn.fail = False
    await session.send_audio(b'\xcc')
    session._translate_event(InputTranscript(text='successful', is_final=True))  # pyright: ignore[reportPrivateUsage]
    assert session.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[SpeechPart(speaker='user', transcript='successful', audio=_wav_content(b'\xcc'))],
                timestamp=IsDatetime(),
            )
        ]
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


async def test_failed_image_send_is_not_recorded_in_history() -> None:
    # A retained image is recorded before the frame goes out, so history can't disagree with the wire
    # when sends interleave. When the send fails there is no image on the wire to agree with, so the
    # record has to come back out.
    class _BuggyConnection(FakeRealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            raise ValueError('bug in the connection')

    session = RealtimeSession(_BuggyConnection([]), profile=_profile(supports_image_input=True))
    with pytest.raises(ValueError, match='bug in the connection'):
        await session.send(BinaryContent(data=b'image', media_type='image/png'))

    assert session.all_messages() == []


async def test_failed_create_response_releases_its_reservation() -> None:
    # `create_response` reserves a response slot before asking for one, so a solicited response isn't
    # mistaken for the model speaking on its own. When the frame never reaches the provider there is
    # no response coming, and leaving the reservation would mis-label the next spontaneous turn.
    class _BuggyConnection(FakeRealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            raise ValueError('bug in the connection')

    session = RealtimeSession(_BuggyConnection([]), profile=_profile(supports_manual_turn_control=True))
    with pytest.raises(ValueError, match='bug in the connection'):
        await session.create_response()
    assert session._pending_response_requests == 0  # pyright: ignore[reportPrivateUsage]


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
            SessionUsage(usage=RequestUsage(input_tokens=10, output_tokens=5)),
            SessionUsage(usage=RequestUsage(input_tokens=3, output_tokens=2)),
            OutputTranscript(text='ok', is_final=True),
            ResponseDone(),
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
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t', tool_name='f', args='{}'), ResponseDone()])

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        return 'ok'

    session = RealtimeSession(conn, runner)
    _ = await collect_events(session)
    assert session.usage.tool_calls == 1


async def test_session_does_not_count_invalid_tool_call_as_successful() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t', tool_name='f', args='not json')])
    session = RealtimeSession(conn)

    _ = await collect_events(session)

    assert session.usage.tool_calls == 0


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


async def test_concurrent_image_and_text_history_matches_wire_order() -> None:
    image_send_started = asyncio.Event()
    release_image = asyncio.Event()

    class _PausedImageConnection(FakeRealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            if isinstance(content, BinaryImage):
                image_send_started.set()
                await release_image.wait()
            await super().send(content)

    conn = _PausedImageConnection([])
    session = RealtimeSession(conn)
    image_task = asyncio.create_task(session.send(BinaryImage(data=b'image', media_type='image/png')))
    await image_send_started.wait()
    text_task = asyncio.create_task(session.send('text'))
    await asyncio.sleep(0)
    release_image.set()
    await asyncio.gather(image_task, text_task)

    assert [type(frame) for frame in conn.sent] == [BinaryImage, str]
    image_request, text_request = session.new_messages()
    assert isinstance(image_request, ModelRequest)
    assert isinstance(text_request, ModelRequest)
    image_part, text_part = image_request.parts[0], text_request.parts[0]
    assert isinstance(image_part, UserPromptPart)
    assert isinstance(text_part, UserPromptPart)
    assert image_part.content == [BinaryImage(data=b'image', media_type='image/png')]
    assert text_part.content == 'text'


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
    assert kinds.count('str') == 3


async def test_image_input_guard() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, profile=_profile(supports_image_input=False))
    with pytest.raises(UserError, match='does not support image input'):
        await session.send(BinaryImage(data=b'\xff\xd8', media_type='image/jpeg'))
    assert conn.sent == []


async def test_owns_media_guard() -> None:
    # A WebRTC sideband session (owns_media=False) doesn't own the audio transport, so the audio
    # methods are unavailable up front — the browser streams audio to the provider directly.
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn, _noop_runner, owns_media=False)
    with pytest.raises(UserError, match='does not own the audio transport'):
        await session.send_audio(b'\x00\x00')
    with pytest.raises(UserError, match='does not own the audio transport'):
        await session.commit_audio()
    with pytest.raises(UserError, match='does not own the audio transport'):
        await session.clear_audio()
    # The routing through `send()` (audio input / audio bytes) is gated by the same guard.
    with pytest.raises(UserError, match='does not own the audio transport'):
        await session.send(BinaryAudio(data=b'\x00\x00', media_type='audio/pcm'))
    with pytest.raises(UserError, match='does not own the audio transport'):
        await session.send(BinaryContent(data=b'\x00\x00', media_type='audio/pcm'))
    with pytest.raises(UserError, match='browser exchanges audio with the provider directly over WebRTC'):
        await anext(session.stream_audio())
    assert conn.sent == []  # nothing reached the connection


async def test_owns_media_default_allows_audio() -> None:
    # The default (owns_media=True) leaves both audio directions available.
    conn = FakeRealtimeConnection([AudioDelta(data=b'\x01\x02')])
    async with RealtimeSession(conn, _noop_runner, profile=_profile()) as session:
        audio = asyncio.ensure_future(anext(session.stream_audio()))
        await asyncio.sleep(0)
        await session.send_audio(b'\x00\x00')
        assert await audio == b'\x01\x02'
    assert conn.sent == [BinaryAudio(data=b'\x00\x00', media_type='audio/pcm')]


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
    session._notify_pending_messages('asap')  # pyright: ignore[reportPrivateUsage]


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
            ResponseDone(),
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
                ],
                timestamp=IsDatetime(),
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
            yield RealtimeInputSpeechEndEvent(item_id='turn')
            await finish_openai.wait()
            yield InputTranscript(text='hello', is_final=True, item_id='turn')
            yield ResponseDone()

    openai_session = RealtimeSession(_SpeechEndConnection([]), _noop_runner, audio_retention='input_audio')
    await openai_session.send_audio(b'speech')
    speech_ended.set()
    async with openai_session:
        events = openai_session.__aiter__()
        assert isinstance(await anext(events), RealtimeInputSpeechEndEvent)
        await openai_session.send_audio(b'inter-turn silence')
        finish_openai.set()
        _ = [event async for event in events]

    gemini_session = RealtimeSession(
        FakeRealtimeConnection([InputTranscript(text='hello', is_final=False, cumulative=True), ResponseDone()]),
        _noop_runner,
        audio_retention='input_audio',
    )
    await gemini_session.send_audio(b'speech')
    await gemini_session.send_audio(b'model-response silence')
    _ = await collect_events(gemini_session)

    assert openai_session.new_messages()[0] == ModelRequest(
        parts=[SpeechPart(speaker='user', transcript='hello', audio=_wav_content(b'speech'))],
        timestamp=IsDatetime(),
    )
    assert gemini_session.new_messages()[0] == ModelRequest(
        parts=[
            SpeechPart(
                speaker='user',
                transcript='hello',
                audio=_wav_content(b'speechmodel-response silence'),
            )
        ],
        timestamp=IsDatetime(),
    )


async def test_audio_retention_uses_profile_rate_for_each_speaker() -> None:
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='hello', is_final=True),
            AudioDelta(data=b'\x01\x02'),
            OutputTranscript(text='hi', is_final=True),
            ResponseDone(),
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
                ],
                timestamp=IsDatetime(),
            )
        ]
    )


async def test_audio_only_user_turn_finalized_on_speech_stopped() -> None:
    # Transcription off + input audio retained: no `InputTranscript` arrives, so the user's turn is
    # finalized from the retained audio at the speech-stopped boundary (server VAD), as an audio-only
    # `SpeechPart` (no transcript). Bracketed with start/end, since there are no transcript deltas.
    conn = FakeRealtimeConnection(
        [RealtimeInputSpeechEndEvent(), OutputTranscript(text='Hi', is_final=True), ResponseDone()],
        input_transcription_enabled=False,
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m', audio_retention='input_audio')
    await session.send_audio(b'\xaa\xbb')
    events = await collect_events(session)
    user_part = SpeechPart(speaker='user', audio=_wav_content(b'\xaa\xbb'))
    assert events[:3] == [
        PartStartEvent(index=0, part=user_part),
        PartEndEvent(index=0, part=user_part),
        RealtimeInputSpeechEndEvent(),
    ]
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user', audio=_wav_content(b'\xaa\xbb'))], timestamp=IsDatetime()),
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
        [OutputTranscript(text='Hi', is_final=True), ResponseDone()],
        input_transcription_enabled=False,
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m', audio_retention='input_audio')
    await session.send_audio(b'\xaa\xbb')
    _ = await collect_events(session)
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user', audio=_wav_content(b'\xaa\xbb'))], timestamp=IsDatetime()),
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
    session._translate_event(ResponseDone())  # pyright: ignore[reportPrivateUsage]
    await session.send_audio(b'\xbb')
    await session.commit_audio()
    session._translate_event(OutputTranscript(text='second', is_final=True))  # pyright: ignore[reportPrivateUsage]
    session._translate_event(ResponseDone())  # pyright: ignore[reportPrivateUsage]
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
            ModelRequest(parts=[SpeechPart(speaker='user', audio=_wav_content(b'\xaa'))], timestamp=IsDatetime()),
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='first')],
                model_name='m',
                timestamp=IsDatetime(),
                finish_reason='stop',
            ),
            ModelRequest(parts=[SpeechPart(speaker='user', audio=_wav_content(b'\xbb'))], timestamp=IsDatetime()),
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='second')],
                model_name='m',
                timestamp=IsDatetime(),
                finish_reason='stop',
            ),
        ]
    )


_SETTLED_IN_FLIGHT = snapshot(
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
)


@pytest.mark.parametrize(
    ('reconnect_restores_in_flight_state', 'state_restored', 'expected_state_restored', 'expected'),
    [
        # Native resumption (xAI; Gemini settles the cut turn in its own connection): the connection
        # reports it restored the in-flight response, so the two transcript halves belong to one response
        # and nothing is settled. This is the only path that keeps `state_restored=True`.
        pytest.param(
            True,
            True,
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
            id='native-resume-continues',
        ),
        # Local replay (OpenAI, Azure OpenAI): the connection reports `state_restored=True` because it
        # replayed the *finalized* history, but it does not restore in-flight state — the response the
        # drop cut off is gone. The session settles it as an interrupted response and downgrades the
        # user-facing flag to `False`, matching the fully-lost path so an app branches the same way.
        pytest.param(False, True, False, _SETTLED_IN_FLIGHT, id='local-replay-settles-in-flight'),
        # Nothing restored at all: same settlement, flag already `False`.
        pytest.param(False, False, False, _SETTLED_IN_FLIGHT, id='state-lost-settles-in-flight'),
    ],
)
async def test_reconnect_response_state(
    reconnect_restores_in_flight_state: bool,
    state_restored: bool,
    expected_state_restored: bool,
    expected: list[ModelMessage],
) -> None:
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='before', is_final=False),
            RealtimeSessionReconnectEvent(state_restored=state_restored),
            OutputTranscript(text='after', is_final=True),
            ResponseDone(),
        ],
        reconnect_restores_in_flight_state=reconnect_restores_in_flight_state,
    )
    session = RealtimeSession(conn)
    events = await collect_events(session)
    # The flag the app sees reflects what actually survived, not just what the connection replayed.
    assert [e.state_restored for e in events if isinstance(e, RealtimeSessionReconnectEvent)] == [
        expected_state_restored
    ]
    assert session.new_messages() == expected


async def test_reconnect_while_idle_on_replay_provider_keeps_state_restored() -> None:
    # A local-replay provider (OpenAI/Azure) that drops while Listening loses nothing: the replay
    # restores the finalized call and there is no in-flight turn to settle. The connection's
    # `state_restored=True` stands, so an app is not told a seamless renewal was a disruption.
    conn = FakeRealtimeConnection(
        [
            RealtimeSessionReconnectEvent(state_restored=True),
            OutputTranscript(text='after', is_final=True),
            ResponseDone(),
        ],
        reconnect_restores_in_flight_state=False,
    )
    session = RealtimeSession(conn)
    events = await collect_events(session)
    assert [e.state_restored for e in events if isinstance(e, RealtimeSessionReconnectEvent)] == [True]
    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='after')],
                timestamp=IsDatetime(),
                finish_reason='stop',
            )
        ]
    )


async def test_session_offers_the_conversation_for_replay_when_seeding_is_supported() -> None:
    """The session hands the connection a live view of the call, for a provider that must replay it.

    Only where seeding is the mechanism: a provider that resumes natively (Gemini, xAI) has nothing to
    replay, and one that cannot seed at all has nowhere to put it.
    """

    class _RecordsConversation(FakeRealtimeConnection):
        def __init__(self, events: list[RealtimeCodecEvent]) -> None:
            super().__init__(events)
            self.message_history: Callable[[], Sequence[ModelMessage]] | None = None

        def set_message_history(self, message_history: Callable[[], Sequence[ModelMessage]]) -> None:
            self.message_history = message_history

    seeding = _RecordsConversation([])
    async with RealtimeSession(seeding, profile=_profile(supports_session_seeding=True)) as session:
        await session.send('hello')
    assert seeding.message_history is not None
    # A live view, not a snapshot: the call grows after the connection is handed it.
    assert [type(m).__name__ for m in seeding.message_history()] == ['ModelRequest']

    no_seeding = _RecordsConversation([])
    async with RealtimeSession(no_seeding, profile=_profile(supports_session_seeding=False)):
        pass
    assert no_seeding.message_history is None


async def test_reconnect_while_idle_passes_through() -> None:
    # A lost-state reconnect with nothing in flight has no pre-drop response to finalize: the event
    # passes through and the next turn is recorded normally.
    conn = FakeRealtimeConnection(
        [
            RealtimeSessionReconnectEvent(state_restored=False),
            OutputTranscript(text='after', is_final=True),
            ResponseDone(),
        ]
    )
    session = RealtimeSession(conn)
    events = await collect_events(session)
    assert any(isinstance(e, RealtimeSessionReconnectEvent) for e in events)
    assert session.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='after')],
                timestamp=IsDatetime(),
                finish_reason='stop',
            )
        ]
    )


async def test_queued_message_flushes_when_reconnect_closes_orphaned_turn() -> None:
    """A boundary closing a reply the reconnect orphaned flushes messages queued behind the turn.

    Gemini never continues an in-flight generation on a re-dialed connection, so its connection closes
    the orphaned turn with `ResponseDone(interrupted=True)` ahead of the reconnect event (see
    `test_reconnect_closes_orphaned_turn_with_interrupted_boundary` in `test_google.py`). Without that
    boundary the open speech part keeps the response active forever: a `when_idle` prompt enqueued
    during the reply would never be delivered and the turn would never officially end.
    """
    agent: Agent[None, str] = Agent()

    class _DropsMidReply(FakeRealtimeConnection):
        def __init__(self) -> None:
            super().__init__([])
            self.audio_started = asyncio.Event()
            self.enqueued = asyncio.Event()

        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            yield ToolCall(tool_call_id='tc', tool_name='queue_followup', args='{}')
            while not any(isinstance(item, ToolResult) for item in self.sent):
                await asyncio.sleep(0)
            yield AudioDelta(data=b'audio')  # the reply to the tool result begins...
            self.audio_started.set()
            await self.enqueued.wait()
            # ...then the connection drops mid-reply and reconnects with state restored.
            yield ResponseDone(interrupted=True)
            yield RealtimeSessionReconnectEvent(state_restored=True)

    conn = _DropsMidReply()

    @agent.tool
    async def queue_followup(ctx: RunContext[object]) -> str:
        async def enqueue_during_reply() -> None:
            await conn.audio_started.wait()
            ctx.enqueue('queued for the boundary', priority='when_idle')
            conn.enqueued.set()

        asyncio.create_task(enqueue_during_reply())
        return 'done'

    async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
        events = await drain_events(session)

    # The orphaned reply is closed as interrupted, the turn ends, and the queued prompt goes out.
    assert any(isinstance(e, RealtimeTurnCompleteEvent) for e in events)
    assert [item for item in conn.sent if isinstance(item, str)] == ['queued for the boundary']
    responses = [m for m in session.all_messages() if isinstance(m, ModelResponse)]
    assert responses[-1].state == 'interrupted'


async def test_reconnect_finalizes_multiple_in_flight_user_items() -> None:
    # Two overlapping user turns (partial transcripts routed by item id), with the second finalizing
    # out of order: `u2` is finalized in `_user_turns` behind the still-open `u1`. A lost-state
    # reconnect must finalize `u1` and then flush both in order, so neither turn is dropped or
    # double-recorded.
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='first turn', is_final=False, item_id='u1'),
            InputTranscript(text='second turn', is_final=False, item_id='u2'),
            InputTranscript(text='second turn', is_final=True, item_id='u2'),
            RealtimeSessionReconnectEvent(state_restored=False),
            OutputTranscript(text='after', is_final=True),
            ResponseDone(),
        ]
    )
    session = RealtimeSession(conn)
    await collect_events(session)
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user', transcript='first turn')], timestamp=IsDatetime()),
            ModelRequest(parts=[SpeechPart(speaker='user', transcript='second turn')], timestamp=IsDatetime()),
            ModelResponse(
                parts=[SpeechPart(speaker='assistant', transcript='after')],
                timestamp=IsDatetime(),
                finish_reason='stop',
            ),
        ]
    )


async def test_finalizing_overlapping_user_item_keeps_turn_active() -> None:
    session = RealtimeSession(FakeRealtimeConnection([]))
    session._handle_input_transcript('first', False, item_id='u1')  # pyright: ignore[reportPrivateUsage]
    session._handle_input_transcript('second', False, item_id='u2')  # pyright: ignore[reportPrivateUsage]

    session._handle_input_transcript('first', True, item_id='u1')  # pyright: ignore[reportPrivateUsage]

    assert session._user_turn_active  # pyright: ignore[reportPrivateUsage]


async def test_reconnect_cancels_obsolete_tool_call() -> None:
    cancelled = asyncio.Event()

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        try:
            await asyncio.Event().wait()
            raise AssertionError('unreachable')  # pragma: no cover - the reconnect cancels this task
        finally:
            cancelled.set()

    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='old-call', tool_name='slow', args='{}'),
            RealtimeSessionReconnectEvent(state_restored=False),
        ]
    )
    session = RealtimeSession(conn, runner)
    _ = await collect_events(session)

    assert cancelled.is_set()
    assert conn.sent == []
    response, request = session.new_messages()
    assert isinstance(response, ModelResponse)
    assert isinstance(request, ModelRequest)
    assert len(request.parts) == 1
    result = request.parts[0]
    assert isinstance(result, ToolReturnPart)
    assert result.tool_name == 'slow'
    assert result.content == 'The tool call was interrupted before a result was produced.'
    assert result.tool_call_id == 'old-call'
    assert result.outcome == 'interrupted'


async def test_audio_retained_with_transcription_enabled_waits_for_transcript() -> None:
    # With transcription enabled, a speech-stopped boundary does NOT emit an audio-only turn: the turn is
    # finalized from the (asynchronously delivered) transcript instead, so there's exactly one user turn —
    # never a duplicate audio-only one racing the transcript.
    conn = FakeRealtimeConnection(
        [RealtimeInputSpeechEndEvent(), InputTranscript(text='hello', is_final=True), ResponseDone()],
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
                ],
                timestamp=IsDatetime(),
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
            yield RealtimeInputSpeechEndEvent(item_id='A')  # segments turn A's audio
            await gate_b.wait()
            yield RealtimeInputSpeechEndEvent(item_id='B')  # segments turn B's audio
            yield InputTranscript(text='second', is_final=True, item_id='B')  # B finalizes first...
            yield InputTranscript(text='first', is_final=True, item_id='A')  # ...then A, out of order
            yield ResponseDone()

    conn = _Overlapping([])
    session = RealtimeSession(conn, _noop_runner, audio_retention='input_audio')
    await session.send_audio(b'\xaa')  # turn A's audio, buffered before its boundary fires
    gate_a.set()
    async with session:
        async for event in session:
            # When A's boundary has passed (its segment is captured), queue B's audio and release B's.
            if isinstance(event, RealtimeInputSpeechEndEvent) and event.item_id == 'A':
                await session.send_audio(b'\xbb')
                gate_b.set()

    assert session.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[SpeechPart(speaker='user', transcript='second', audio=_wav_content(b'\xbb'))],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[SpeechPart(speaker='user', transcript='first', audio=_wav_content(b'\xaa'))],
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_retained_input_audio_kept_when_transcription_fails() -> None:
    # A failed transcript must keep the retained audio on its placeholder without leaking it to a later turn.
    conn = FakeRealtimeConnection(
        [
            RealtimeInputSpeechEndEvent(item_id='A'),
            RealtimeInputTranscriptionErrorEvent(message='transcription failed', item_id='A'),
            InputTranscript(text='hi', is_final=True, item_id='B'),
            ResponseDone(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, audio_retention='input_audio')
    await session.send_audio(b'\xaa')
    _ = await collect_events(session)
    assert session.new_messages() == snapshot(
        [
            ModelRequest(parts=[SpeechPart(speaker='user', audio=_wav_content(b'\xaa'))], timestamp=IsDatetime()),
            ModelRequest(parts=[SpeechPart(speaker='user', transcript='hi')], timestamp=IsDatetime()),
        ]
    )


async def test_no_transcription_and_no_input_retention_records_placeholder_once() -> None:
    conn = FakeRealtimeConnection(
        [RealtimeInputSpeechStartEvent(), RealtimeInputSpeechEndEvent(), ResponseDone()],
        input_transcription_enabled=False,
    )
    session = RealtimeSession(conn, _noop_runner)
    events = await collect_events(session)

    assert [type(event).__name__ for event in events] == [
        'RealtimeInputSpeechStartEvent',
        'PartStartEvent',
        'PartEndEvent',
        'RealtimeInputSpeechEndEvent',
        'RealtimeTurnCompleteEvent',
    ]
    assert session.new_messages() == [ModelRequest(parts=[SpeechPart(speaker='user')], timestamp=IsDatetime())]


async def test_no_transcription_with_input_retention_is_allowed() -> None:
    # Disabling transcription is fine as long as input audio is retained (audio-only user turns).
    conn = FakeRealtimeConnection([], input_transcription_enabled=False)
    RealtimeSession(conn, _noop_runner, audio_retention='input_audio')  # no error


async def test_manual_contentless_user_turn_without_transcription() -> None:
    conn = FakeRealtimeConnection([], input_transcription_enabled=False)
    session = RealtimeSession(conn, _noop_runner)

    await session.commit_audio()
    events = await collect_events(session)

    assert session.new_messages() == [ModelRequest(parts=[SpeechPart(speaker='user')], timestamp=IsDatetime())]
    assert [type(event).__name__ for event in events] == ['PartStartEvent', 'PartEndEvent']


async def test_text_output_modality_produces_text_part() -> None:
    # A text-output response (`OutputTranscript(output_text=True)`, from `output_modalities=('text',)`) must
    # be emitted and persisted as a `TextPart`, not a `SpeechPart` — it carries no speech.
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='hi', is_final=False, output_text=True),
            OutputTranscript(text='hi there', is_final=True, output_text=True),
            ResponseDone(),
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
    conn = FakeRealtimeConnection([AudioDelta(data=b'\x00\x01'), ResponseDone()])
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    events = await collect_events(session)
    assert [type(e).__name__ for e in events] == [
        'PartStartEvent',
        'PartDeltaEvent',
        'PartEndEvent',
        'RealtimeTurnCompleteEvent',
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
    assert session.new_messages() == [ModelRequest(parts=[SpeechPart(speaker='user')], timestamp=IsDatetime())]


async def test_duplicate_final_input_transcript_is_idempotent() -> None:
    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='hello', is_final=True, item_id='user-1'),
            InputTranscript(text='hello', is_final=True, item_id='user-1'),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='m')
    _ = await collect_events(session)

    assert session.all_messages() == [
        ModelRequest(parts=[SpeechPart(speaker='user', transcript='hello')], timestamp=IsDatetime())
    ]


async def test_transcript_only_default_drops_audio() -> None:
    conn = FakeRealtimeConnection(
        [AudioDelta(data=b'\x00'), OutputTranscript(text='hi', is_final=True), ResponseDone()]
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
    conn = FakeRealtimeConnection([OutputTranscript(text='reply', is_final=True), ResponseDone()])
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
    conn = FakeRealtimeConnection([OutputTranscript(text='one', is_final=True), ResponseDone()])
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
            ResponseDone(),
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
            ModelRequest(parts=[SpeechPart(speaker='user', transcript='hello')], timestamp=IsDatetime()),
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
    assert seed_speech_content(part=user, provider_name='test', supports_audio=False) == ''
    assert seed_speech_content(part=assistant, provider_name='test', supports_audio=False) == ''

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
    parts: list[NativeToolCallPart | NativeToolReturnPart], *, first_index: int = 0
) -> list[PartStartEvent | PartEndEvent]:
    """Start/end pairs for native-tool parts, numbered from `first_index`.

    A connection numbers these from its own counter (so the default `0`), while the session remaps
    them onto its session-unique allocator before forwarding — hence the two numberings.
    """
    return [
        event
        for index, part in enumerate(parts, start=first_index)
        for event in (PartStartEvent(index=index, part=part), PartEndEvent(index=index, part=part))
    ]


async def test_native_part_end_without_a_start_gets_an_index_of_its_own() -> None:
    # A `PartEndEvent` the session never saw a start for closes nothing, so it must not inherit the
    # connection's index: that index isn't session-unique and would name a live part — here the
    # assistant's speech part, which also holds index 0.
    part = _grounding_parts()[0]
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='It is sunny in Rome', is_final=True),
            PartEndEvent(index=0, part=part),
            ResponseDone(),
        ]
    )
    session = RealtimeSession(conn, _noop_runner, model_name='gemini-live-2.5-flash')
    events = await collect_events(session)

    speech_indexes = {event.index for event in events if isinstance(event, PartStartEvent)}
    orphan = next(event for event in events if isinstance(event, PartEndEvent) and event.part == part)
    assert speech_indexes == {0}
    assert orphan.index not in speech_indexes


async def test_grounding_streams_and_folds_native_tool_parts() -> None:
    # Grounding parts stream to the consumer and fold into the assistant response ahead of speech,
    # mirroring the classic `GoogleModel`.
    grounding = _grounding_parts()
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='It is sunny in Rome', is_final=True),
            *_native_part_events(grounding),
            ResponseDone(),
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
        # The connection numbered these `0` and `1` from its own counter, which would have collided
        # with the speech part's index `0` above — a repeated index *replaces* the part at it, so a
        # consumer keyed on the index would have shown the search in place of the model's answer. The
        # session remaps them onto its own allocator, which had already handed `0` to the speech part.
        *_native_part_events(grounding, first_index=1),
        PartEndEvent(index=0, part=SpeechPart(speaker='assistant', transcript='It is sunny in Rome')),
        RealtimeTurnCompleteEvent(),
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
            ResponseDone(),
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
            ModelRequest(
                parts=[UserPromptPart(content='weather in rome?', timestamp=IsDatetime())], timestamp=IsDatetime()
            ),
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
            ResponseDone(),
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
            ModelRequest(
                parts=[UserPromptPart(content='what is 1 + 1?', timestamp=IsDatetime())], timestamp=IsDatetime()
            ),
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
        [ToolCall(tool_call_id='tc_5', tool_name='greet', args='{"name": "Alice"}'), ResponseDone()]
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
    conn = FakeRealtimeConnection([ResponseDone()])
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
    conn = FakeRealtimeConnection([ResponseDone()])
    model = FakeRealtimeModel(conn, profile=_profile(supports_session_seeding=False))
    seed = [ModelRequest(parts=[UserPromptPart(content='earlier question')])]
    with pytest.raises(UserError, match='does not support seeding a session'):
        async with agent.realtime(model, message_history=seed).session():
            pass  # pragma: no cover


async def test_agent_realtime_session_rejects_text_output_when_unsupported() -> None:
    # A model that only speaks rejects `output_modality='text'` up front, before dialing. Unlike a
    # setting the provider merely ignores, this one changes what the caller gets back: Gemini fails the
    # handshake over it and xAI would quietly answer with audio.
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([ResponseDone()])
    model = FakeRealtimeModel(conn, profile=_profile(supports_text_output=False))
    with pytest.raises(UserError, match="does not support `output_modality='text'`"):
        async with agent.realtime(model, model_settings=RealtimeModelSettings(output_modality='text')).session():
            pass  # pragma: no cover

    # The same model is fine for the default audio modality, so the guard is scoped to the request.
    async with agent.realtime(model, model_settings=RealtimeModelSettings(output_modality='audio')).session():
        pass


async def test_agent_realtime_session_allows_text_output_by_default() -> None:
    # `supports_text_output` defaults to `True` — a realtime model that can't write is the exception —
    # so a profile that says nothing about it still accepts `output_modality='text'`.
    assert 'supports_text_output' not in _profile_without_text_output()
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([ResponseDone()])
    model = FakeRealtimeModel(conn, profile=_profile_without_text_output())
    async with agent.realtime(model, model_settings=RealtimeModelSettings(output_modality='text')).session():
        pass
    assert (model.last_model_settings or {}).get('output_modality') == 'text'


def _profile_without_text_output() -> RealtimeModelProfile:
    """A profile that never mentions `supports_text_output`, to pin the `True` default."""
    return RealtimeModelProfile(supports_session_seeding=True)


async def test_agent_realtime_session_audio_retention_forwarded() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection(
        [AudioDelta(data=b'\x07'), OutputTranscript(text='hi', is_final=True), ResponseDone()]
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
        ModelRequest(
            parts=[UserPromptPart(content=[images[0]], timestamp=IsDatetime())],
            timestamp=IsDatetime(),
            conversation_id=IsStr(),
            run_id=IsStr(),
        ),
        ModelRequest(
            parts=[UserPromptPart(content=[images[2]], timestamp=IsDatetime())],
            timestamp=IsDatetime(),
            conversation_id=IsStr(),
            run_id=IsStr(),
        ),
    ]


async def test_agent_realtime_session_additional_instructions() -> None:
    agent: Agent[None, str] = Agent(instructions='Default')
    conn = FakeRealtimeConnection([ResponseDone()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model, instructions='Custom').session() as session:
        _ = [e async for e in session]
    # Per-run instructions are their own block, so they're separated from the agent's by a blank
    # line, exactly as in a graph run.
    assert model.last_instructions == 'Default\n\nCustom'


async def test_agent_realtime_session_default_instructions_empty() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([ResponseDone()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        _ = [e async for e in session]
    assert model.last_instructions == ''


async def test_agent_realtime_session_unknown_tool() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc_x', tool_name='nonexistent', args='{}'), ResponseDone()])
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

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='explode', args='{}'), ResponseDone()])
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

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='boom', args='{}'), ResponseDone()])
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

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='double', args='{"x": "21"}'), ResponseDone()])
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
        [ToolCall(tool_call_id='tc', tool_name='double', args='{"x": "not a number"}'), ResponseDone()]
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
        yield SessionUsage(usage=RequestUsage(input_tokens=1, output_tokens=1))
        yield ResponseDone()


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
        yield ResponseDone()


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

    assert not any(isinstance(item, str) for item in conn.sent_before_response_complete)
    assert [item for item in conn.sent if isinstance(item, str)] == ['follow-up context']


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

    assert [type(item).__name__ for item in conn.sent] == ['ToolResult', 'str']
    call_response, tool_return, followup = session.new_messages()
    assert isinstance(call_response, ModelResponse) and isinstance(call_response.parts[0], ToolCallPart)
    assert isinstance(tool_return, ModelRequest) and isinstance(tool_return.parts[0], ToolReturnPart)
    assert followup == ModelRequest(
        parts=[UserPromptPart(content='follow-up context', timestamp=IsDatetime())],
        timestamp=IsDatetime(),
        conversation_id=IsStr(),
        run_id=IsStr(),
    )


class _ConcurrentEnqueueConnection(FakeRealtimeConnection):
    """Block the first pending-message send while a second sync tool appends to the queue."""

    def __init__(self) -> None:
        super().__init__([])
        self.first_send_started = ThreadEvent()
        self.second_enqueued = ThreadEvent()

    async def send(self, content: RealtimeInput) -> None:
        self.sent.append(content)
        if content == 'first':
            self.first_send_started.set()
            while not self.second_enqueued.is_set():
                await asyncio.sleep(0)

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        yield ToolCall(tool_call_id='tc-1', tool_name='queue_concurrently', args='{"text": "first"}')
        yield ToolCall(tool_call_id='tc-2', tool_name='queue_concurrently', args='{"text": "second"}')
        while sum(isinstance(item, ToolResult) for item in self.sent) < 2:
            await asyncio.sleep(0)
        yield ResponseDone()


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

    assert [item for item in conn.sent if isinstance(item, str)] == ['first', 'second']
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
        with pytest.raises(UserError, match='supports plain-text prompts and system-prompt parts only'):
            _ = [event async for event in session]


@pytest.mark.parametrize(
    'messages',
    [
        [ModelResponse(parts=[TextPart(content='not a request')])],
        [ModelRequest(parts=[UserPromptPart(content=[BinaryImage(data=b'x', media_type='image/png')])])],
        [ModelRequest(parts=[])],
    ],
)
async def test_realtime_pending_messages_reject_unsupported_message_shapes(messages: list[ModelMessage]) -> None:
    session = RealtimeSession(FakeRealtimeConnection([]))
    manager = session._tool_manager  # pyright: ignore[reportPrivateUsage]
    assert manager.ctx is not None
    assert manager.ctx.pending_messages is not None
    with pytest.raises(UserError, match='supports plain-text prompts and system-prompt parts only'):
        manager.ctx.pending_messages.append(PendingMessage(messages=messages))


def test_realtime_pending_messages_join_text_and_tag_system_prompts() -> None:
    # Multiple text parts join across messages, and a mid-conversation `SystemPromptPart` degrades to
    # `<system>`-tagged user text — the same treatment `Model.prepare_messages` applies in a standard run.
    pending = PendingMessage(
        messages=[
            ModelRequest(parts=[UserPromptPart(content='one'), SystemPromptPart(content='rule')]),
            ModelRequest(parts=[UserPromptPart(content='two')]),
        ]
    )
    assert _pending_message_text(pending) == 'one\n\n<system>rule</system>\n\ntwo'


async def test_session_exit_is_idempotent_and_flushes_unfinalized_user() -> None:
    session = RealtimeSession(FakeRealtimeConnection([InputTranscript(text='partial')]))
    await session.__aexit__(None, None, None)
    async with session:
        _ = [event async for event in session]
    await session.__aexit__(None, None, None)

    assert session.new_messages() == [
        ModelRequest(parts=[SpeechPart(speaker='user', transcript='partial')], timestamp=IsDatetime())
    ]


def test_session_accepts_unprepared_tool_manager_without_pending_context() -> None:
    manager = ToolManager(FunctionToolset())
    session = _RealtimeSession(FakeRealtimeConnection([]), tool_manager=manager)
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


async def test_existing_assistant_speech_associates_late_item_id_in_session_state() -> None:
    session = RealtimeSession(
        FakeRealtimeConnection(
            [
                AudioDelta(data=b'first'),
                AudioDelta(data=b'second', item_id='assistant-item'),
                OutputTranscript(text='spoken', item_id='assistant-item'),
                ResponseDone(),
            ]
        ),
        provider_name='openai',
    )
    _ = await collect_events(session)

    response = session.new_messages()[0]
    assert isinstance(response, ModelResponse)
    part = response.parts[0]
    assert isinstance(part, SpeechPart)
    assert part.id is None
    assert len(response.parts) == 1


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
        ModelRequest(parts=[SpeechPart(speaker='user')], timestamp=IsDatetime()),
        ModelRequest(parts=[SpeechPart(speaker='user', transcript='kept')], timestamp=IsDatetime()),
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
    session._notify_pending_messages('asap')  # pyright: ignore[reportPrivateUsage]
    session._closed = True  # pyright: ignore[reportPrivateUsage]
    session._start_pending_message_drain('asap')  # pyright: ignore[reportPrivateUsage]


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
        call_part: ToolCallPart,
        *,
        validation_done: asyncio.Event,
        execution_prerequisites: tuple[asyncio.Event, ...],
        response_usage_follows: bool,
        run_step: int,
        reserved_budget: bool,
    ) -> tuple[ToolReturnPart, None]:
        del validation_done, execution_prerequisites, response_usage_follows
        del run_step, reserved_budget
        session._tool_calls_awaiting_usage.clear()  # pyright: ignore[reportPrivateUsage]
        return ToolReturnPart(tool_name=call_part.tool_name, content='done', tool_call_id=call_part.tool_call_id), None

    session._tool_calls_awaiting_usage.add('call')  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(session, '_execute_tool', complete_after_usage)
    completion = asyncio.Event()
    await session._run_tool(  # pyright: ignore[reportPrivateUsage]
        ToolCallPart(tool_name='noop', args={}, tool_call_id='call'),
        validation_done=validation_done,
        execution_prerequisites=(),
        completion=completion,
        response_usage_follows=True,
        run_step=0,
        reserved_budget=True,
        order_index=0,
        ordered_events=False,
    )

    assert 'after tool' in conn.sent


async def test_deferred_asap_drain_failure_after_tool_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    # The post-`finally` deferred `asap` drain in `_run_tool` runs OUTSIDE its try/except. If its
    # `connection.send` fails (e.g. the socket just dropped), `_tool_task_done` must forward the error to
    # the consumer — mirroring `_pending_message_task_done` — instead of letting it vanish as an
    # unretrieved-task-exception warning at GC, silently losing the enqueued follow-up.
    class _FailingDrain(FakeRealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:
            if isinstance(content, str):
                raise RuntimeError('drain send failed')
            # This test only drives the drain's text-turn send.
            await super().send(content)  # pragma: no cover

    conn = _FailingDrain([])
    session = RealtimeSession(conn)
    session._asap_drain_deferred = True  # pyright: ignore[reportPrivateUsage]
    session._pending_messages.append(  # pyright: ignore[reportPrivateUsage]
        PendingMessage(messages=[ModelRequest(parts=[UserPromptPart(content='after tool')])], priority='asap')
    )

    async def complete_after_usage(
        call_part: ToolCallPart,
        *,
        validation_done: asyncio.Event,
        execution_prerequisites: tuple[asyncio.Event, ...],
        response_usage_follows: bool,
        run_step: int,
        reserved_budget: bool,
    ) -> tuple[ToolReturnPart, None]:
        del validation_done, execution_prerequisites, response_usage_follows
        del run_step, reserved_budget
        session._tool_calls_awaiting_usage.clear()  # pyright: ignore[reportPrivateUsage]
        return ToolReturnPart(tool_name=call_part.tool_name, content='done', tool_call_id=call_part.tool_call_id), None

    session._tool_calls_awaiting_usage.add('call')  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(session, '_execute_tool', complete_after_usage)

    task = asyncio.create_task(
        session._run_tool(  # pyright: ignore[reportPrivateUsage]
            ToolCallPart(tool_name='noop', args={}, tool_call_id='call'),
            validation_done=asyncio.Event(),
            execution_prerequisites=(),
            completion=asyncio.Event(),
            response_usage_follows=True,
            run_step=0,
            reserved_budget=True,
            order_index=0,
            ordered_events=False,
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
                ResponseDone(),
            ]
        ),
        runner=runner,
        usage_limits=UsageLimits(tool_calls_limit=1),
    )

    async with session:
        with pytest.raises(UsageLimitExceeded, match='tool_calls_limit'):
            _ = [event async for event in session]
    assert session.usage.tool_calls == 1


async def test_close_settles_in_flight_state_into_history() -> None:
    # A session closed mid-turn settles what it still holds open, exactly as a reconnect settles
    # state the provider lost: the partial reply lands in history as interrupted and the running tool
    # call gets a cancelled return, so `all_messages()` hands off a valid history instead of dropping
    # the tail or ending on a dangling `ToolCallPart`.
    async def runner(*args: Any) -> str:
        await asyncio.Event().wait()  # never completes; cancelled at session close
        return 'unreachable'  # pragma: no cover

    session = RealtimeSession(
        FakeRealtimeConnection(
            [
                OutputTranscript(text='one moment', is_final=False),
                ToolCall(tool_call_id='tc1', tool_name='slow', args='{}'),
            ]
        ),
        runner=runner,
    )
    async with session:
        async for event in session:
            if isinstance(event, FunctionToolCallEvent):
                break

    messages = session.all_messages()
    response = messages[0]
    assert isinstance(response, ModelResponse)
    assert [type(part).__name__ for part in response.parts] == ['SpeechPart', 'ToolCallPart']
    cancelled = messages[1]
    assert isinstance(cancelled, ModelRequest)
    cancelled_part = cancelled.parts[0]
    assert isinstance(cancelled_part, ToolReturnPart)
    assert cancelled_part.tool_call_id == 'tc1'
    assert cancelled_part.outcome == 'interrupted'


async def test_close_settles_partial_reply_as_interrupted() -> None:
    # A reply still streaming when the session closes lands in history as an interrupted response,
    # instead of the buffered transcript silently vanishing from `all_messages()`.
    session = RealtimeSession(FakeRealtimeConnection([OutputTranscript(text='one moment', is_final=False)]))
    async with session:
        async for event in session:
            if isinstance(event, PartDeltaEvent):
                break

    messages = session.all_messages()
    assert len(messages) == 1
    response = messages[0]
    assert isinstance(response, ModelResponse)
    assert response.state == 'interrupted'
    speech = response.parts[0]
    assert isinstance(speech, SpeechPart) and speech.transcript == 'one moment'


async def test_tool_call_limit_counts_in_flight_calls() -> None:
    # `ToolManager` records a call on `usage.tool_calls` only once it *succeeds*, so with a slow tool
    # a burst of parallel calls would each compare against the same pre-burst count and all clear a
    # limit only one of them fits under. The projection must count calls still in flight.
    started = asyncio.Event()

    async def runner(*args: Any) -> str:
        started.set()
        await asyncio.Event().wait()  # never completes; cancelled at session close
        return 'unreachable'  # pragma: no cover

    session = RealtimeSession(
        FakeRealtimeConnection(
            [
                ToolCall(tool_call_id='first', tool_name='slow', args='{}'),
                ToolCall(tool_call_id='second', tool_name='slow', args='{}'),
                ResponseDone(),
            ]
        ),
        runner=runner,
        usage_limits=UsageLimits(tool_calls_limit=1),
    )

    async with session:
        with pytest.raises(UsageLimitExceeded, match='tool_calls_limit'):
            _ = [event async for event in session]
    assert started.is_set()  # the first call was running — and unrecorded — when the second was checked
    assert session.usage.tool_calls == 0


async def test_iterator_reuses_receive_pump_started_by_session_owner() -> None:
    session = RealtimeSession(FakeRealtimeConnection([ResponseDone()]))
    async with session:
        session._pump_task = asyncio.create_task(  # pyright: ignore[reportPrivateUsage]
            session._pump(session._session_instrumentation.context)  # pyright: ignore[reportPrivateUsage]
        )
        assert [event async for event in session] == [RealtimeTurnCompleteEvent()]


async def test_receive_pump_stops_when_event_handler_trips_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    session = RealtimeSession(FakeRealtimeConnection([ResponseDone(), ResponseDone()]))
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
        [ToolCall(tool_call_id='tc', tool_name='weather', args='{"city": "forbidden"}'), ResponseDone()]
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

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='info', args='{}'), ResponseDone()])
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

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='danger', args='{}'), ResponseDone()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model, capabilities=[HandleDeferredToolCalls(handler=deny)]).session() as session:
        events = [e async for e in session]

    lifecycle = [event for event in events if isinstance(event, (DeferredToolRequestsEvent, DeferredToolResultsEvent))]
    assert lifecycle == [
        DeferredToolRequestsEvent(
            DeferredToolRequests(approvals=[ToolCallPart(tool_name='danger', args='{}', tool_call_id='tc')])
        ),
        DeferredToolResultsEvent(DeferredToolResults(approvals={'tc': False})),
    ]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, ToolReturnPart)
    assert result.part.outcome == 'denied'
    assert 'denied' in str(result.part.content).lower()


# --- declarative `requires_approval=True` gating ------------------------------------------------
#
# A tool can be deferred *declaratively* (`requires_approval=True` → `ToolDefinition.kind='unapproved'`)
# as well as by raising `ApprovalRequired`. The graph pipeline classifies by kind before executing
# anything; the session's `handle_call` path used to execute first and react to what was raised, so a
# `requires_approval=True` tool ran with approval silently skipped and no handler ever consulted. The
# four tests below pin the whole matrix, and each asserts *whether the body ran* — the thing that
# actually matters when the gate is approval.


def _approval_agent() -> tuple[Agent[None, str], list[str]]:
    """An agent whose only tool is declaratively approval-gated, plus an execution log."""
    executed: list[str] = []
    agent: Agent[None, str] = Agent()

    @agent.tool_plain(requires_approval=True)
    def transfer_funds() -> str:
        executed.append('ran')
        return 'transferred'

    return agent, executed


async def test_standard_run_pauses_on_a_declaratively_approval_gated_tool() -> None:
    # The reference behavior the session has to match: the run ends with the call awaiting approval,
    # and the body never runs.
    agent, executed = _approval_agent()

    def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('transfer_funds', {}, tool_call_id='tc')])

    result = await agent.run('go', model=FunctionModel(call_tool), output_type=[str, DeferredToolRequests])
    assert isinstance(result.output, DeferredToolRequests)
    assert [call.tool_name for call in result.output.approvals] == ['transfer_funds']
    assert executed == []


async def test_realtime_session_does_not_execute_an_approval_gated_tool_without_a_handler() -> None:
    # No handler to resolve the approval: the session answers the model with the documented
    # explanation and — the point of the gate — never runs the tool.
    agent, executed = _approval_agent()

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='transfer_funds', args='{}'), ResponseDone()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        events = [e async for e in session]

    assert executed == []
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, ToolReturnPart)
    assert 'requires approval' in str(result.part.content)
    assert 'cannot be completed during a realtime session' in str(result.part.content)
    # A refused approval is recorded as a failure, so a handoff to `Agent.run` can't read it as a
    # tool that ran and returned an error-shaped string.
    assert result.part.outcome == 'failed'


async def test_realtime_session_denies_an_approval_gated_tool_through_a_handler() -> None:
    # A `HandleDeferredToolCalls` handler that denies is now actually consulted, and the denial is
    # recorded as such rather than as a successful return.
    agent, executed = _approval_agent()

    def deny(ctx: RunContext[Any], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: False for call in requests.approvals})

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='transfer_funds', args='{}'), ResponseDone()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model, capabilities=[HandleDeferredToolCalls(handler=deny)]).session() as session:
        events = [e async for e in session]

    assert executed == []
    lifecycle = [event for event in events if isinstance(event, DeferredToolRequestsEvent)]
    assert lifecycle == [
        DeferredToolRequestsEvent(
            DeferredToolRequests(approvals=[ToolCallPart(tool_name='transfer_funds', args='{}', tool_call_id='tc')])
        )
    ]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, ToolReturnPart)
    assert result.part.outcome == 'denied'


async def test_realtime_session_executes_an_approval_gated_tool_once_approved() -> None:
    # Approval granted inline: the tool runs and its real return reaches the model, so the gate
    # blocks rather than breaks the tool.
    agent, executed = _approval_agent()

    def approve(ctx: RunContext[Any], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='transfer_funds', args='{}'), ResponseDone()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model, capabilities=[HandleDeferredToolCalls(handler=approve)]).session() as session:
        events = [e async for e in session]

    assert executed == ['ran']
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, ToolReturnPart)
    assert result.part.content == 'transferred'
    assert result.part.outcome == 'success'


async def test_deferred_call_does_not_consume_the_tool_call_limit() -> None:
    # A `requires_approval=True` call with no handler is refused and never reaches the tool body, so it
    # never increments `usage.tool_calls`. The graph leaves such calls out of its pre-check projection
    # entirely (`function_indices` is built from `('function', 'unknown')`), so charging the budget for
    # one here would trip the limit over a call that costs nothing.
    agent, executed = _approval_agent()

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='transfer_funds', args='{}'), ResponseDone()])
    model = FakeRealtimeModel(conn)
    # A budget of zero: any charge at all would raise `UsageLimitExceeded`.
    async with agent.realtime(model, usage_limits=UsageLimits(tool_calls_limit=0)).session() as session:
        events = [e async for e in session]

    assert executed == []
    assert session.usage.tool_calls == 0
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, ToolReturnPart)
    assert 'requires approval' in str(result.part.content)


async def test_executed_call_still_consumes_the_tool_call_limit() -> None:
    # The other side of the same branch: an ordinary tool is charged exactly as before, so skipping the
    # reservation for deferred calls didn't quietly stop enforcing the limit.
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def ordinary() -> str:  # pragma: no cover — the limit trips before the body runs
        return 'ran'

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='ordinary', args='{}'), ResponseDone()])
    model = FakeRealtimeModel(conn)
    with pytest.raises(UsageLimitExceeded):
        async with agent.realtime(model, usage_limits=UsageLimits(tool_calls_limit=0)).session() as session:
            _ = [e async for e in session]


async def test_tools_from_one_response_share_a_run_step() -> None:
    # Pins the graph's invariant: every call from one response runs at the step in effect when that
    # response produced it, the way a graph batch shares the step advanced before its request.
    #
    # This passes against the previous code too — reading `self._tool_run_step` inside `_execute_tool`
    # happened to be safe, because `on_validate` sets `validation_done` *before* awaiting the barrier
    # and the pump blocks on it, pinning every manager sync ahead of the next upstream event. The
    # capture makes the invariant the code's own rather than a consequence of that interleaving, and
    # this test is what would catch it if either side of that subtle arrangement moved.
    seen_steps: list[int] = []
    agent: Agent[None, str] = Agent(deps_type=type(None))

    @agent.tool(sequential=True)
    def barrier(ctx: RunContext[None]) -> str:
        seen_steps.append(ctx.run_step)
        return 'first'

    @agent.tool
    def follower(ctx: RunContext[None]) -> str:
        seen_steps.append(ctx.run_step)
        return 'second'

    # The OpenAI shape: both calls belong to one response whose usage — and so whose finalization,
    # which advances the step — arrives after them. The follower is held behind the barrier until
    # after that advance, which is exactly when reading the step late would diverge.
    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='tc1', tool_name='barrier', args='{}', response_usage_follows=True),
            ToolCall(tool_call_id='tc2', tool_name='follower', args='{}', response_usage_follows=True),
            SessionUsage(usage=RequestUsage(input_tokens=10, output_tokens=5)),
            ResponseDone(),
        ]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        _ = [e async for e in session]

    assert len(seen_steps) == 2
    assert seen_steps[0] == seen_steps[1]


def _ordered_events_agent() -> Agent[None, str]:
    """Two tools where the *second* call finishes first, so completion order != call order."""
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    async def slow() -> str:
        await asyncio.sleep(0.05)
        return 'slow'

    @agent.tool_plain
    async def quick() -> str:
        return 'quick'

    return agent


async def test_parallel_ordered_events_emits_results_in_call_order() -> None:
    # `parallel_ordered_events` promises parallel execution with events in call order. The session used
    # to read the mode only to spot `'sequential'`, so this fell through to plain parallel and results
    # streamed in completion order — the setting silently did nothing.
    agent = _ordered_events_agent()
    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='tc1', tool_name='slow', args='{}'),
            ToolCall(tool_call_id='tc2', tool_name='quick', args='{}'),
            ResponseDone(),
        ]
    )
    model = FakeRealtimeModel(conn)
    with ToolManager.parallel_execution_mode('parallel_ordered_events'):
        async with agent.realtime(model).session() as session:
            events = [e async for e in session]

    results = [e.part.tool_name for e in events if isinstance(e, FunctionToolResultEvent)]
    assert results == ['slow', 'quick']
    # Execution stays concurrent — the provider still gets each result as it lands, so the fast tool's
    # `ToolResult` goes out first even though its event is held back.
    sent = [s.tool_call_id for s in conn.sent if isinstance(s, ToolResult)]
    assert sent == ['tc2', 'tc1']


async def test_parallel_default_still_emits_results_in_completion_order() -> None:
    # The other side of the branch: the default mode is unchanged, so the ordering buffer only engages
    # where it was asked for.
    agent = _ordered_events_agent()
    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='tc1', tool_name='slow', args='{}'),
            ToolCall(tool_call_id='tc2', tool_name='quick', args='{}'),
            ResponseDone(),
        ]
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        events = [e async for e in session]

    results = [e.part.tool_name for e in events if isinstance(e, FunctionToolResultEvent)]
    assert results == ['quick', 'slow']


async def test_parallel_ordered_events_are_not_stranded_by_a_failing_sibling() -> None:
    # The buffer is released from the tool task's done-callback, which runs even when the tool raised,
    # so a sibling that produces no events can't leave the batch waiting forever.
    agent: Agent[None, str] = Agent()
    fine_finished = asyncio.Event()

    @agent.tool_plain
    async def boom() -> str:
        # Fail only once the sibling has fully run: `fine`'s buffered events are then waiting on this
        # first-in-order call to settle, which is exactly the state a hung release would strand.
        await fine_finished.wait()
        raise RuntimeError('tool exploded')

    @agent.tool_plain
    async def fine() -> str:
        fine_finished.set()
        return 'fine'

    conn = FakeRealtimeConnection(
        [
            ToolCall(tool_call_id='tc1', tool_name='boom', args='{}'),
            ToolCall(tool_call_id='tc2', tool_name='fine', args='{}'),
            ResponseDone(),
        ]
    )
    model = FakeRealtimeModel(conn)
    with ToolManager.parallel_execution_mode('parallel_ordered_events'):
        with pytest.raises(RuntimeError, match='tool exploded'):
            async with agent.realtime(model).session() as session:
                _ = [e async for e in session]


async def test_tool_can_cancel_realtime_session() -> None:
    """A fake connection makes the absence of a provider-bound tool result directly assertable."""
    agent = Agent[None, str](deps_type=type(None))

    @agent.tool
    def cancel(ctx: RunContext[None]) -> str:
        ctx.cancel()
        return 'must be discarded'

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='cancel', args='{}'), ResponseDone()])
    model = FakeRealtimeModel(conn)

    session: _RealtimeSession | None = None
    with pytest.raises(RunCancelled) as exc_info:
        async with agent.realtime(model).session() as session:
            _ = [e async for e in session]

    assert session is not None
    assert session.closed
    parts = [part for message in exc_info.value.all_messages() for part in message.parts]
    assert any(isinstance(part, ToolCallPart) for part in parts)
    assert any(isinstance(part, ToolReturnPart) and part.outcome == 'interrupted' for part in parts)
    assert not any(isinstance(item, ToolResult) for item in conn.sent)


async def test_realtime_cancellation_does_not_wait_for_sync_tool_worker() -> None:
    """A unit test because a recording cannot observe whether the local worker thread outlives the session."""
    worker_started = ThreadEvent()
    worker_release = ThreadEvent()
    worker_finished = ThreadEvent()
    agent = Agent[None, str](deps_type=type(None))

    @agent.tool
    def cancel_and_block(ctx: RunContext[None]) -> str:
        ctx.cancel()
        worker_started.set()
        worker_release.wait()
        worker_finished.set()
        return 'must be discarded'

    conn = FakeRealtimeConnection(
        [ToolCall(tool_call_id='tc', tool_name='cancel_and_block', args='{}'), ResponseDone()]
    )

    async def consume() -> None:
        async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
            _ = [event async for event in session]

    task = asyncio.create_task(consume())
    await asyncio.to_thread(worker_started.wait)
    try:
        with pytest.raises(RunCancelled):
            await asyncio.wait_for(asyncio.shield(task), timeout=1)
        assert not worker_finished.is_set()
        assert not any(isinstance(item, ToolResult) for item in conn.sent)
    finally:
        worker_release.set()
        assert await asyncio.to_thread(worker_finished.wait, 5)
        await asyncio.gather(task, return_exceptions=True)


async def test_nested_run_cancellation_is_isolated_into_a_failed_tool_return() -> None:
    # A sub-agent run awaited inside a tool that cancels *itself* must not take the session with it.
    # The graph isolates exactly this into a failed tool return (#7199); the session's own cancellation
    # arrives as `CancelledError`, so a `RunCancelled` seen in a tool body is always a nested run's.
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def delegate() -> str:
        raise RunCancelled('the sub-agent run was cancelled')

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='delegate', args='{}'), ResponseDone()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        events = [e async for e in session]

    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, ToolReturnPart)
    assert result.part.outcome == 'failed'
    assert 'sub-agent run was cancelled' in str(result.part.content)
    # The session drained normally rather than re-raising, and the failed return is in history for the
    # model to react to — the conversation survives the sub-agent's decision.
    assert result.part in session.all_messages()[-1].parts


async def test_realtime_session_explains_a_declaratively_external_tool() -> None:
    # `kind='external'` diverged the same way, differently: `execute_tool_call` refuses outright with
    # `RuntimeError('External tools cannot be called')`, which would have escaped the session as an
    # internal error. Classified with the same helper, it now takes the `CallDeferred` path and gets
    # the documented explanation instead.
    agent: Agent[None, str] = Agent(
        toolsets=[ExternalToolset(tool_defs=[ToolDefinition(name='lookup', kind='external')])]
    )

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='lookup', args='{}'), ResponseDone()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        events = [e async for e in session]

    assert not any(isinstance(e, RealtimeSessionErrorEvent) for e in events)
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, ToolReturnPart)
    assert 'runs externally' in str(result.part.content)
    assert 'cannot be completed during a realtime session' in str(result.part.content)


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

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='whoami', args='{}'), ResponseDone()])
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

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='inspect_model', args='{}'), ResponseDone()])
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

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='tc', tool_name='inspect_model', args='{}'), ResponseDone()])
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
        [ToolCall(tool_call_id='bg', tool_name='slow_lookup', args='{}'), ResponseDone()],
        release=release,
    )
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        events = [e async for e in session]

    assert [type(e).__name__ for e in events] == snapshot(
        [
            'PartStartEvent',
            'PartEndEvent',
            'FunctionToolCallEvent',
            'FunctionToolResultEvent',
        ]
    )
    result = events[-1]
    assert isinstance(result, FunctionToolResultEvent)
    assert result.part.content == 'background result'


async def test_agent_realtime_session_forwards_model_settings() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([ResponseDone()])
    model = FakeRealtimeModel(conn)
    settings = RealtimeModelSettings(max_tokens=50)
    async with agent.realtime(model, model_settings=settings).session() as session:
        _ = [e async for e in session]
    assert model.last_model_settings == settings


async def test_agent_realtime_session_merges_model_and_call_settings() -> None:
    """Call-time realtime settings override the model defaults key by key."""
    agent: Agent[None, str] = Agent(model_settings=ModelSettings(temperature=0.1, max_tokens=100))
    conn = FakeRealtimeConnection([ResponseDone()])
    model = FakeRealtimeModel(conn, settings=RealtimeModelSettings(max_tokens=100, parallel_tool_calls=False))
    async with agent.realtime(
        model, model_settings=RealtimeModelSettings(parallel_tool_calls=True)
    ).session() as session:
        _ = [e async for e in session]
    assert model.last_model_settings == RealtimeModelSettings(max_tokens=100, parallel_tool_calls=True)


async def test_agent_realtime_session_ignores_regular_model_settings_override() -> None:
    """`Agent.override(model_settings=...)` does not affect realtime settings."""
    agent: Agent[None, str] = Agent(model_settings=ModelSettings(temperature=0.1))
    conn = FakeRealtimeConnection([ResponseDone()])
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
    assert conn.sent == [BinaryAudio(data=b'\xab\xcd', media_type='audio/pcm')]


# --- parity with run/iter: instructions, toolsets, usage, usage_limits, capabilities, metadata ---


async def test_agent_realtime_session_dynamic_instructions() -> None:
    agent: Agent[None, str] = Agent(instructions='Base')

    @agent.instructions
    def extra() -> str:
        return 'Dynamic'

    @agent.instructions
    def skipped() -> str | None:
        return None  # a dynamic instruction returning None contributes nothing

    conn = FakeRealtimeConnection([ResponseDone()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        _ = [e async for e in session]
    # Static literal then dynamic function, double-newline separated — same as `run`/`iter`.
    assert model.last_instructions == 'Base\n\nDynamic'


async def test_agent_realtime_session_sorts_static_instructions_before_dynamic() -> None:
    """A static toolset `InstructionPart` sorts before dynamic agent instructions, like `run`/`iter`.

    Regression: the session joined instruction parts in assembly order, so a toolset's static
    instruction landed after dynamic agent instruction functions instead of in the cacheable
    static prefix that classic runs build via `InstructionPart.sorted`.
    """

    class StaticInstructionsToolset(FunctionToolset[object]):
        async def get_instructions(self, ctx: RunContext[object]) -> list[InstructionPart]:
            return [InstructionPart(content='Static toolset', dynamic=False)]

    agent: Agent[None, str] = Agent(instructions='Base')

    @agent.instructions
    def extra() -> str:
        return 'Dynamic'

    conn = FakeRealtimeConnection([ResponseDone()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model, toolsets=[StaticInstructionsToolset()]).session() as session:
        _ = [e async for e in session]
    assert model.last_instructions == 'Base\n\nStatic toolset\n\nDynamic'


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
    conn = FakeRealtimeConnection([ResponseDone()])
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

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t1', tool_name='extra_tool', args='{}'), ResponseDone()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model, toolsets=[extra_toolset]).session() as session:
        events = [e async for e in session]
    assert 'extra_tool' in [t.name for t in model.last_tools or []]
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert result.part.content == 'x'  # the extra toolset's tool is offered AND callable


async def test_agent_realtime_session_external_usage_accumulates() -> None:
    usage = RunUsage()
    conn = FakeRealtimeConnection([SessionUsage(usage=RequestUsage(input_tokens=7, output_tokens=3)), ResponseDone()])
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
            SessionUsage(usage=RequestUsage(details={'input_transcription_seconds': 3}), response_scoped=False),
            OutputTranscript(text='second response', is_final=True),
            ResponseDone(),
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
        [SessionUsage(usage=RequestUsage(input_tokens=100, output_tokens=100)), ResponseDone()]
    )
    model = FakeRealtimeModel(conn)
    agent: Agent[None, str] = Agent()
    async with agent.realtime(model, usage_limits=UsageLimits(total_tokens_limit=50)).session() as session:
        with pytest.raises(UsageLimitExceeded, match='Exceeded the total_tokens_limit of 50'):
            _ = [e async for e in session]


async def test_agent_realtime_session_cost_limit_raises_on_usage() -> None:
    conn = FakeRealtimeConnection([SessionUsage(usage=RequestUsage(cost=Decimal('0.51'))), ResponseDone()])
    agent: Agent[None, str] = Agent()
    async with agent.realtime(
        FakeRealtimeModel(conn), usage_limits=UsageLimits(cost_limit=Decimal('0.50'))
    ).session() as session:
        with pytest.raises(UsageLimitExceeded, match=r'Exceeded the `cost_limit` of 0.50'):
            _ = [e async for e in session]


async def test_when_idle_enqueue_after_pump_finishes_is_delivered() -> None:
    conn = FakeRealtimeConnection([])
    session = RealtimeSession(conn)
    async with session:
        assert await drain_events(session) == []
        session._pending_messages.append(  # pyright: ignore[reportPrivateUsage]
            PendingMessage(
                messages=[ModelRequest(parts=[UserPromptPart(content='late idle message')])],
                priority='when_idle',
            )
        )
        for _ in range(10):  # pragma: no branch - the queued drain lands within a few loop passes
            if conn.sent:
                break
            await asyncio.sleep(0)

    assert conn.sent == ['late idle message']


def test_finalized_response_terminal_does_not_begin_another_response(monkeypatch: pytest.MonkeyPatch) -> None:
    session = RealtimeSession(FakeRealtimeConnection([]))
    session._response_finalized_before_terminal = True  # pyright: ignore[reportPrivateUsage]
    begins = 0

    def begin() -> None:  # pragma: no cover — the test asserts this is never called
        nonlocal begins
        begins += 1

    monkeypatch.setattr(session, '_begin_response', begin)
    session._translate_event(ResponseDone())  # pyright: ignore[reportPrivateUsage]

    assert begins == 0


async def test_agent_realtime_session_per_request_input_token_limit_raises() -> None:
    conn = FakeRealtimeConnection(
        [SessionUsage(usage=RequestUsage(input_tokens=51), response_scoped=True), ResponseDone()]
    )
    agent: Agent[None, str] = Agent()
    async with agent.realtime(
        FakeRealtimeModel(conn), usage_limits=UsageLimits(per_request_input_tokens_limit=50)
    ).session() as session:
        with pytest.raises(UsageLimitExceeded, match='per_request_input_tokens_limit of 50'):
            _ = [e async for e in session]


async def test_agent_realtime_session_request_limit_raises() -> None:
    conn = FakeRealtimeConnection(
        [
            SessionUsage(usage=RequestUsage(input_tokens=1, output_tokens=1)),
            OutputTranscript(text='first', is_final=True),
            ResponseDone(),
            SessionUsage(usage=RequestUsage(input_tokens=1, output_tokens=1)),
            OutputTranscript(text='second', is_final=True),
            ResponseDone(),
        ]
    )
    model = FakeRealtimeModel(conn)
    agent: Agent[None, str] = Agent()
    async with agent.realtime(model, usage_limits=UsageLimits(request_limit=1)).session() as session:
        events: list[RealtimeEvent] = []
        with pytest.raises(UsageLimitExceeded, match='next request would exceed the request_limit of 1'):
            async for event in session:
                events.append(event)
    assert sum(isinstance(event, RealtimeTurnCompleteEvent) for event in events) == 1
    assert session.usage.requests == 1


async def test_agent_realtime_session_request_limit_blocks_explicit_send() -> None:
    conn = FakeRealtimeConnection([])
    agent: Agent[None, str] = Agent()
    async with agent.realtime(FakeRealtimeModel(conn), usage_limits=UsageLimits(request_limit=0)).session() as session:
        with pytest.raises(UsageLimitExceeded, match='next request would exceed the request_limit of 0'):
            await session.send('hello')

    assert conn.sent == []


async def test_agent_realtime_session_request_limit_blocks_create_response() -> None:
    conn = FakeRealtimeConnection([])
    agent: Agent[None, str] = Agent()
    async with agent.realtime(FakeRealtimeModel(conn), usage_limits=UsageLimits(request_limit=0)).session() as session:
        with pytest.raises(UsageLimitExceeded, match='next request would exceed the request_limit of 0'):
            await session.create_response()

    assert conn.sent == []


async def test_agent_realtime_session_request_limit_blocks_tool_result_response() -> None:
    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t1', tool_name='greet', args='{}')])
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def greet() -> str:
        return 'hi'

    async with agent.realtime(FakeRealtimeModel(conn), usage_limits=UsageLimits(request_limit=1)).session() as session:
        with pytest.raises(UsageLimitExceeded, match='next request would exceed the request_limit of 1'):
            _ = [event async for event in session]

    assert conn.sent == []


async def test_agent_realtime_session_response_without_usage_counts_toward_request_limit() -> None:
    conn = FakeRealtimeConnection([OutputTranscript(text='response', is_final=True), ResponseDone()])
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

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t1', tool_name='greet', args='{}'), ResponseDone()])
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
            SessionUsage(usage=RequestUsage(input_tokens=1, output_tokens=1)),
            ToolCall(tool_call_id='t1', tool_name='greet', args='{}'),
            ResponseDone(),
        ]
    )
    model = FakeRealtimeModel(conn)
    limits = UsageLimits(total_tokens_limit=1000, tool_calls_limit=10)
    async with agent.realtime(model, usage_limits=limits).session() as session:
        events = [e async for e in session]
    assert not any(isinstance(e, RealtimeSessionErrorEvent) for e in events)
    assert any(isinstance(e, FunctionToolResultEvent) for e in events)


async def test_agent_realtime_session_resolves_conversation_id_like_a_run() -> None:
    # A session continues the conversation its history came from, and forks off it on `'new'`, exactly
    # as `run`/`iter` do — otherwise telemetry and a later handoff disagree about which conversation
    # this was, and `'new'` becomes a literal id shared by every session that passes it.
    history = [ModelRequest(parts=[UserPromptPart(content='earlier')], conversation_id='c1')]

    agent: Agent[None, str] = Agent()

    async def recorded_conversation_id(**kwargs: Any) -> str | None:
        # The resolved id is what gets stamped on everything the session records, so read it from there
        # rather than from a private attribute.
        async with agent.realtime(FakeRealtimeModel(FakeRealtimeConnection([])), **kwargs).session() as session:
            await session.send('hi')
            return session.new_messages()[0].conversation_id

    assert await recorded_conversation_id(message_history=history) == 'c1'
    forked = await recorded_conversation_id(message_history=history, conversation_id='new')
    assert forked not in (None, 'new', 'c1')
    assert await recorded_conversation_id()


async def test_agent_realtime_session_run_id_matches_a_run() -> None:
    seen_run_ids: list[str | None] = []
    agent: Agent[None, str] = Agent(deps_type=type(None))

    @agent.tool
    def record_run_id(ctx: RunContext[None]) -> str:
        seen_run_ids.append(ctx.run_id)
        return 'ok'

    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='hello', is_final=True),
            ToolCall(tool_call_id='t1', tool_name='record_run_id', args='{}'),
            OutputTranscript(text='hi', is_final=True),
            ResponseDone(),
        ]
    )
    async with agent.realtime(FakeRealtimeModel(conn), run_id='realtime-run-1').session() as session:
        _ = [event async for event in session]

    assert seen_run_ids == ['realtime-run-1']
    assert session.new_messages()
    assert all(message.run_id == 'realtime-run-1' for message in session.new_messages())


async def test_agent_realtime_session_generates_distinct_run_ids() -> None:
    agent: Agent[None, str] = Agent()

    async def recorded_run_id() -> str | None:
        async with agent.realtime(FakeRealtimeModel(FakeRealtimeConnection([]))).session() as session:
            await session.send('hi')
            return session.new_messages()[0].run_id

    first = await recorded_run_id()
    second = await recorded_run_id()
    assert first is not None
    assert second is not None
    assert first != second


@pytest.mark.parametrize('run_id', ['', 'prior-run'])
async def test_agent_realtime_session_rejects_invalid_run_id(run_id: str) -> None:
    history = [ModelRequest(parts=[UserPromptPart(content='earlier')], run_id='prior-run')]
    agent: Agent[None, str] = Agent()

    with pytest.raises(UserError, match='run_id'):
        async with agent.realtime(
            FakeRealtimeModel(FakeRealtimeConnection([])), message_history=history, run_id=run_id
        ).session():
            pass


async def test_agent_realtime_session_tool_context_matches_a_run() -> None:
    # `validation_context` and `root_capability` are both resolved from the context itself, so they're
    # easy to leave unset in a code path that builds its own. A tool validated in a session must see
    # the same Pydantic context as one validated in a run, and a capability introspecting the chain
    # must find its root rather than a `None` that contradicts a populated `ctx.capabilities`.
    seen: list[tuple[Any, Any]] = []

    agent: Agent[None, str] = Agent(deps_type=type(None), validation_context=lambda ctx: {'from': 'ctx'})

    @agent.tool
    def peek(ctx: RunContext[None]) -> str:
        seen.append((ctx.validation_context, ctx.root_capability))
        return 'ok'

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t1', tool_name='peek', args='{}'), ResponseDone()])
    async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
        _ = [e async for e in session]

    assert len(seen) == 1
    validation_context, root_capability = seen[0]
    assert validation_context == {'from': 'ctx'}
    assert root_capability is not None


async def test_agent_realtime_session_tool_context_sees_retry_override() -> None:
    seen: list[int] = []
    agent: Agent[None, str] = Agent(deps_type=type(None), retries=3)

    @agent.tool
    def peek(ctx: RunContext[None]) -> str:
        seen.append(ctx.max_retries)
        return 'ok'

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t1', tool_name='peek', args='{}'), ResponseDone()])
    with agent.override(retries={'tools': 0}):
        async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
            _ = [event async for event in session]

    assert seen == [0]


async def test_agent_realtime_session_tool_sees_conversation_so_far() -> None:
    # A tool reads `ctx.messages` to reason about the conversation, exactly as it would in a classic
    # run. The session's context is built once at connect, so the turns that happened since have to be
    # synchronized into it before each call.
    seen: list[ModelMessage] = []

    agent: Agent[None, str] = Agent(deps_type=type(None))

    @agent.tool
    def recall(ctx: RunContext[None]) -> str:
        seen.extend(ctx.messages)
        return 'ok'

    conn = FakeRealtimeConnection(
        [
            InputTranscript(text='what did I just say?', is_final=True),
            ToolCall(tool_call_id='t1', tool_name='recall', args='{}'),
            ResponseDone(),
        ]
    )
    async with agent.realtime(FakeRealtimeModel(conn)).session() as session:
        _ = [e async for e in session]

    assert [p.transcript for m in seen for p in m.parts if isinstance(p, SpeechPart)] == ['what did I just say?']


async def test_agent_realtime_session_native_tools_from_capability() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([ResponseDone()])
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
    conn = FakeRealtimeConnection([ResponseDone()])
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

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t1', tool_name='greet', args='{}'), ResponseDone()])
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

    conn = FakeRealtimeConnection([ToolCall(tool_call_id='t1', tool_name='whoami', args='{}'), ResponseDone()])
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
            ResponseDone(),
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
    conn = FakeRealtimeConnection([ResponseDone()])
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
        ModelRequest(
            parts=[UserPromptPart(content=[images[0]], timestamp=IsDatetime())],
            timestamp=IsDatetime(),
            conversation_id=IsStr(),
            run_id=IsStr(),
        ),
        ModelRequest(
            parts=[UserPromptPart(content=[images[2]], timestamp=IsDatetime())],
            timestamp=IsDatetime(),
            conversation_id=IsStr(),
            run_id=IsStr(),
        ),
    ]


async def test_agent_realtime_session_drops_auto_injected_tool_search() -> None:
    agent: Agent[None, str] = Agent()
    conn = FakeRealtimeConnection([ResponseDone()])
    model = FakeRealtimeModel(conn)
    async with agent.realtime(model).session() as session:
        _ = [e async for e in session]
    assert model.last_native_tools == []
