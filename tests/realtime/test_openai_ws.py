"""Cassette-backed tests for the OpenAI Realtime provider, exercising the real WebSocket protocol.

These complement the network-free `test_openai.py` unit tests: the fakes there pin event mapping and
send/handshake logic cheaply, while these replay recorded provider frames end-to-end through
[`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session] to prove the real protocol —
the streamed part events, the tool round-trip, and message-history seeding. Recorded once against the
live API with `--record-mode=rewrite`, then replayed offline forever.
"""

from __future__ import annotations as _annotations

import asyncio
import importlib
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryContent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RealtimeSessionErrorEvent,
    SpeechPart,
    SpeechPartDelta,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.realtime import RealtimeModelProfile, RealtimeOutputSpeechEndEvent, RealtimeTurnCompleteEvent
from pydantic_ai.usage import RunUsage

from ..conftest import IsDatetime, IsStr, try_import
from .conftest import REAL_SDP_OFFER
from .ws_cassettes import RealtimeCassette
from .ws_helpers import collapse_event_types, sent_frames_containing

with try_import() as imports_successful:
    from pydantic_ai.providers import Provider
    from pydantic_ai.realtime.openai import (
        OpenAIRealtimeConnection,
        OpenAIRealtimeModel,
        OpenAIRealtimeModelSettings,
    )

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire

_WAV_HEADER_BYTES = 44
"""Retained speech audio is a WAV file; subtract its header to compare against the PCM that was sent."""

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed'),
]


async def test_text_in_audio_out_turn(openai_ws_cassette: tuple[Provider[Any], RealtimeCassette]) -> None:
    """A text-in turn yields streamed audio+transcript parts and a classic-shaped history."""
    provider, cassette = openai_ws_cassette
    model = OpenAIRealtimeModel('gpt-realtime', provider=provider)
    agent = Agent(instructions='Answer in two or three words.')

    events: list[Any] = []
    async with agent.realtime(model).session(audio_retention='output_audio') as session:
        await session.send('Say a short greeting.')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    assert sent_frames_containing(cassette, 'Answer in two or three words.') == snapshot(
        [
            {
                'type': 'session.update',
                'session': {
                    'type': 'realtime',
                    'instructions': 'Answer in two or three words.',
                    'output_modalities': ['audio'],
                    'audio': {
                        'input': {
                            'format': {'type': 'audio/pcm', 'rate': 24000},
                            'turn_detection': {
                                'type': 'server_vad',
                                'create_response': True,
                                'interrupt_response': True,
                            },
                            'transcription': {'model': 'gpt-realtime-whisper'},
                        },
                        'output': {'format': {'type': 'audio/pcm', 'rate': 24000}},
                    },
                },
            }
        ]
    )

    messages = session.all_messages()
    assert collapse_event_types(events) == snapshot(
        ['PartStartEvent', 'PartDeltaEvent', 'PartEndEvent', 'RealtimeTurnCompleteEvent']
    )
    assert [type(m).__name__ for m in messages] == snapshot(['ModelRequest', 'ModelResponse'])
    assert messages[0] == ModelRequest(
        parts=[UserPromptPart(content='Say a short greeting.', timestamp=IsDatetime())],
        timestamp=IsDatetime(),
        conversation_id=IsStr(),
        run_id=IsStr(),
    )
    response = messages[1]
    assert isinstance(response, ModelResponse)
    assert response.model_name == 'gpt-realtime'
    part = response.parts[0]
    assert isinstance(part, SpeechPart)
    assert part.speaker == 'assistant'
    assert part.transcript == snapshot('Hi there!')
    assert isinstance(part.audio, BinaryContent)
    assert part.audio.media_type == 'audio/wav'
    assert len(part.audio.data) > 0


async def test_audio_in_server_vad_turn(
    openai_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path
) -> None:
    """A spoken user turn (audio in, server VAD) is transcribed into a user turn in history.

    The default microphone workflow — no explicit turn control, input transcription on by default —
    must land the user's turn in history, not just the assistant's reply. This is the end-to-end
    guard for the dropped-user-turn bug: without a transcription default, an audio-only turn produces
    neither an `InputTranscript` nor a retained recording, so `all_messages()` would hold only the
    assistant response.

    It also guards live-transcript *attribution*. OpenAI transcribes the user's audio asynchronously,
    so this recording interleaves the two speakers' transcripts delta by delta. Each delta must be
    attributable on its own — via `SpeechPartDelta.speaker` — and the two concurrently-assembling
    parts must hold distinct indices (they were both 0 once, which made the interleaved stream
    impossible to split).
    """
    provider, _ = openai_ws_cassette
    model = OpenAIRealtimeModel('gpt-realtime', provider=provider)
    agent = Agent(instructions='Reply in a few words.')
    pcm = assets_path.joinpath('marcelo_24khz.pcm').read_bytes()

    events: list[Any] = []
    async with agent.realtime(model).session() as session:
        # Stream the clip in ~100 ms chunks like a live mic; the trailing silence lets server VAD end
        # the turn without any manual `commit_audio()`.
        for start in range(0, len(pcm), 4800):
            await session.send_audio(pcm[start : start + 4800])
        with anyio.fail_after(45):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    # Pin the canonical spoken-turn event order: speech start -> stop -> user turn -> assistant reply.
    assert collapse_event_types(events) == snapshot(
        [
            'RealtimeInputSpeechStartEvent',
            'RealtimeInputSpeechEndEvent',
            'PartStartEvent',
            'PartDeltaEvent',
            'PartStartEvent',
            'PartDeltaEvent',
            'PartEndEvent',
            'PartDeltaEvent',
            'PartEndEvent',
            'RealtimeTurnCompleteEvent',
        ]
    )

    # Reassembling both transcripts from the deltas alone — no correlating back to a `PartStartEvent` —
    # recovers each speaker's turn intact, which is only possible if every delta names its speaker.
    speakers_by_index = {
        e.index: e.part.speaker for e in events if isinstance(e, PartStartEvent) and isinstance(e.part, SpeechPart)
    }
    streamed = {'user': '', 'assistant': ''}
    for event in events:
        if isinstance(event, PartDeltaEvent) and isinstance(delta := event.delta, SpeechPartDelta):
            if delta.transcript_delta:
                assert delta.speaker is not None
                # The delta agrees with the part it belongs to, so either is usable on its own.
                assert delta.speaker == speakers_by_index[event.index]
                streamed[delta.speaker] += delta.transcript_delta
    # Two parts assembled at once, on distinct indices, one per speaker.
    assert sorted(speakers_by_index.values()) == ['assistant', 'user']
    assert streamed['user'].strip() == snapshot('Hello, my name is Marcelo.')
    # The model says a curly apostrophe, normalized here to a straight one: the smartquote
    # pre-commit hook rewrites a literal curly quote in source, which would break the comparison.
    assert streamed['assistant'].strip().replace('\u2019', "'") == snapshot(
        'Hello, Marcelo! Great to meet you. How can I help you today?'
    )

    messages = session.all_messages()
    # The spoken turn is transcribed into a user request ahead of the assistant's reply.
    assert [type(m).__name__ for m in messages] == snapshot(['ModelRequest', 'ModelResponse'])
    user_turn = messages[0]
    assert isinstance(user_turn, ModelRequest)
    user_part = user_turn.parts[0]
    assert isinstance(user_part, SpeechPart)
    assert user_part.speaker == 'user'
    assert user_part.transcript == snapshot('Hello, my name is Marcelo.')
    reply = messages[1]
    assert isinstance(reply, ModelResponse)
    assert isinstance(reply.parts[0], SpeechPart)
    assert session.usage == snapshot(
        RunUsage(
            input_tokens=41,
            output_tokens=164,
            input_audio_tokens=27,
            output_audio_tokens=136,
            details={
                'input_transcription_seconds': 3,
                'input_text_tokens': 14,
                'input_image_tokens': 0,
                'output_text_tokens': 28,
                'audio_tokens': 136,
            },
            requests=1,
        )
    )
    assert reply.usage.details.get('input_transcription_seconds') is None


async def test_input_audio_retention_segments_three_server_vad_turns(
    openai_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path
) -> None:
    """Every retained OpenAI input turn ends at its own server-VAD boundary.

    Each turn sends the same schedule: one clip plus two seconds of trailing silence. Server VAD cuts
    the turn partway through, so what is retained must always be *less* than the schedule — a turn
    that retains all of it is one that carried the previous turn's leftovers as its own prefix.
    """
    provider, cassette = openai_ws_cassette
    recording = not cassette.interactions
    model = OpenAIRealtimeModel('gpt-realtime', provider=provider)
    agent = Agent(instructions='Reply in two or three words.')
    pcm = assets_path.joinpath('marcelo_24khz.pcm').read_bytes() + bytes(96_000)
    retained_pcm: list[int] = []

    async with agent.realtime(model).session(audio_retention='input_audio') as session:
        turn_completed = anyio.Event()

        async def consume() -> None:
            async for event in session:  # pragma: no branch - always returns on the third turn
                if (
                    isinstance(event, PartEndEvent)
                    and isinstance(event.part, SpeechPart)
                    and event.part.speaker == 'user'
                    and event.part.audio is not None
                ):
                    retained_pcm.append(len(event.part.audio.data) - _WAV_HEADER_BYTES)
                elif isinstance(event, RealtimeTurnCompleteEvent):
                    turn_completed.set()
                    if len(retained_pcm) == 3:
                        return

        async with anyio.create_task_group() as tg:
            tg.start_soon(consume)
            for _ in range(3):
                turn_completed = anyio.Event()
                for offset in range(0, len(pcm), 4_800):
                    await session.send_audio(pcm[offset : offset + 4_800])
                    await anyio.sleep(0.1 if recording else 0)
                with anyio.fail_after(45):
                    await turn_completed.wait()
                await anyio.sleep(1 if recording else 0)

    assert len(pcm) == 359_424
    assert retained_pcm == [206_400, 321_024, 325_824]
    assert all(retained < len(pcm) for retained in retained_pcm)


async def test_tool_call_round(openai_ws_cassette: tuple[Provider[Any], RealtimeCassette]) -> None:
    """A tool call is executed by the session and its result folded back into a classic-shaped history.

    Consumed the way the docs tell you to — stopping on `RealtimeTurnCompleteEvent` — against a real recording,
    where the protocol suppresses the function-call-only `response.done` and the *only* terminal frame is
    the answer's. That makes this the case a turn boundary derived from the suppressed response's
    bookkeeping would silently swallow, leaving the loop hanging until the socket closed.
    """
    provider, cassette = openai_ws_cassette
    model = OpenAIRealtimeModel(
        'gpt-realtime', provider=provider, settings=OpenAIRealtimeModelSettings(output_modality='text')
    )
    agent = Agent(instructions='Use the get_weather tool for any weather question, then answer in one short sentence.')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return f'It is foggy and 12 degrees in {city}.'

    events: list[Any] = []
    async with agent.realtime(model).session() as session:
        await session.send('What is the weather in London?')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    # One turn boundary, after the tool round is over — not one per response.
    assert [type(event).__name__ for event in events].count('RealtimeTurnCompleteEvent') == 1
    assert sent_frames_containing(cassette, 'Look up the weather for a city.') == snapshot(
        [
            {
                'type': 'session.update',
                'session': {
                    'type': 'realtime',
                    'instructions': 'Use the get_weather tool for any weather question, then answer in one short sentence.',
                    'output_modalities': ['text'],
                    'audio': {
                        'input': {
                            'format': {'type': 'audio/pcm', 'rate': 24000},
                            'turn_detection': {
                                'type': 'server_vad',
                                'create_response': True,
                                'interrupt_response': True,
                            },
                            'transcription': {'model': 'gpt-realtime-whisper'},
                        },
                        'output': {'format': {'type': 'audio/pcm', 'rate': 24000}},
                    },
                    'tools': [
                        {
                            'type': 'function',
                            'name': 'get_weather',
                            'parameters': {
                                'additionalProperties': False,
                                'properties': {'city': {'type': 'string'}},
                                'required': ['city'],
                                'type': 'object',
                            },
                            'description': 'Look up the weather for a city.',
                        }
                    ],
                },
            }
        ]
    )

    call_events = [e for e in events if isinstance(e, FunctionToolCallEvent)]
    result_events = [e for e in events if isinstance(e, FunctionToolResultEvent)]
    assert len(call_events) == 1
    assert call_events[0].part.tool_name == 'get_weather'
    assert call_events[0].part.args_as_dict() == {'city': 'London'}
    assert len(result_events) == 1
    assert isinstance(result_events[0].part, ToolReturnPart)
    assert result_events[0].part.content == 'It is foggy and 12 degrees in London.'

    messages = session.all_messages()
    assert [type(m).__name__ for m in messages] == snapshot(
        ['ModelRequest', 'ModelResponse', 'ModelRequest', 'ModelResponse']
    )
    assert messages[0] == ModelRequest(
        parts=[UserPromptPart(content='What is the weather in London?', timestamp=IsDatetime())],
        timestamp=IsDatetime(),
        conversation_id=IsStr(),
        run_id=IsStr(),
    )
    tool_response = messages[1]
    assert isinstance(tool_response, ModelResponse)
    assert tool_response.parts == [ToolCallPart(tool_name='get_weather', args=IsStr(), tool_call_id=IsStr())]
    assert (tool_response.usage.input_tokens, tool_response.usage.output_tokens) == (63, 22)
    tool_return = messages[2]
    assert isinstance(tool_return, ModelRequest)
    assert tool_return.parts == [
        ToolReturnPart(
            tool_name='get_weather',
            content='It is foggy and 12 degrees in London.',
            tool_call_id=IsStr(),
            timestamp=IsDatetime(),
        )
    ]
    final = messages[3]
    assert isinstance(final, ModelResponse)
    assert (final.usage.input_tokens, final.usage.output_tokens) == (103, 13)
    final_part = final.parts[0]
    # The session runs in text-output modality, so the reply is a `TextPart`, not a `SpeechPart`.
    assert isinstance(final_part, TextPart)
    assert 'fog' in final_part.content.lower()

    # Usage from BOTH provider responses is accounted for. The intermediate function-call-only
    # response's `response.done` maps to no turn event (the turn isn't over), but its tokens are
    # still counted — the connection emits a `SessionUsage` for every `response.done`, so a
    # tool-calling turn reports two usage updates, not just the final text response's.
    assert session.usage.requests == 2
    assert session.usage.input_tokens > 0 and session.usage.output_tokens > 0


async def test_message_history_seeding(openai_ws_cassette: tuple[Provider[Any], RealtimeCassette]) -> None:
    """Seeded prior turns are sent on the wire and reflected in the model's reply."""
    provider, cassette = openai_ws_cassette
    model = OpenAIRealtimeModel(
        'gpt-realtime', provider=provider, settings=OpenAIRealtimeModelSettings(output_modality='text')
    )
    agent = Agent()

    history = [
        ModelRequest(parts=[UserPromptPart(content='My name is Alice and my favorite color is teal.')]),
        ModelResponse(parts=[TextPart(content='Nice to meet you, Alice!')]),
    ]

    events: list[Any] = []
    async with agent.realtime(model, message_history=history).session() as session:
        await session.send('What is my name and favorite color?')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    # A server-side rejection of the seeded items (e.g. a bad content-type shape) surfaces as a
    # `RealtimeSessionErrorEvent`; assert none occurred so a broken seed payload fails the test loudly.
    assert [event for event in events if isinstance(event, RealtimeSessionErrorEvent)] == []

    # The seeded user/assistant turns were sent as `conversation.item.create` frames on the wire.
    assert sent_frames_containing(cassette, 'My name is Alice') == snapshot(
        [
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': 'My name is Alice and my favorite color is teal.'}],
                },
            }
        ]
    )
    # The seeded assistant turn is sent as an `output_text` item (its own serialization path, distinct
    # from the user seed above), so a wrong role/item/content shape fails here rather than passing on a
    # mere substring match.
    assert sent_frames_containing(cassette, 'Nice to meet you') == snapshot(
        [
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': 'Nice to meet you, Alice!'}],
                },
            }
        ]
    )

    # `all_messages()` carries the seeded history ahead of this session's turns.
    messages = session.all_messages()
    assert messages[:2] == history
    reply = messages[-1]
    assert isinstance(reply, ModelResponse)
    reply_part = reply.parts[0]
    # Text-output modality → a `TextPart` reply.
    assert isinstance(reply_part, TextPart)
    content = reply_part.content.lower()
    assert 'alice' in content and 'teal' in content


def test_profile_allow_seeding() -> None:
    """Unit guard: the model advertises session seeding, which the seeding cassette test relies on.

    Kept as a plain unit assertion (not a cassette test) because it pins an intrinsic capability flag
    that a recording wouldn't protect.
    """
    profile = OpenAIRealtimeModel('gpt-realtime').profile
    assert profile == RealtimeModelProfile(
        supports_image_input=True,
        supports_manual_turn_control=True,
        supports_interruption=True,
        supports_output_truncation=True,
        supports_text_output=True,
        supports_session_seeding=True,
        supports_webrtc=True,
        supports_seeding_images=True,
        supports_seeding_audio=True,
        supports_thinking=False,  # GA `gpt-realtime` is not a reasoning model
        supports_async_tool_calls=True,  # the realtime models keep talking through a tool call
        supports_tool_return_schema=False,  # no native surface; opted-in schemas go into descriptions
        supported_native_tools=frozenset(),
        emits_input_speech_events=True,
        audio_input_sample_rate=24000,
        audio_output_sample_rate=24000,
    )


# Applies the provider's WebRTC answer to the media peer, connecting the audio path.
MediaConnect = Callable[[str], Awaitable[None]]


async def _no_media_to_connect(answer_sdp: str) -> None:
    """Replay's `MediaConnect`: the recorded control frames already carry the playback boundaries."""


@asynccontextmanager
async def _live_webrtc_media_peer() -> AsyncGenerator[tuple[str, MediaConnect]]:  # pragma: no cover
    """Negotiate a real WebRTC call with `aiortc`, standing in for the browser that owns the media.

    Recording only — see `_webrtc_media_peer`. `aiortc` is not a project dependency (it pulls in a
    media stack no offline test needs), so it's imported dynamically and installed ad hoc when
    recording: `uv run --with aiortc --env-file .env pytest ... --record-mode=rewrite`.
    """
    aiortc = importlib.import_module('aiortc')
    mediastreams = importlib.import_module('aiortc.mediastreams')

    # No STUN server: a server-reflexive candidate would put the recorder's own public address in the
    # cassette, and OpenAI's ICE-lite endpoint is reachable from the host candidates alone.
    pc = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=[]))
    pc.addTrack(mediastreams.AudioStreamTrack())

    @pc.on('track')
    def _drain_inbound_audio(track: Any) -> None:
        # The provider stops filling its output buffer if nobody reads the track, which would cut the
        # playback window this recording exists to capture.
        async def pump() -> None:
            while True:
                try:
                    await track.recv()
                except Exception:
                    return

        asyncio.ensure_future(pump())

    await pc.setLocalDescription(await pc.createOffer())
    while pc.iceGatheringState != 'complete':
        await anyio.sleep(0.1)

    async def connect(answer_sdp: str) -> None:
        await pc.setRemoteDescription(aiortc.RTCSessionDescription(sdp=answer_sdp, type='answer'))

    try:
        yield pc.localDescription.sdp, connect
    finally:
        await pc.close()


@asynccontextmanager
async def _webrtc_media_peer(*, recording: bool) -> AsyncGenerator[tuple[str, MediaConnect]]:
    """The browser side of a WebRTC call, as `(offer_sdp, connect)`.

    The provider reports playback boundaries (`output_audio_buffer.*`, and so the `OutputSpeech*`
    events) only while media is actually flowing: it answers a canned offer that never completes ICE
    quite happily, but then never sends a single playback frame. So a recording that is to contain
    them needs a real peer, which `aiortc` negotiates headlessly.

    Replay needs no peer at all — the cassette holds the frames, and dialling a recorded answer's
    long-dead ICE candidates would put real UDP traffic in an offline test — so it reuses the canned
    offer and connects nothing. The recorded offer is never matched on, so the two are interchangeable.
    """
    if recording:  # pragma: no cover
        async with _live_webrtc_media_peer() as peer:
            yield peer
        return
    yield REAL_SDP_OFFER, _no_media_to_connect


@pytest.mark.vcr
async def test_webrtc_sideband_text_turn(
    openai_ws_sideband_cassette: tuple[Provider[Any], RealtimeCassette],
) -> None:
    """The secure WebRTC flow end to end: relay the offer over HTTP, then run the agent over the sideband.

    The HTTP offer relay is a VCR cassette; the control WebSocket (attached by `call_id`) is a WS
    cassette. The browser's media path isn't involved — this proves the server-side control plane: the
    sideband applies the session config, runs the tool-free turn, and builds the conversation history,
    while the audio methods are unavailable because the session doesn't own the media transport.
    """
    provider, cassette = openai_ws_sideband_cassette
    model = OpenAIRealtimeModel(
        'gpt-realtime', provider=provider, settings=OpenAIRealtimeModelSettings(output_modality='text')
    )
    agent = Agent(instructions='Answer in two words.')

    answer = await model.answer_webrtc_offer(REAL_SDP_OFFER, instructions='Answer in two words.')
    assert answer.sdp.startswith('v=0')
    assert answer.session.provider_name == 'openai'
    assert answer.session.call_id.startswith('rtc_')

    events: list[Any] = []
    async with agent.realtime(model).session(provider_session=answer.session) as session:
        # The sideband doesn't own the audio transport, so the audio methods are unavailable.
        with pytest.raises(UserError, match='does not own the audio transport'):
            await session.send_audio(b'\x00\x00')

        # It sees no output-audio deltas either, which is what makes a barge-in clear the provider's
        # outbound buffer instead of clamping a truncation against a byte counter that stays zero.
        connection = session._connection  # pyright: ignore[reportPrivateUsage]
        assert isinstance(connection, OpenAIRealtimeConnection)
        assert not connection._observes_output_audio  # pyright: ignore[reportPrivateUsage]

        await session.send('Say hello.')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch - the loop always breaks on RealtimeTurnCompleteEvent
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    # The first control frame applies the session config (no `session.created` handshake wait).
    assert cassette.interactions[0].data['type'] == 'session.update'  # type: ignore[union-attr]

    assert [event for event in events if isinstance(event, RealtimeSessionErrorEvent)] == []
    messages = session.all_messages()
    assert [type(m).__name__ for m in messages] == snapshot(['ModelRequest', 'ModelResponse'])
    assert messages[0] == ModelRequest(
        parts=[UserPromptPart(content='Say hello.', timestamp=IsDatetime())],
        timestamp=IsDatetime(),
        conversation_id=IsStr(),
        run_id=IsStr(),
    )
    reply = messages[1]
    assert isinstance(reply, ModelResponse)
    assert reply.model_name == 'gpt-realtime'
    assert isinstance(reply.parts[0], TextPart)


def _span_tree(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render exported spans as nested `{name, msg, children}` dicts, ordered by start time.

    Spans are keyed by both their OTel name and their `logfire.msg` — the latter is what a user
    actually reads in Logfire (a `chat {model}` span displays as `response {model}`), so a rename of
    either can't slip through unnoticed.
    """
    by_id = {span['context']['span_id']: span for span in spans}
    children: dict[int | None, list[dict[str, Any]]] = {}
    for span in spans:
        parent = span['parent']
        parent_id = parent['span_id'] if parent is not None and parent['span_id'] in by_id else None
        children.setdefault(parent_id, []).append(span)

    def render(span: dict[str, Any]) -> dict[str, Any]:
        kids = sorted(children.get(span['context']['span_id'], []), key=lambda child: child['start_time'])
        return {
            'name': span['name'],
            'msg': span['attributes'].get('logfire.msg'),
            'children': [render(kid) for kid in kids],
        }

    return [render(root) for root in sorted(children.get(None, []), key=lambda span: span['start_time'])]


@pytest.mark.vcr
@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_webrtc_sideband_audio_turn(
    openai_ws_sideband_cassette: tuple[Provider[Any], RealtimeCassette],
    request: pytest.FixtureRequest,
    capfire: CaptureLogfire,
) -> None:
    """A sideband session in the default audio modality, where the provider reports playback.

    The sideband never sees an audio byte — the browser holds the media — so the only thing that tells
    the session when the model is *audible* is the provider's `output_audio_buffer.*` frames. This is
    the one recording that contains them (a canned offer that never completes ICE yields none, which is
    how a barge-in bug on this path went unnoticed), pinning that they arrive at all, that they bracket
    the turn as `RealtimeOutputSpeechStartEvent` / `RealtimeOutputSpeechEndEvent`, and that speech outlasts generation:
    the end event lands *after* `RealtimeTurnCompleteEvent`, which is exactly the gap the `speak` span measures
    and the reason it can't be derived from the turn spans.
    """
    provider, _ = openai_ws_sideband_cassette
    model = OpenAIRealtimeModel('gpt-realtime', provider=provider)
    # Default `InstrumentationSettings`, so the trace is the one a user gets from `logfire.configure()`.
    agent = Agent(instructions='Answer in two words.', capabilities=[Instrumentation()])

    recording = request.config.getoption('record_mode') == 'rewrite'
    async with _webrtc_media_peer(recording=recording) as (offer_sdp, connect_media):
        answer = await model.answer_webrtc_offer(offer_sdp, instructions='Answer in two words.')
        await connect_media(answer.sdp)

        events: list[Any] = []
        async with agent.realtime(model).session(provider_session=answer.session) as session:
            await session.send('Say hello.')
            with anyio.fail_after(60):
                async for event in session:  # pragma: no branch - the loop always breaks on the end event
                    events.append(event)
                    if isinstance(event, RealtimeOutputSpeechEndEvent):
                        break

    assert [event for event in events if isinstance(event, RealtimeSessionErrorEvent)] == []
    assert collapse_event_types(events) == snapshot(
        [
            'PartStartEvent',
            'PartDeltaEvent',
            'RealtimeOutputSpeechStartEvent',
            'PartEndEvent',
            'RealtimeTurnCompleteEvent',
            'RealtimeOutputSpeechEndEvent',
        ]
    )

    # The turn lands in history as a spoken assistant turn with a transcript — and only a transcript,
    # since no audio bytes reach a sideband session.
    messages = session.all_messages()
    assert [type(m).__name__ for m in messages] == snapshot(['ModelRequest', 'ModelResponse'])
    assert messages[0] == ModelRequest(
        parts=[UserPromptPart(content='Say hello.', timestamp=IsDatetime())],
        timestamp=IsDatetime(),
        conversation_id=IsStr(),
        run_id=IsStr(),
    )
    reply = messages[1]
    assert isinstance(reply, ModelResponse)
    assert reply.model_name == 'gpt-realtime'
    part = reply.parts[0]
    assert isinstance(part, SpeechPart)
    assert part.speaker == 'assistant'
    assert part.transcript == snapshot('Hello there.')
    assert part.audio is None

    # The `speak` span hangs off the session span, alongside the turn's own spans.
    assert _span_tree(capfire.exporter.exported_spans_as_dict()) == snapshot(
        [
            {
                'name': 'invoke_agent agent',
                'msg': 'agent realtime',
                'children': [
                    {'name': 'chat gpt-realtime', 'msg': 'response gpt-realtime', 'children': []},
                    {'name': 'speak gpt-realtime', 'msg': 'speak gpt-realtime', 'children': []},
                    {'name': 'model turn complete', 'msg': 'model turn complete', 'children': []},
                ],
            }
        ]
    )
