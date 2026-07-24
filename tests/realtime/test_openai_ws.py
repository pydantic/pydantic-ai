"""Cassette-backed tests for the OpenAI Realtime provider, exercising the real WebSocket protocol.

These complement the network-free `test_openai.py` unit tests: the fakes there pin event mapping and
send/handshake logic cheaply, while these replay recorded provider frames end-to-end through
[`Agent.realtime`][pydantic_ai.agent.Agent.realtime] to prove the real protocol —
the streamed part events, the tool round-trip, and message-history seeding. Recorded once against the
live API with `--record-mode=rewrite`, then replayed offline forever.
"""

from __future__ import annotations as _annotations

from pathlib import Path
from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryContent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    SpeechPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.realtime import RealtimeModelProfile, TurnCompleteEvent
from pydantic_ai.realtime._base import SessionErrorEvent
from pydantic_ai.usage import RunUsage

from ..conftest import IsDatetime, IsStr, try_import
from .ws_cassettes import RealtimeCassette
from .ws_helpers import collapse_event_types, sent_frames_containing

# A complete WebRTC SDP offer (audio + data-channel sections) that the realtime `/realtime/calls`
# endpoint accepts. Media never actually flows in a test — this is only enough for the provider to
# create the call and return a `call_id` for the sideband control connection to attach to.
SAMPLE_WEBRTC_SDP_OFFER = """v=0
o=- 3984138995 3984138995 IN IP4 192.0.2.10
s=-
t=0 0
a=group:BUNDLE 0 1
a=msid-semantic:WMS *
m=audio 55983 UDP/TLS/RTP/SAVPF 96 0 8
c=IN IP4 198.51.100.20
a=sendrecv
a=mid:0
a=rtcp-mux
a=rtpmap:96 opus/48000/2
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=ice-ufrag:JCSJ
a=ice-pwd:4VjSN9HmbfuizcxBRohAKf
a=fingerprint:sha-256 A4:48:06:9B:5F:3C:23:B7:FF:02:08:A2:30:6A:BB:39:D3:0E:5A:8D:3E:EB:B9:BA:AC:91:C0:AD:3F:A1:46:B8
a=setup:actpass
m=application 57304 UDP/DTLS/SCTP webrtc-datachannel
c=IN IP4 198.51.100.20
a=mid:1
a=sctp-port:5000
a=max-message-size:65536
a=ice-ufrag:xNB3
a=ice-pwd:imPFbPwvIpZaDrxqpdwWiI
a=fingerprint:sha-256 A4:48:06:9B:5F:3C:23:B7:FF:02:08:A2:30:6A:BB:39:D3:0E:5A:8D:3E:EB:B9:BA:AC:91:C0:AD:3F:A1:46:B8
a=setup:actpass
"""

with try_import() as imports_successful:
    from pydantic_ai.providers import Provider
    from pydantic_ai.realtime.openai import OpenAIRealtimeModel, OpenAIRealtimeModelSettings

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
            async for event in session:  # pragma: no branch - the loop always breaks on TurnCompleteEvent
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
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
        ['PartStartEvent', 'PartDeltaEvent', 'PartEndEvent', 'TurnCompleteEvent']
    )
    assert [type(m).__name__ for m in messages] == snapshot(['ModelRequest', 'ModelResponse'])
    assert messages[0] == ModelRequest(parts=[UserPromptPart(content='Say a short greeting.', timestamp=IsDatetime())])
    response = messages[1]
    assert isinstance(response, ModelResponse)
    assert response.model_name == 'gpt-realtime'
    part = response.parts[0]
    assert isinstance(part, SpeechPart)
    assert part.speaker == 'assistant'
    assert part.transcript == snapshot('Hey there! Great to chat with you.')
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
            async for event in session:  # pragma: no branch - the loop always breaks on TurnCompleteEvent
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
                    break

    # Pin the canonical spoken-turn event order: speech start -> stop -> user turn -> assistant reply.
    assert collapse_event_types(events) == snapshot(
        [
            'InputSpeechStartEvent',
            'InputSpeechEndEvent',
            'PartStartEvent',
            'PartDeltaEvent',
            'PartStartEvent',
            'PartDeltaEvent',
            'PartEndEvent',
            'PartDeltaEvent',
            'PartEndEvent',
            'TurnCompleteEvent',
        ]
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
            output_tokens=105,
            input_audio_tokens=27,
            output_audio_tokens=76,
            details={
                'input_transcription_seconds': 3,
                'input_text_tokens': 14,
                'input_image_tokens': 0,
                'output_text_tokens': 29,
            },
            requests=1,
        )
    )
    assert reply.usage.details.get('input_transcription_seconds') is None


async def test_tool_call_round(openai_ws_cassette: tuple[Provider[Any], RealtimeCassette]) -> None:
    """A tool call is executed by the session and its result folded back into a classic-shaped history."""
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
            async for event in session:  # pragma: no branch - the loop always breaks on TurnCompleteEvent
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
                    break

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
        parts=[UserPromptPart(content='What is the weather in London?', timestamp=IsDatetime())]
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
    assert (final.usage.input_tokens, final.usage.output_tokens) == (96, 13)
    final_part = final.parts[0]
    # The session runs in text-output modality, so the reply is a `TextPart`, not a `SpeechPart`.
    assert isinstance(final_part, TextPart)
    assert 'fog' in final_part.content.lower()

    # Usage from BOTH provider responses is accounted for. The intermediate function-call-only
    # response's `response.done` maps to no turn event (the turn isn't over), but its tokens are
    # still counted — the connection emits a `SessionUsageEvent` for every `response.done`, so a
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
            async for event in session:  # pragma: no branch - the loop always breaks on TurnCompleteEvent
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
                    break

    # A server-side rejection of the seeded items (e.g. a bad content-type shape) surfaces as a
    # `SessionErrorEvent`; assert none occurred so a broken seed payload fails the test loudly.
    assert [event for event in events if isinstance(event, SessionErrorEvent)] == []

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
        supports_session_seeding=True,
        supports_seeding_images=True,
        supports_seeding_audio=True,
        supports_thinking=False,  # GA `gpt-realtime` is not a reasoning model
        supported_native_tools=frozenset(),
        audio_input_sample_rate=24000,
        audio_output_sample_rate=24000,
    )


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

    answer = await model.answer_webrtc_offer(SAMPLE_WEBRTC_SDP_OFFER, instructions='Answer in two words.')
    assert answer.sdp.startswith('v=0')
    assert answer.session.provider_name == 'openai'
    assert answer.session.call_id.startswith('rtc_')

    events: list[Any] = []
    async with agent.realtime(model).session(provider_session=answer.session) as session:
        # The sideband doesn't own the audio transport, so the audio methods are unavailable.
        with pytest.raises(UserError, match='does not own the audio transport'):
            await session.send_audio(b'\x00\x00')

        await session.send('Say hello.')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch - the loop always breaks on TurnCompleteEvent
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
                    break

    # The first control frame applies the session config (no `session.created` handshake wait).
    assert cassette.interactions[0].data['type'] == 'session.update'  # type: ignore[union-attr]

    assert [event for event in events if isinstance(event, SessionErrorEvent)] == []
    messages = session.all_messages()
    assert [type(m).__name__ for m in messages] == snapshot(['ModelRequest', 'ModelResponse'])
    assert messages[0] == ModelRequest(parts=[UserPromptPart(content='Say hello.', timestamp=IsDatetime())])
    reply = messages[1]
    assert isinstance(reply, ModelResponse)
    assert reply.model_name == 'gpt-realtime'
    assert isinstance(reply.parts[0], TextPart)
