"""Cassette-backed tests for the Gemini Live provider, exercising the real WebSocket protocol.

These complement the network-free `test_google.py` unit tests: the fakes there pin event mapping and
send logic cheaply, while these replay recorded provider frames end-to-end through
[`Agent.realtime`][pydantic_ai.agent.Agent.realtime] to prove the real protocol —
the streamed part events, the tool round-trip, and message-history seeding. Gemini Live runs over the
`google-genai` SDK's WebSocket, which the cassette engine patches at `google.genai.live.ws_connect`.
Recorded once against the live API with `--record-mode=rewrite`, then replayed offline forever.
"""

from __future__ import annotations as _annotations

from pathlib import Path
from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RequestUsage, RunContext
from pydantic_ai.messages import (
    BinaryContent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    SpeechPart,
    SpeechPartDelta,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.realtime import RealtimeModelProfile, RealtimeTurnCompleteEvent

from ..conftest import IsDatetime, IsStr, try_import
from .ws_cassettes import RealtimeCassette
from .ws_helpers import collapse_event_types, sent_frames_containing

with try_import() as imports_successful:
    from pydantic_ai.providers import Provider
    from pydantic_ai.realtime.google import GoogleRealtimeModel

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
]

# The Gemini Developer API only exposes the native-audio Live model to the recording key, and it only
# produces audio output — so every scenario below runs audio-out (transcripts drive the assertions).
_MODEL = 'gemini-2.5-flash-native-audio-preview-09-2025'


async def test_audio_in_server_vad_turn(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path
) -> None:
    """A spoken user turn (audio in, automatic VAD) is transcribed into a user turn in history.

    The default microphone workflow — Gemini transcribes input natively — must land the user's turn in
    history, not just the assistant's reply (the dropped-user-turn guard).
    """
    provider, _ = gemini_ws_cassette
    model = GoogleRealtimeModel(_MODEL, provider=provider)
    agent = Agent(instructions='Reply in a few words.')
    pcm = assets_path.joinpath('marcelo_16khz.pcm').read_bytes()  # Gemini wants 16 kHz input

    events: list[Any] = []
    async with agent.realtime(model).session() as session:
        for start in range(0, len(pcm), 3200):  # ~100 ms chunks at 16 kHz
            await session.send_audio(pcm[start : start + 3200])
        with anyio.fail_after(45):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    # Pin the spoken-turn event order for this cassette (Gemini streams input transcripts natively).
    assert collapse_event_types(events) == snapshot(
        [
            'PartStartEvent',
            'PartDeltaEvent',
            'PartStartEvent',
            'PartDeltaEvent',
            'PartEndEvent',
            'RealtimeTurnCompleteEvent',
        ]
    )

    messages = session.all_messages()
    # Automatic VAD may split the clip into several short user turns; the invariant is that the spoken
    # input is transcribed into user history (not dropped) ahead of the assistant's reply.
    user_speech = [part for message in messages if isinstance(message, ModelRequest) for part in message.parts]
    assert user_speech and all(isinstance(p, SpeechPart) and p.speaker == 'user' for p in user_speech)
    assert any(isinstance(p, SpeechPart) and p.transcript for p in user_speech)  # at least one transcribed
    responses = [message for message in messages if isinstance(message, ModelResponse)]
    assert responses and isinstance(responses[-1].parts[0], SpeechPart)


async def test_text_in_audio_out_turn(gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette]) -> None:
    """A text-in turn yields streamed audio+transcript parts and a classic-shaped history."""
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel(_MODEL, provider=provider)
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
                'setup': {
                    'model': 'models/gemini-2.5-flash-native-audio-preview-09-2025',
                    'generationConfig': {'responseModalities': ['AUDIO']},
                    'systemInstruction': {'parts': [{'text': 'Answer in two or three words.'}], 'role': 'user'},
                    'inputAudioTranscription': {},
                    'outputAudioTranscription': {},
                }
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
    assert response.model_name == _MODEL
    part = response.parts[0]
    assert isinstance(part, SpeechPart)
    assert part.speaker == 'assistant'
    assert part.transcript == snapshot('Hello there.')
    assert isinstance(part.audio, BinaryContent)
    assert part.audio.media_type == 'audio/wav'
    assert len(part.audio.data) > 0

    # Reasoning (`thoughtsTokenCount`) is billed but left out of Gemini's response/total counts, so the
    # session captures it in `details` rather than dropping it.
    assert response.usage.details.get('thoughts_tokens') == snapshot(24)


async def test_tool_call_round(gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette]) -> None:
    """Gemini Live receives the tool schema and uses its deliberately unguessable parameter names.

    Both are unguessable on purpose: Live silently ignores `parametersJsonSchema`, so a tool sent that
    way is advertised with no parameters at all and the model invents plausible names — which a
    `city`-shaped argument would hide. The optional one additionally pins `nullable`, which only the
    OpenAPI-subset `Schema` can express.
    """
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel(_MODEL, provider=provider)
    agent = Agent(instructions='Use record_reading when asked to record a reading, then confirm it in one sentence.')

    @agent.tool_plain
    def record_reading(zqx_measurement: int, qbf_note: str | None = None) -> str:
        """Store the supplied sensor value."""
        return f'Recorded {zqx_measurement} ({qbf_note}).'

    events: list[Any] = []
    async with agent.realtime(model).session() as session:
        await session.send('Please record a reading of 5 with the note "steady".')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    assert sent_frames_containing(cassette, 'Store the supplied sensor value.') == snapshot(
        [
            {
                'setup': {
                    'model': 'models/gemini-2.5-flash-native-audio-preview-09-2025',
                    'generationConfig': {'responseModalities': ['AUDIO']},
                    'systemInstruction': {
                        'parts': [
                            {
                                'text': 'Use record_reading when asked to record a reading, then confirm it in one sentence.'
                            }
                        ],
                        'role': 'user',
                    },
                    'tools': [
                        {
                            'functionDeclarations': [
                                {
                                    'description': 'Store the supplied sensor value.',
                                    'name': 'record_reading',
                                    'parameters': {
                                        'properties': {
                                            'zqx_measurement': {'type': 'INTEGER'},
                                            'qbf_note': {'nullable': True, 'type': 'STRING'},
                                        },
                                        'required': ['zqx_measurement'],
                                        'type': 'OBJECT',
                                    },
                                }
                            ]
                        }
                    ],
                    'inputAudioTranscription': {},
                    'outputAudioTranscription': {},
                }
            }
        ]
    )

    call_events = [e for e in events if isinstance(e, FunctionToolCallEvent)]
    result_events = [e for e in events if isinstance(e, FunctionToolResultEvent)]
    assert len(call_events) == 1
    assert call_events[0].part.tool_name == 'record_reading'
    assert call_events[0].part.args_as_dict() == snapshot({'zqx_measurement': 5, 'qbf_note': 'steady'})
    assert len(result_events) == 1
    assert isinstance(result_events[0].part, ToolReturnPart)
    assert result_events[0].part.content == snapshot('Recorded 5 (steady).')

    messages = session.all_messages()
    assert [type(m).__name__ for m in messages] == snapshot(
        ['ModelRequest', 'ModelResponse', 'ModelRequest', 'ModelResponse']
    )
    assert messages[0] == ModelRequest(
        parts=[UserPromptPart(content='Please record a reading of 5 with the note "steady".', timestamp=IsDatetime())],
        timestamp=IsDatetime(),
        conversation_id=IsStr(),
        run_id=IsStr(),
    )
    tool_response = messages[1]
    assert isinstance(tool_response, ModelResponse)
    assert tool_response.parts == [ToolCallPart(tool_name='record_reading', args=IsStr(), tool_call_id=IsStr())]
    # Gemini's tool-call frame carries no usage metadata; the later completed turn owns the only usage
    # report the provider supplies, so the intermediate response remains honestly empty.
    assert tool_response.usage == RequestUsage()
    tool_return = messages[2]
    assert isinstance(tool_return, ModelRequest)
    assert tool_return.parts == [
        ToolReturnPart(
            tool_name='record_reading',
            content='Recorded 5 (steady).',
            tool_call_id=IsStr(),
            timestamp=IsDatetime(),
        )
    ]
    final = messages[3]
    assert isinstance(final, ModelResponse)
    final_part = final.parts[0]
    assert isinstance(final_part, SpeechPart)
    assert final_part.transcript is not None and 'record' in final_part.transcript.lower()

    # Gemini packs `turnComplete` and `usageMetadata` into the same message; the codec emits the usage
    # before the turn boundary so the session folds it into this final `ModelResponse` instead of
    # dropping it after the response was already finalized. (Regression test for usage attribution.)
    # The per-modality split is mapped too — audio bills far higher than text, so `output_audio_tokens`
    # must not be collapsed into the output total.
    assert final.usage == (
        RequestUsage(
            input_tokens=1267,
            output_tokens=103,
            input_text_tokens=1267,
            output_audio_tokens=81,
            output_text_tokens=22,
            details={
                'text_prompt_tokens': 1267,
                'text_response_tokens': 22,
                'audio_response_tokens': 81,
            },
        )
    )
    assert session.usage.total_tokens == final.usage.total_tokens


async def test_asap_enqueue_waits_for_response_boundary(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette],
) -> None:
    """An `asap` message queued by a tool does not interrupt Gemini's active spoken response."""
    provider, _ = gemini_ws_cassette
    model = GoogleRealtimeModel(_MODEL, provider=provider)
    agent: Agent[None, str] = Agent(
        deps_type=type(None),
        instructions=(
            'Call queue_followup, then say exactly "FIRST RESPONSE COMPLETE". '
            'After any later user message, say exactly "QUEUED MARKER RECEIVED".'
        ),
    )
    tool_ctx: RunContext[None] | None = None

    @agent.tool
    def queue_followup(ctx: RunContext[None]) -> str:
        nonlocal tool_ctx
        tool_ctx = ctx
        return 'armed'

    completions: list[RealtimeTurnCompleteEvent] = []
    enqueued = False
    async with agent.realtime(model).session() as session:
        await session.send('Begin.')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch
                if (
                    not enqueued
                    and isinstance(event, PartDeltaEvent)
                    and isinstance(event.delta, SpeechPartDelta)
                    and event.delta.audio_chunk
                ):
                    assert tool_ctx is not None
                    tool_ctx.enqueue('This is the queued follow-up.')
                    enqueued = True
                if isinstance(event, RealtimeTurnCompleteEvent):
                    completions.append(event)
                    if len(completions) == 2:
                        break

    assert len(completions) == 2
    transcripts = [
        part.transcript
        for message in session.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, SpeechPart)
    ]
    assert transcripts == ['FIRST RESPONSE COMPLETE', 'QUEUED MARKER RECEIVED']


async def test_message_history_seeding(gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette]) -> None:
    """Seeded prior turns are sent on the wire and reflected in the model's reply."""
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel(_MODEL, provider=provider)
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

    # The seeded turns were sent on the wire as inactive context: a single `client_content` frame
    # carrying both turns with `turnComplete` false (so Gemini doesn't respond to the seed yet). A
    # wrong role, turn ordering, or completion flag fails here rather than passing on a substring match.
    seeded = sent_frames_containing(cassette, 'My name is Alice')
    assert seeded == sent_frames_containing(cassette, 'Nice to meet you')  # one frame carries both turns
    assert seeded == snapshot(
        [
            {
                'client_content': {
                    'turns': [
                        {'parts': [{'text': 'My name is Alice and my favorite color is teal.'}], 'role': 'user'},
                        {'parts': [{'text': 'Nice to meet you, Alice!'}], 'role': 'model'},
                    ],
                    'turnComplete': False,
                }
            }
        ]
    )

    # `all_messages()` carries the seeded history ahead of this session's turns.
    messages = session.all_messages()
    assert messages[:2] == history
    reply = messages[-1]
    assert isinstance(reply, ModelResponse)
    reply_part = reply.parts[0]
    assert isinstance(reply_part, SpeechPart)
    transcript = (reply_part.transcript or '').lower()
    assert 'alice' in transcript and 'teal' in transcript


def test_profile_allow_seeding() -> None:
    """Unit guard: the model advertises session seeding, which the seeding cassette test relies on.

    Kept as a plain unit assertion (not a cassette test) because it pins an intrinsic capability flag
    that a recording wouldn't protect. Gemini Live has no manual turn control or server-side
    interruption (automatic VAD only).
    """
    profile = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest').profile
    assert profile == RealtimeModelProfile(
        supports_image_input=True,
        supports_manual_turn_control=False,
        supports_interruption=False,
        supports_output_truncation=False,
        supports_text_output=False,  # every Live model rejects a TEXT response modality
        supports_session_seeding=True,
        supports_webrtc=False,
        supports_seeding_images=True,
        supports_seeding_audio=False,
        supports_thinking=True,  # every current Gemini Live model takes a thinking config
        # Supported, not enabled: gates the opt-in `google_async_tool_calls` setting.
        supports_async_tool_calls=True,
        # Gemini Live renders an opted-in return schema natively (the declaration's `response`).
        supports_tool_return_schema=True,
        # Search grounding only: Live models reject or silently ignore code execution and URL context.
        supported_native_tools=frozenset({WebSearchTool}),
        # Gemini Live never reports user speech start/end; a UI must key off interruption events.
        emits_input_speech_events=False,
        audio_input_sample_rate=16000,
        audio_output_sample_rate=24000,
    )
