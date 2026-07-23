"""Cassette-backed tests for the Gemini Live provider, exercising the real WebSocket protocol.

These complement the network-free `test_google.py` unit tests: the fakes there pin event mapping and
send logic cheaply, while these replay recorded provider frames end-to-end through
[`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session] to prove the real protocol —
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

from pydantic_ai import Agent, RequestUsage
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
from pydantic_ai.native_tools import CodeExecutionTool, WebFetchTool, WebSearchTool
from pydantic_ai.realtime import RealtimeModelProfile, TurnCompleteEvent

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
    async with agent.realtime_session(model=model) as session:
        for start in range(0, len(pcm), 3200):  # ~100 ms chunks at 16 kHz
            await session.send_audio(pcm[start : start + 3200])
        with anyio.fail_after(45):
            async for event in session:  # pragma: no branch - the loop always breaks on TurnCompleteEvent
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
                    break

    # Pin the spoken-turn event order for this cassette (Gemini streams input transcripts natively).
    assert collapse_event_types(events) == snapshot(
        ['PartStartEvent', 'PartDeltaEvent', 'PartStartEvent', 'PartDeltaEvent', 'PartEndEvent', 'TurnCompleteEvent']
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
    async with agent.realtime_session(model=model, audio_retention='output_audio') as session:
        await session.send('Say a short greeting.')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch - the loop always breaks on TurnCompleteEvent
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
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
        ['PartStartEvent', 'PartDeltaEvent', 'PartEndEvent', 'TurnCompleteEvent']
    )
    assert [type(m).__name__ for m in messages] == snapshot(['ModelRequest', 'ModelResponse'])
    assert messages[0] == ModelRequest(parts=[UserPromptPart(content='Say a short greeting.', timestamp=IsDatetime())])
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
    assert response.usage.details.get('thoughts_tokens') == snapshot(23)


async def test_tool_call_round(gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette]) -> None:
    """A tool call is executed by the session and its result folded back into a classic-shaped history."""
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel(_MODEL, provider=provider)
    agent = Agent(instructions='Use the get_weather tool for any weather question, then answer in one short sentence.')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return f'It is foggy and 12 degrees in {city}.'

    events: list[Any] = []
    async with agent.realtime_session(model=model) as session:
        await session.send('What is the weather in London?')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch - the loop always breaks on TurnCompleteEvent
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
                    break

    assert sent_frames_containing(cassette, 'Look up the weather for a city.') == snapshot(
        [
            {
                'setup': {
                    'model': 'models/gemini-2.5-flash-native-audio-preview-09-2025',
                    'generationConfig': {'responseModalities': ['AUDIO']},
                    'systemInstruction': {
                        'parts': [
                            {
                                'text': 'Use the get_weather tool for any weather question, then answer in one short sentence.'
                            }
                        ],
                        'role': 'user',
                    },
                    'tools': [
                        {
                            'functionDeclarations': [
                                {
                                    'description': 'Look up the weather for a city.',
                                    'name': 'get_weather',
                                    'parameters_json_schema': {
                                        'additionalProperties': False,
                                        'properties': {'city': {'type': 'string'}},
                                        'required': ['city'],
                                        'type': 'object',
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
    # Gemini's tool-call frame carries no usage metadata; the later completed turn owns the only usage
    # report the provider supplies, so the intermediate response remains honestly empty.
    assert tool_response.usage == RequestUsage()
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
    final_part = final.parts[0]
    assert isinstance(final_part, SpeechPart)
    assert final_part.transcript is not None and 'fog' in final_part.transcript.lower()

    # Gemini packs `turnComplete` and `usageMetadata` into the same message; the codec emits the usage
    # before the turn boundary so the session folds it into this final `ModelResponse` instead of
    # dropping it after the response was already finalized. (Regression test for usage attribution.)
    # The per-modality split is mapped too — audio bills far higher than text, so `output_audio_tokens`
    # must not be collapsed into the output total.
    assert final.usage == snapshot(
        RequestUsage(
            input_tokens=1203,
            output_tokens=80,
            output_audio_tokens=68,
            details={'input_text_tokens': 1203, 'output_text_tokens': 12},
        )
    )
    assert session.usage.total_tokens == final.usage.total_tokens


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
    async with agent.realtime_session(model=model, message_history=history) as session:
        await session.send('What is my name and favorite color?')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch - the loop always breaks on TurnCompleteEvent
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
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
    profile = GoogleRealtimeModel().profile
    assert profile == RealtimeModelProfile(
        supports_image_input=True,
        supports_manual_turn_control=False,
        supports_interruption=False,
        supports_output_truncation=False,
        supports_session_seeding=True,
        supports_seeding_images=True,
        supports_seeding_audio=False,
        supports_thinking=True,  # the default native-audio model supports a thinking config
        supported_native_tools=frozenset({WebSearchTool, WebFetchTool, CodeExecutionTool}),
        audio_input_sample_rate=16000,
        audio_output_sample_rate=24000,
        owns_media=True,
    )
