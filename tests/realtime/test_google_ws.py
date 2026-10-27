"""Cassette-backed tests for the Gemini Live provider, exercising the real WebSocket protocol.

These complement the network-free `test_google.py` unit tests: the fakes there pin event mapping and
send logic cheaply, while these replay recorded provider frames end-to-end through
[`Agent.realtime`][pydantic_ai.agent.Agent.realtime] to prove the real protocol —
the streamed part events, the tool round-trip, and message-history seeding. Gemini Live runs over the
`google-genai` SDK's WebSocket, which the cassette engine patches at `google.genai.live.ws_connect`.
Recorded once against the live API with `--record-mode=rewrite`, then replayed offline forever.
"""

from __future__ import annotations as _annotations

import asyncio
import io
import json
import os
import wave
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any

import anyio
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, StringConstraints

from pydantic_ai import Agent, RequestUsage, RunContext, ToolReturn
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    BinaryContent,
    BinaryImage,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    SpeechPart,
    SpeechPartDelta,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.realtime import (
    RealtimeError,
    RealtimeResponseInterruptedEvent,
    RealtimeSession,
    RealtimeTurnCompleteEvent,
)

from ..conftest import IsDatetime, IsStr, try_import
from .ws_cassettes import CassetteMessage, RealtimeCassette
from .ws_helpers import collapse_event_types, sent_frames_containing

with try_import() as imports_successful:
    from pydantic_ai.providers import Provider
    from pydantic_ai.realtime.google import GoogleRealtimeModel, GoogleRealtimeModelProfile, GoogleRealtimeModelSettings

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
]

# The Gemini Developer API only exposes the native-audio Live model to the recording key, and it only
# produces audio output — so every scenario below runs audio-out (transcripts drive the assertions).
_MODEL = 'gemini-2.5-flash-native-audio-preview-09-2025'

# The reasoning Live model, which differs from every other one in three ways the adapter has to
# handle: it requires a thinking level, rejects blocking function declarations, and rejects the
# function-response scheduling the async path otherwise sends.
_EXTENDED_THINKING_MODEL = 'gemini-3.8-live-extended-thinking'


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


async def test_input_transcription_off_keeps_user_words_out_of_history(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path
) -> None:
    """With input transcription off, a Gemini 3.x model's own transcript of the user stays out of history.

    The 3.x Live models transcribe the user's speech even when the setup leaves out
    `inputAudioTranscription` (the recording has the `inputTranscription` frames), so the setting is
    honored on our side: the spoken turn lands as a content-less placeholder.
    """
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel('gemini-3.8-live', provider=provider)
    agent = Agent(instructions='Reply in a few words.')
    pcm = assets_path.joinpath('marcelo_16khz.pcm').read_bytes()

    async with agent.realtime(model, model_settings={'input_transcription_model': None}).session() as session:
        for start in range(0, len(pcm), 3200):  # ~100 ms chunks at 16 kHz
            await session.send_audio(pcm[start : start + 3200])
        with anyio.fail_after(45):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    [setup] = sent_frames_containing(cassette, 'Reply in a few words.')
    assert 'inputAudioTranscription' not in setup['setup']
    received = [
        json.dumps(message.data)
        for message in cassette.interactions
        if isinstance(message, CassetteMessage) and message.direction == 'received'
    ]
    assert any('inputTranscription' in frame for frame in received)

    messages = session.all_messages()
    user_parts = [part for message in messages if isinstance(message, ModelRequest) for part in message.parts]
    assert user_parts == snapshot([SpeechPart(speaker='user')])
    assert isinstance(messages[-1], ModelResponse)


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


async def test_text_only_model_answers_in_text_by_default(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette],
) -> None:
    """A model whose profile says it can't speak gets a `TEXT` session by default, and its answer is a `TextPart`.

    `gemini-robotics-er-2-streaming-preview` rejects `AUDIO` and streams its written answer as an output
    transcription.
    """
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel('gemini-robotics-er-2-streaming-preview', provider=provider)
    agent = Agent(instructions='Answer in one short sentence.')

    async with agent.realtime(model).session() as session:
        await session.send('What is the capital of France?')
        with anyio.fail_after(30):
            await session.wait_for_reply()

    [setup] = sent_frames_containing(cassette, 'Answer in one short sentence.')
    assert setup['setup']['generationConfig'] == {'responseModalities': ['TEXT']}
    response = session.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.parts == snapshot([TextPart(content='The capital of France is Paris.')])


async def test_rejected_config_raises_realtime_error(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette],
) -> None:
    """A config Gemini rejects closes the socket during setup, and that close raises `RealtimeError`.

    The close carries a WebSocket close code (`1007`), not an HTTP status, so it isn't a `ModelHTTPError`.
    """
    provider, _ = gemini_ws_cassette
    model = GoogleRealtimeModel(_MODEL, provider=provider)
    with pytest.raises(RealtimeError) as exc_info:
        async with Agent().realtime(model, model_settings=GoogleRealtimeModelSettings(google_voice='alloy')).session():
            pass  # pragma: no cover
    assert not isinstance(exc_info.value, ModelHTTPError)
    assert exc_info.value.message == snapshot(
        "Gemini Live connection closed: 1007 None. Requested voice api_name 'alloy' is not available for model models/gemini-2.5-flash-native-audio-preview-09-2025"
    )


async def test_proactive_audio_on_the_default_client(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette],
) -> None:
    """`google_proactive_audio` opens and answers on the provider's default client.

    `proactivity` is served on `v1alpha` only, and the provider's client is on the SDK default `v1beta`, so the
    Live connection dials `v1alpha` itself; the setup frame carries the field.
    """
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel(_MODEL, provider=provider)
    agent = Agent(instructions='Answer in two or three words.')

    settings = GoogleRealtimeModelSettings(google_proactive_audio=True)
    async with agent.realtime(model, model_settings=settings).session() as session:
        await session.send('Say a short greeting.')
        with anyio.fail_after(30):
            await session.wait_for_reply()

    [setup] = sent_frames_containing(cassette, 'proactivity')
    assert setup['setup']['proactivity'] == {'proactive_audio': True}
    response = session.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    part = response.parts[0]
    assert isinstance(part, SpeechPart) and part.transcript


@pytest.mark.parametrize('model_name', [_MODEL, 'gemini-3.8-live'])
async def test_image_then_typed_question(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path, model_name: str
) -> None:
    """An image sent right before a typed question is seen.

    The image goes out as a video frame, which these models' typed turns don't see (3.8 answers that
    it can't see an image, 2.5 misreads it), so the question's client content carries it again.
    """
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel(model_name, provider=provider)
    agent = Agent(instructions='Answer in one short sentence.')
    image = BinaryImage(data=assets_path.joinpath('kiwi.jpg').read_bytes(), media_type='image/jpeg')

    async with agent.realtime(model).session() as session:
        await session.send(image)
        await session.send('What fruit is in the image?')
        with anyio.fail_after(45):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    [video] = sent_frames_containing(cassette, '"video"')
    [question] = sent_frames_containing(cassette, 'What fruit is in the image?')
    assert [list(part) for part in question['client_content']['turns'][0]['parts']] == [['inlineData'], ['text']]
    sent = [message.data for message in cassette.interactions if isinstance(message, CassetteMessage)]
    assert sent.index(video) < sent.index(question)

    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == ['ModelRequest', 'ModelRequest', 'ModelResponse']
    response = messages[-1]
    assert isinstance(response, ModelResponse) and isinstance(response.parts[0], SpeechPart)
    assert 'kiwi' in (response.parts[0].transcript or '').lower()


@pytest.mark.parametrize('model_name', [_MODEL, 'gemini-3.8-live'])
async def test_image_then_spoken_question(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path, model_name: str
) -> None:
    """An image sent right before a spoken question is seen, as the video frame it goes out as."""
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel(model_name, provider=provider)
    agent = Agent(instructions='Answer in one short sentence.')
    image = BinaryImage(data=assets_path.joinpath('kiwi.jpg').read_bytes(), media_type='image/jpeg')
    # The question, then a second of silence so voice activity detection ends the turn.
    pcm = assets_path.joinpath('what_fruit_is_in_the_image_16khz.pcm').read_bytes() + bytes(32000)

    async with agent.realtime(model).session() as session:
        await session.send(image)
        for start in range(0, len(pcm), 3200):  # ~100 ms chunks at 16 kHz
            await session.send_audio(pcm[start : start + 3200])
        with anyio.fail_after(45):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    realtime_inputs = [
        next(iter(frame['realtime_input']))
        for frame in sent_frames_containing(cassette, 'realtime_input')
        if 'realtime_input' in frame
    ]
    assert realtime_inputs[0] == 'video'
    assert set(realtime_inputs[1:]) == {'audio'}

    messages = session.all_messages()
    assert isinstance(messages[0], ModelRequest) and isinstance(messages[0].parts[0], UserPromptPart)
    response = messages[-1]
    assert isinstance(response, ModelResponse) and isinstance(response.parts[0], SpeechPart)
    assert 'kiwi' in (response.parts[0].transcript or '').lower()


async def test_web_search_turn(gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette]) -> None:
    """A Google Search turn on a native-audio model completes, with the search in history.

    Native-audio models announce a search with a `codeExecutionResult` part that no `executableCode`
    part precedes (the recording has it); that status line must not end the session. The search itself
    arrives as grounding metadata and lands as a `web_search` native tool call/return pair.
    """
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel(_MODEL, provider=provider)
    agent = Agent(instructions='Answer in one short sentence.', capabilities=[WebSearch()])

    async with agent.realtime(model).session() as session:
        await session.send('Search the web: who won the most recent Formula 1 race?')
        with anyio.fail_after(45):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    received = [
        json.dumps(message.data)
        for message in cassette.interactions
        if isinstance(message, CassetteMessage) and message.direction == 'received'
    ]
    assert any('codeExecutionResult' in frame for frame in received)
    assert not any('executableCode' in frame for frame in received)
    response = session.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert [(type(part).__name__, getattr(part, 'tool_name', None)) for part in response.parts] == snapshot(
        [('NativeToolCallPart', 'web_search'), ('NativeToolReturnPart', 'web_search'), ('SpeechPart', None)]
    )


async def test_text_context_waits_for_next_turn(gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette]) -> None:
    provider, _ = gemini_ws_cassette
    model = GoogleRealtimeModel(_MODEL, provider=provider)
    agent = Agent(instructions='Answer in one short sentence.')

    async with agent.realtime(model).session() as session:
        await session.send('The visitor is called Ada.', respond=False)
        await asyncio.sleep(1)
        assert not [message for message in session.new_messages() if isinstance(message, ModelResponse)]
        await session.send('What is the visitor called?')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == snapshot(
        ['ModelRequest', 'ModelRequest', 'ModelResponse']
    )
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    part = response.parts[0]
    assert isinstance(part, SpeechPart)
    assert 'ada' in (part.transcript or '').lower()


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
    # report the provider supplies, so the intermediate response remains honestly empty: no tokens, and
    # therefore a zero price rather than an unknown one.
    assert tool_response.usage == RequestUsage(cost=Decimal('0'))
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
            cost=Decimal('0.0016495'),
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


@pytest.mark.parametrize('model_name', [_MODEL, 'gemini-3.8-live'])
async def test_tool_exception_still_answers_the_call(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette], model_name: str
) -> None:
    """A tool that raises still gets a response, so the model moves on and a later reply is answered.

    Left unanswered, the call stays open: Gemini 2.5's blocking call holds the model waiting on it, so no later
    turn is answered, and Gemini 3.8's non-blocking call is made again on every later input.
    """
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel(model_name, provider=provider)
    agent = Agent(
        instructions='Use the get_vault_code tool when asked for the vault code. Answer in one short sentence.'
    )

    @agent.tool_plain
    async def get_vault_code() -> str:
        """Return the current vault code."""
        raise RuntimeError('vault database exploded')

    async with agent.realtime(model).session() as session:
        await session.send('Please tell me the vault code.')
        with pytest.raises(RuntimeError, match='vault database exploded'), anyio.fail_after(30):
            async for _ in session:
                pass
        # The session goes on receiving: the model answers the failure, then the next question.
        completions = 0
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    completions += 1
                    if completions == 2:
                        break
                    await session.send('What is two plus two?')

    assert [frame['tool_response'] for frame in sent_frames_containing(cassette, 'tool_response')] == [
        {
            'functionResponses': [
                {
                    'id': IsStr(),
                    'name': 'get_vault_code',
                    'response': {'output': '{"error":"The tool raised an unhandled error."}'},
                }
            ]
        }
    ]
    final = session.all_messages()[-1]
    assert isinstance(final, ModelResponse)
    assert final.parts == [SpeechPart(speaker='assistant', transcript=IsStr(regex=r'(?i).*four.*'))]


@pytest.mark.parametrize(
    ('model_name', 'async_tool_calls'),
    [('gemini-3.1-flash-live-preview', False), ('gemini-3.8-live', False), ('gemini-3.8-live', True)],
)
async def test_tool_result_image(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette],
    assets_path: Path,
    model_name: str,
    async_tool_calls: bool,
) -> None:
    """An image a tool returns goes in the function response, and the model sees it.

    The tool's text says nothing about the picture, so naming the fruit means the model read the image.
    Covers a blocking call and an async one, whose result is scheduled to cut into the speech.
    """
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel(model_name, provider=provider)
    agent = Agent(instructions='Use take_photo when asked what is on the table, then answer in one short sentence.')
    image = BinaryImage(data=assets_path.joinpath('kiwi.jpg').read_bytes(), media_type='image/jpeg')

    @agent.tool_plain
    async def take_photo() -> ToolReturn:
        """Take a photo of the table."""
        return ToolReturn(return_value='Photo taken.', content=[image])

    # The exchange is over once the model answered the call's result, whether it blocked on it or not.
    async with agent.realtime(model, model_settings={'async_tool_calls': async_tool_calls}).session() as session:
        await session.send('What fruit is on the table?')
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    [response] = sent_frames_containing(cassette, 'Photo taken.')
    [function_response] = response['toolResponse']['functionResponses']
    assert [part['inlineData']['mimeType'] for part in function_response['parts']] == ['image/jpeg']
    answer = ' '.join(
        part.transcript or ''
        for message in session.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, SpeechPart)
    )
    assert 'kiwi' in answer.lower()


class _TreeNode(BaseModel):
    name: str
    children: dict[str, _TreeNode] = {}
    children_by_slug: dict[Annotated[str, StringConstraints(pattern='^[a-z-]+$')], _TreeNode] = {}


async def test_tool_with_recursive_map_values(gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette]) -> None:
    """A tool taking recursive `dict[str, Node]` fields is declared, and the model calls it.

    A `dict`'s values sit under `additionalProperties`, or `patternProperties` when its keys carry a pattern.
    Live declarations drop both, so the recursion lives only in subschemas that are dropped, and each field
    goes out as a plain object like any other `dict` field.
    """
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel('gemini-3.1-flash-live-preview', provider=provider)
    agent = Agent(instructions='Use save_tree when asked to save a tree, then confirm it in one short sentence.')

    @agent.tool_plain
    def save_tree(tree: _TreeNode) -> str:
        """Save a tree of named nodes."""
        return f'Saved {tree.name}.'

    events: list[Any] = []
    async with agent.realtime(model).session() as session:
        await session.send('Save a tree whose root is named "oak" and has no children.')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    [setup] = sent_frames_containing(cassette, 'Save a tree of named nodes.')
    assert setup['setup']['tools'] == snapshot(
        [
            {
                'functionDeclarations': [
                    {
                        'description': 'Save a tree of named nodes.',
                        'name': 'save_tree',
                        'parameters': {
                            'properties': {
                                'name': {'type': 'STRING'},
                                'children': {'default': {}, 'type': 'OBJECT'},
                                'children_by_slug': {'default': {}, 'type': 'OBJECT'},
                            },
                            'required': ['name'],
                            'type': 'OBJECT',
                        },
                    }
                ]
            }
        ]
    )
    assert [e.part.args_as_dict() for e in events if isinstance(e, FunctionToolCallEvent)] == snapshot(
        [{'name': 'oak'}]
    )
    assert [e.part.content for e in events if isinstance(e, FunctionToolResultEvent)] == snapshot(['Saved oak.'])


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


async def test_session_when_idle_enqueue_waits_for_response_boundary(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette],
) -> None:
    """A `when_idle` system prompt enqueued on the session waits for Gemini's active spoken response to finish."""
    provider, _ = gemini_ws_cassette
    model = GoogleRealtimeModel(_MODEL, provider=provider)
    agent: Agent[None, str] = Agent(
        instructions='First say exactly "FIRST RESPONSE COMPLETE". After any later message, follow its instruction exactly.',
    )

    completions: list[RealtimeTurnCompleteEvent] = []
    enqueued = False
    async with agent.realtime(model).session() as session:
        await session.send('Begin.')
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                if (
                    not enqueued
                    and isinstance(event, PartDeltaEvent)
                    and isinstance(event.delta, SpeechPartDelta)
                    and event.delta.audio_chunk
                ):
                    session.enqueue(
                        SystemPromptPart(content='Say exactly "QUEUED MARKER RECEIVED".'), priority='when_idle'
                    )
                    enqueued = True
                if isinstance(event, RealtimeTurnCompleteEvent):
                    completions.append(event)
                    if len(completions) == 2:
                        break

    assert len(completions) == 2
    assert [
        (type(part).__name__, part.content)
        for message in session.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, (UserPromptPart, SystemPromptPart))
    ] == [('UserPromptPart', 'Begin.'), ('SystemPromptPart', 'Say exactly "QUEUED MARKER RECEIVED".')]
    assert [
        part.transcript
        for message in session.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, SpeechPart)
    ] == ['FIRST RESPONSE COMPLETE', 'QUEUED MARKER RECEIVED']


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


async def test_message_history_function_part_seeding(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette],
) -> None:
    """Seeded tool calls and results go in as native function parts on 3.8, and the model reads them.

    The tool result carries a detail the model can't guess, so a correct answer shows it read the
    `function_response` rather than the text around it.
    """
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel('gemini-3.8-live', provider=provider)
    agent = Agent(instructions='Answer in one short sentence.')
    history = [
        ModelRequest(parts=[UserPromptPart(content='What is the weather in Paris?')]),
        ModelResponse(parts=[ToolCallPart(tool_name='get_weather', args={'city': 'Paris'}, tool_call_id='call_1')]),
        ModelRequest(
            parts=[ToolReturnPart(tool_name='get_weather', content='Hailing, wind code ZEBRA-7', tool_call_id='call_1')]
        ),
        ModelResponse(parts=[TextPart(content='It is hailing in Paris.')]),
    ]

    async with agent.realtime(model, message_history=history).session() as session:
        await session.send('What wind code did the weather tool return?')
        with anyio.fail_after(45):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    [setup] = sent_frames_containing(cassette, '"setup"')
    # `google-genai` sends the field in snake case, which the API accepts.
    assert setup['setup']['historyConfig'] == {'initial_history_in_client_content': True}
    [seeded] = sent_frames_containing(cassette, 'wind code ZEBRA-7')
    assert seeded == snapshot(
        {
            'client_content': {
                'turns': [
                    {'parts': [{'text': 'What is the weather in Paris?'}], 'role': 'user'},
                    {
                        'parts': [{'functionCall': {'id': 'call_1', 'args': {'city': 'Paris'}, 'name': 'get_weather'}}],
                        'role': 'model',
                    },
                    {
                        'parts': [
                            {
                                'functionResponse': {
                                    'id': 'call_1',
                                    'name': 'get_weather',
                                    'response': {'output': 'Hailing, wind code ZEBRA-7'},
                                }
                            }
                        ],
                        'role': 'user',
                    },
                    {'parts': [{'text': 'It is hailing in Paris.'}], 'role': 'model'},
                ],
                'turnComplete': True,
            }
        }
    )
    reply = session.all_messages()[-1]
    assert isinstance(reply, ModelResponse) and isinstance(reply.parts[0], SpeechPart)
    assert 'zebra' in (reply.parts[0].transcript or '').lower()


async def test_message_history_audio_seeding(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path
) -> None:
    """A seeded user turn with retained audio and no transcript is heard by a 3.x model.

    `gemini-3.8-live` takes audio in seeded turns, so the user's words reach it as audio rather than
    being refused for lack of a transcript, as they are on 2.5.
    """
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel('gemini-3.8-live', provider=provider)
    agent = Agent(instructions='Answer in one short sentence.')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(assets_path.joinpath('my_name_is_alice_16khz.pcm').read_bytes())
    history = [
        ModelRequest(
            parts=[SpeechPart(speaker='user', audio=BinaryContent(buffer.getvalue(), media_type='audio/wav'))]
        ),
        ModelResponse(parts=[TextPart(content='Nice to meet you!')]),
    ]

    async with agent.realtime(model, message_history=history).session() as session:
        await session.send('What is my name?')
        with anyio.fail_after(45):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    [seeded] = sent_frames_containing(cassette, 'Nice to meet you!')
    assert [(turn['role'], [list(part) for part in turn['parts']]) for turn in seeded['client_content']['turns']] == [
        ('user', [['inlineData']]),
        ('model', [['text']]),
    ]
    reply = session.all_messages()[-1]
    assert isinstance(reply, ModelResponse) and isinstance(reply.parts[0], SpeechPart)
    assert 'alice' in (reply.parts[0].transcript or '').lower()


@pytest.mark.usefixtures('no_genai_prices_context_window')
def test_profile_allow_seeding() -> None:
    """Unit guard: the model advertises session seeding, which the seeding cassette test relies on.

    Kept as a plain unit assertion (not a cassette test) because it pins an intrinsic capability flag
    that a recording wouldn't protect. Gemini Live has no manual turn control or server-side
    interruption (automatic VAD only).
    """
    profile = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest').profile
    assert profile == GoogleRealtimeModelProfile(
        supports_image_input=True,
        image_input_requires_response=False,
        supports_manual_turn_control=False,
        supports_interruption=False,
        supports_output_truncation=False,
        supports_text_output=False,  # the Developer API Live models reject a TEXT response modality
        supports_audio_output=True,
        supports_session_seeding=True,
        supports_system_messages=False,
        supports_webrtc=False,
        supports_seeding_images=True,
        supports_seeding_audio=False,
        supports_thinking=True,  # native-audio and 3.x Live models take a thinking config
        # The session's choice, via the `async_tool_calls` setting, which is off by default.
        async_tool_call_mode='optional',
        supports_async_tool_calls=True,  # deprecated, derived from `async_tool_call_mode`
        # Gemini Live renders an opted-in return schema natively (the declaration's `response`).
        supports_tool_return_schema=True,
        # Search grounding only: Live models reject or silently ignore code execution and URL context.
        supported_native_tools=frozenset({WebSearchTool}),
        # Gemini Live never reports user speech start/end; a UI must key off interruption events.
        emits_input_speech_events=False,
        synthesizes_turn_boundary=False,
        responses_are_requests=True,
        response_usage_covers_context=True,
        audio_input_sample_rate=16000,
        audio_output_sample_rate=24000,
        context_window=None,
        # Thinking is optional, tool calls block unless opted in, and an async result can be scheduled.
        google_thinking_always_enabled=False,
        google_async_tool_calls_by_default=False,
        google_supports_async_tool_call_scheduling=True,
        google_supports_affective_dialog=True,
        # A typed turn doesn't see an image sent just before it as a video frame (verified live).
        google_text_turns_see_video_frames=False,
        # 2.5 guesses at media in a function response, so tool results carry text only.
        google_supported_mime_types_in_tool_returns=(),
        # 2.5 rejects function parts in seeded turns, so seeded tool calls go in as text.
        google_supports_seeding_function_parts=False,
        google_closes_tool_call_turn_separately=False,
        # The native-audio models finish a reply an input reaches mid-generation, rather than cutting it off.
        google_input_cuts_off_reply=False,
    )


async def test_handle_barge_in_over_live_speech(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path
) -> None:
    """`handle_barge_in=True` against Gemini Live: only the local flush is left to do.

    Gemini reports no speech onset; it interrupts its own reply when the user speaks over it and
    says so with `RealtimeResponseInterruptedEvent`. The session's half is purely local —
    flushing buffered playback audio — so nothing barge-in-related goes out on the wire, and the
    barged-in utterance still gets a reply.
    """
    provider, _ = gemini_ws_cassette
    model = GoogleRealtimeModel(_MODEL, provider=provider)
    # A long reply is still playing when the user speaks over it, so the recording actually captures
    # the provider interrupting itself. Gemini generates faster than real time, so the cassette has
    # `generationComplete` before `interrupted`: the interruption lands during playback.
    agent = Agent(instructions='Reply with several full sentences; be expansive.')
    pcm = assets_path.joinpath('marcelo_16khz.pcm').read_bytes()

    events: list[Any] = []
    async with agent.realtime(model).session(handle_barge_in=True) as session:
        stream = session.stream_audio()
        with anyio.fail_after(90):
            for start in range(0, len(pcm), 3200):  # ~100 ms chunks at 16 kHz
                await session.send_audio(pcm[start : start + 3200])
            # Wait for the reply's audio to start flowing before speaking over it.
            assert len(await anext(stream)) > 0
            for start in range(0, len(pcm), 3200):
                await session.send_audio(pcm[start : start + 3200])
            turns_complete = 0
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    turns_complete += 1
                    if turns_complete == 2:
                        break

    assert any(isinstance(event, RealtimeResponseInterruptedEvent) for event in events)
    responses = [message for message in session.all_messages() if isinstance(message, ModelResponse)]
    assert 'interrupted' in [response.state for response in responses]


async def test_async_tool_speech_stays_before_its_result(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette], realtime_recording: bool
) -> None:
    """What the model says while an asynchronous tool runs is recorded before the tool's result.

    With `NON_BLOCKING` tools, Gemini 2.5 keeps talking in the same turn after the `toolCall` frame ("I'm
    searching…"), and finishes generating (`generationComplete`) while holding the turn open for the result. The
    result then cuts in (`scheduling: INTERRUPT`: `interrupted`, then `turnComplete` with the turn's usage), and the
    model answers it in a turn of its own. The calling response goes on past the call until the result goes out, so
    the speech is recorded with the call, ahead of the result: the order it was said in.
    """
    provider, _ = gemini_ws_cassette
    model = GoogleRealtimeModel(_MODEL, provider=provider)
    agent = Agent(
        instructions=(
            'Before calling a tool, briefly tell the user what you are doing, '
            'and keep them company while it runs. Keep replies short.'
        )
    )

    @agent.tool_plain
    async def search_flights(destination: str) -> str:
        """Search for flights to a destination. Takes a while."""
        # Only the live model needs a slow tool to talk through; replay follows the recorded frame order.
        await anyio.sleep(3 if realtime_recording else 0)
        return f'Cheapest flight to {destination}: KLM 1234 at 09:15 for 180 euros.'

    async with agent.realtime(model, model_settings={'async_tool_calls': True}).session() as session:
        await session.send('Find me a flight to Lisbon please.')
        with anyio.fail_after(60):
            await session.wait_for_reply()

    messages = session.all_messages()
    assert [
        (
            type(message).__name__,
            [part.transcript if isinstance(part, SpeechPart) else type(part).__name__ for part in message.parts],
        )
        for message in messages
    ] == snapshot(
        [
            ('ModelRequest', ['UserPromptPart']),
            ('ModelResponse', ['ToolCallPart', 'Please hold while I search for flights to Lisbon.']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelResponse', ['The cheapest flight to Lisbon is KLM 1234 at 9:15 for 180 euros.']),
        ]
    )
    calling, answer = messages[1], messages[3]
    assert isinstance(calling, ModelResponse) and isinstance(answer, ModelResponse)
    assert (calling.state, calling.finish_reason) == snapshot(('complete', 'tool_call'))
    # Gemini reports a turn's usage only as it ends: the turn the result cut off ended with nothing more said, so its
    # usage goes to the next response, as usage between responses does.
    assert [(m.usage.input_tokens, m.usage.output_tokens) for m in (calling, answer)] == snapshot([(0, 0), (3051, 241)])
    assert session.usage.requests == 2


@pytest.mark.realtime_ws_hold_open
async def test_typed_turn_while_an_async_tool_runs(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette], realtime_recording: bool
) -> None:
    """A typed turn sent while the model talks on past an asynchronous call is recorded after what it had said.

    The model finishes its sentence and ends its turn, which ends the calling response; the typed turn joins the
    conversation after it, and is answered in a turn of its own while the tool runs. The tool's result, sent later,
    still directly follows its call.
    """
    provider, _ = gemini_ws_cassette
    model = GoogleRealtimeModel(_MODEL, provider=provider)
    agent = Agent(
        instructions=(
            'Before calling a tool, briefly tell the user what you are doing, '
            'and keep them company while it runs. Keep replies short.'
        )
    )
    typed = anyio.Event()

    @agent.tool_plain
    async def search_flights(destination: str) -> str:
        """Search for flights to a destination. Takes a while."""
        await typed.wait()
        # Only the live model needs a slow tool; replay follows the recorded frame order.
        await anyio.sleep(8 if realtime_recording else 0)
        return f'Cheapest flight to {destination}: KLM 1234 at 09:15 for 180 euros.'

    speaking = anyio.Event()
    called = False

    async def watch(session: RealtimeSession) -> None:
        nonlocal called
        async for event in session:  # pragma: no branch
            if isinstance(event, FunctionToolCallEvent):
                called = True
            elif called and isinstance(event, PartDeltaEvent) and isinstance(event.delta, SpeechPartDelta):
                speaking.set()

    async with (
        agent.realtime(model, model_settings={'async_tool_calls': True}).session() as session,
        anyio.create_task_group() as tasks,
    ):
        tasks.start_soon(watch, session)
        await session.send('Find me a flight to Lisbon please.')
        with anyio.fail_after(60):
            await speaking.wait()
            await session.send('While you search: what is the capital of Portugal? Answer in one word.')
            typed.set()
            await session.wait_for_reply()
        tasks.cancel_scope.cancel()

    assert [
        (
            type(message).__name__,
            [part.transcript if isinstance(part, SpeechPart) else type(part).__name__ for part in message.parts],
        )
        for message in session.all_messages()
    ] == snapshot(
        [
            ('ModelRequest', ['UserPromptPart']),
            ('ModelResponse', ['ToolCallPart', 'Please hold on, I am searching for flights to Lisbon for you.']),
            ('ModelRequest', ['ToolReturnPart']),
            ('ModelRequest', ['UserPromptPart']),
            ('ModelResponse', ['Lisbon. The flight search is still running.']),
            ('ModelResponse', ['The cheapest flight to Lisbon is KLM 1234 at 9 15 AM for 180 euros.']),
        ]
    )


async def test_extended_thinking_async_tool_round(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette], realtime_recording: bool
) -> None:
    """`gemini-3.8-live-extended-thinking` speaks a filler, runs the tool in the background, then answers.

    The model has no blocking mode, so the session is async whether or not it asked, and the tool result
    goes back *without* a `scheduling` field — the two things this model rejects outright. It also
    requires a thinking level, which the session supplies on its behalf.
    """
    provider, cassette = gemini_ws_cassette
    model = GoogleRealtimeModel(_EXTENDED_THINKING_MODEL, provider=provider)
    agent = Agent(instructions='You are a flight booking assistant. Always use search_flights before answering.')

    @agent.tool_plain
    async def search_flights(origin: str, destination: str) -> str:
        """Search flights between two cities. Takes several seconds."""
        # Only the live model needs a slow tool to speak its filler; replay follows the recorded frame order.
        await anyio.sleep(5 if realtime_recording else 0)
        return 'KLM at 120 dollars'

    events: list[Any] = []
    async with agent.realtime(model).session() as session:
        await session.send('Find me a flight from Amsterdam to Lisbon, then tell me the cheapest one.')
        with anyio.fail_after(90):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    # Exactly one, at the end: the filler's `turn_complete` arrives with `interaction_status: IN_PROGRESS`,
    # and the exchange isn't over until the model says `IDLE`. Breaking on the first one above is what
    # pins this — a premature boundary would have ended the loop before the tool ever ran.
    assert sum(isinstance(event, RealtimeTurnCompleteEvent) for event in events) == 1
    assert [type(event.part).__name__ for event in events if isinstance(event, FunctionToolCallEvent)] == [
        'ToolCallPart'
    ]

    assert sent_frames_containing(cassette, 'Search flights between two cities.') == snapshot(
        [
            {
                'setup': {
                    'model': 'models/gemini-3.8-live-extended-thinking',
                    'generationConfig': {'responseModalities': ['AUDIO'], 'thinkingConfig': {'thinking_level': 'LOW'}},
                    'systemInstruction': {
                        'parts': [
                            {'text': 'You are a flight booking assistant. Always use search_flights before answering.'}
                        ],
                        'role': 'user',
                    },
                    'tools': [
                        {
                            'functionDeclarations': [
                                {
                                    'description': 'Search flights between two cities. Takes several seconds.',
                                    'name': 'search_flights',
                                    'parameters': {
                                        'properties': {'origin': {'type': 'STRING'}, 'destination': {'type': 'STRING'}},
                                        'required': ['origin', 'destination'],
                                        'type': 'OBJECT',
                                    },
                                    'behavior': 'NON_BLOCKING',
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
    assert sent_frames_containing(cassette, 'KLM at 120 dollars') == snapshot(
        [
            {
                'tool_response': {
                    'functionResponses': [
                        {
                            'id': 'call_3850_fc_0_0',
                            'name': 'search_flights',
                            'response': {'output': 'KLM at 120 dollars'},
                        }
                    ]
                }
            }
        ]
    )
    messages = session.all_messages()
    assert [type(m).__name__ for m in messages] == snapshot(
        # One `ModelResponse` for the whole stalled exchange: the model's `turn_complete` after the filler
        # came with `interaction_status: IN_PROGRESS`, so the utterance and the tool call it was stalling
        # for stay together rather than splitting into two responses.
        ['ModelRequest', 'ModelResponse', 'ModelRequest', 'ModelResponse']
    )
    stalled = messages[1]
    assert isinstance(stalled, ModelResponse)
    filler_part = stalled.parts[0]
    assert isinstance(filler_part, SpeechPart)
    assert filler_part.transcript == snapshot('Let me check the available flights for you.')
    assert stalled.parts[1] == ToolCallPart(tool_name='search_flights', args=IsStr(), tool_call_id=IsStr())
    # The filler's reasoning is billed against the response that carries it; Gemini's tool-call frame has
    # no usage of its own to merge in.
    assert stalled.usage.details['thoughts_tokens'] == snapshot(71)

    final = messages[3]
    assert isinstance(final, ModelResponse)
    final_part = final.parts[0]
    assert isinstance(final_part, SpeechPart)
    assert final_part.transcript is not None and 'KLM' in final_part.transcript
    # Priced since `genai-prices` 0.1.9, the first release with the Gemini 3.8 Live rates.
    assert final.usage.cost == snapshot(Decimal('0.01005375'))


@pytest.mark.realtime_ws_hold_open
async def test_text_that_cuts_off_the_answer_to_a_tool_result_is_answered_by_the_next_turn(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette],
) -> None:
    """Text sent right after a tool result cuts off the turn answering the result before it said anything.

    Gemini reports that turn `interrupted` and complete, then answers the text in a turn of its own: the cut-off
    turn answers nothing, so the wait for the text's reply goes on to that one.
    """
    provider, _ = gemini_ws_cassette
    model = GoogleRealtimeModel('gemini-3.1-flash-live-preview', provider=provider)
    agent = Agent(instructions='Keep replies to one short sentence. Use the lookup tool for any order.')

    @agent.tool_plain
    def lookup(order: str) -> str:
        """Look up an order's status."""
        return f'Order {order} has shipped and arrives on Friday.'

    messages: list[ModelMessage] = []
    result_sent = anyio.Event()
    turn_complete = anyio.Event()
    turns: list[str] = []
    """What the session says, in order: the model's words, and each turn it reports complete."""

    async def watch(session: RealtimeSession) -> None:
        async for event in session:  # pragma: no branch
            if isinstance(event, FunctionToolResultEvent):
                result_sent.set()
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, SpeechPartDelta):
                turns.append(event.delta.transcript_delta or '')
            elif isinstance(event, RealtimeTurnCompleteEvent):
                turns.append('<turn complete>')
                turn_complete.set()

    async with agent.realtime(model).session() as session, anyio.create_task_group() as tasks:
        tasks.start_soon(watch, session)
        await session.send('What is the status of my order 4411?')
        with anyio.fail_after(45):
            await result_sent.wait()
            await session.send('Also, what is 2 plus 2? Answer with the number.')
            await session.wait_for_reply()
            await turn_complete.wait()
        messages = session.all_messages()
        tasks.cancel_scope.cancel()

    # The cut-off turn completes no exchange: the turn answering the text does.
    assert turns == snapshot(['4', '<turn complete>'])
    assert isinstance(messages[-1], ModelResponse)
    assert [part.transcript for part in messages[-1].parts if isinstance(part, SpeechPart)] == snapshot(['4'])


@pytest.mark.realtime_ws_hold_open
async def test_text_that_cuts_off_a_tool_call_turn_is_answered_by_the_next_turn(
    gemini_ws_cassette: tuple[Provider[Any], RealtimeCassette],
) -> None:
    """Text that reaches Gemini while it generates a tool call, before the session has read the call.

    Gemini cancels the call, reports the turn `interrupted` and complete, and answers both requests in the turns
    that follow. The session took the tool-call turn for the text's reply too, as it couldn't tell the text
    arrived after the turn began: the cancellation says it did, so the text's reply is the one that follows.
    """
    provider, _ = gemini_ws_cassette
    model = GoogleRealtimeModel('gemini-3.1-flash-live-preview', provider=provider)
    agent = Agent(instructions='Keep replies to one short sentence. Use the lookup tool for any order.')

    calls = 0

    @agent.tool_plain
    async def lookup(order: str) -> str:
        """Look up an order's status."""
        nonlocal calls
        calls += 1
        if calls == 1:
            # Still running when Gemini cancels the call: the session stops it, and no result goes out.
            await anyio.sleep_forever()
        return f'Order {order} has shipped and arrives on Friday.'

    said: list[str] = []
    turn_complete = anyio.Event()

    async def watch(session: RealtimeSession) -> None:
        async for event in session:  # pragma: no branch
            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, SpeechPartDelta):
                said.append(event.delta.transcript_delta or '')
            elif isinstance(event, RealtimeTurnCompleteEvent):
                said.append('<turn complete>')
                turn_complete.set()

    async with agent.realtime(model).session() as session, anyio.create_task_group() as tasks:
        tasks.start_soon(watch, session)
        await session.send('What is the status of my order 4411?')
        # Recording only: GEMINI_CUT_DELAY spaces the two sends so the text reaches Gemini mid-generation, before the
        # session reads the tool call; replay keeps the recorded order.
        if delay := os.environ.get('GEMINI_CUT_DELAY'):  # pragma: no cover
            await anyio.sleep(float(delay))
        await session.send('Also, what is 2 plus 2? Answer with the number.')
        with anyio.fail_after(45):
            await session.wait_for_reply()
            said.append('<wait over>')
            await turn_complete.wait()
        tasks.cancel_scope.cancel()

    # The wait is over only once the reply to both requests is.
    assert ''.join(said) == snapshot('Order 4411 4.<turn complete><wait over>')
