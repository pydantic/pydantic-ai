"""Cassette-backed end-to-end test for the Azure AI Voice Live realtime provider."""

from __future__ import annotations as _annotations

from pathlib import Path
from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunUsage
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
from pydantic_ai.realtime import RealtimeModelProfile, RealtimeSessionErrorEvent, RealtimeTurnCompleteEvent

from ..conftest import IsDatetime, IsStr, try_import
from .ws_cassettes import RealtimeCassette
from .ws_helpers import collapse_event_types, sent_frames_containing

with try_import() as imports_successful:
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.realtime import WebRTCSession
    from pydantic_ai.realtime.azure import AzureRealtimeModel, AzureRealtimeModelSettings

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not imports_successful(), reason='websockets not installed')]


async def test_text_output_modality_returns_text(
    azure_voice_live_ws_cassette: tuple[AzureProvider, RealtimeCassette],
) -> None:
    """`output_modality='text'` really produces text, which is why the profile reports it supported.

    Recorded against the live service rather than inferred from the session config: Gemini Live also
    *accepts* `modalities: ['text']` on the wire and then rejects the combination at session setup, so
    the plumbing mapping the setting through proves nothing on its own. Voice Live answers with text
    deltas and no audio, so `supports_text_output` stays `True` and `Agent.realtime`'s guard lets the
    session open.
    """
    provider, cassette = azure_voice_live_ws_cassette
    model = AzureRealtimeModel(
        'gpt-realtime',
        provider=provider,
        settings=AzureRealtimeModelSettings(azure_voice_live=True, output_modality='text'),
    )
    agent = Agent(instructions='Answer in two or three words.')

    events: list[Any] = []
    async with agent.realtime(model).session() as session:
        await session.send('Say a short greeting.')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch - breaks on the recorded terminal event
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    # Only `text` is requested, and the session opens rather than being rejected at setup.
    assert [frame['session']['modalities'] for frame in sent_frames_containing(cassette, 'two or three words')] == (
        snapshot([['text']])
    )
    assert [event for event in events if isinstance(event, RealtimeSessionErrorEvent)] == []

    messages = session.all_messages()
    response = messages[1]
    assert isinstance(response, ModelResponse)
    part = response.parts[0]
    # A `TextPart`, not a `SpeechPart` with a transcript: the model wrote rather than spoke.
    assert isinstance(part, TextPart)
    assert part.content == snapshot('Hello there!')


async def test_text_in_audio_out_turn(
    azure_voice_live_ws_cassette: tuple[AzureProvider, RealtimeCassette],
) -> None:
    """A text turn produces live audio/transcript events and standard message history."""
    provider, cassette = azure_voice_live_ws_cassette
    model = AzureRealtimeModel(
        'gpt-realtime', provider=provider, settings=AzureRealtimeModelSettings(azure_voice_live=True)
    )
    agent = Agent(instructions='Answer in two or three words.')

    events: list[Any] = []
    async with agent.realtime(model).session(audio_retention='output_audio') as session:
        await session.send('Say a short greeting.')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch - breaks on the recorded terminal event
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    assert sent_frames_containing(cassette, 'Answer in two or three words.') == snapshot(
        [
            {
                'type': 'session.update',
                'session': {
                    'instructions': 'Answer in two or three words.',
                    'modalities': ['text', 'audio'],
                    'input_audio_format': 'pcm16',
                    'output_audio_format': 'pcm16',
                    'input_audio_sampling_rate': 24000,
                    'turn_detection': {
                        'type': 'server_vad',
                        'create_response': True,
                        'interrupt_response': True,
                    },
                    'input_audio_transcription': {'model': 'whisper-1'},
                },
            }
        ]
    )

    assert collapse_event_types(events) == snapshot(
        ['PartStartEvent', 'PartDeltaEvent', 'PartEndEvent', 'RealtimeTurnCompleteEvent']
    )
    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == snapshot(['ModelRequest', 'ModelResponse'])
    assert messages[0] == ModelRequest(
        parts=[UserPromptPart(content='Say a short greeting.', timestamp=IsDatetime())],
        timestamp=IsDatetime(),
        conversation_id=IsStr(),
        run_id=IsStr(),
    )
    response = messages[1]
    assert isinstance(response, ModelResponse)
    assert response.model_name == 'gpt-realtime-global-standard'
    part = response.parts[0]
    assert isinstance(part, SpeechPart)
    assert part.speaker == 'assistant'
    assert part.transcript == snapshot('Hola, ¿qué tal?')
    assert isinstance(part.audio, BinaryContent)
    assert part.audio.media_type == 'audio/wav'
    assert len(part.audio.data) > 0
    # `output_reasoning_tokens` is an extension attribute rather than a declared field, so a whole-object
    # `snapshot(RunUsage(...))` can't round-trip it — and `RunUsage` counts "absent" and "explicitly 0"
    # as different, so it can't just be left out. Snapshot the detail keys, state the counts here.
    assert session.usage == RunUsage(
        input_tokens=16,
        output_tokens=45,
        output_audio_tokens=31,
        output_reasoning_tokens=0,
        details=snapshot(
            {
                'input_text_tokens': 16,
                'input_image_tokens': 0,
                'output_text_tokens': 14,
                'audio_tokens': 31,
                'reasoning_tokens': 0,
            }
        ),
        requests=1,
    )


async def test_audio_in_server_vad_turn(
    azure_voice_live_ws_cassette: tuple[AzureProvider, RealtimeCassette], assets_path: Path
) -> None:
    """A spoken user turn is segmented by server VAD and retained as transcribed history."""
    provider, _ = azure_voice_live_ws_cassette
    model = AzureRealtimeModel(
        'gpt-realtime', provider=provider, settings=AzureRealtimeModelSettings(azure_voice_live=True)
    )
    # `gpt-realtime` is served by both APIs, so it carries no `azure_realtime_apis` constraint;
    # `azure_voice_live=True` selected Voice Live here.
    assert model.profile == RealtimeModelProfile(
        supports_image_input=True,
        supports_manual_turn_control=True,
        supports_interruption=True,
        supports_output_truncation=True,
        supports_session_seeding=True,
        supports_seeding_images=True,
        supports_seeding_audio=True,
        # Voice Live negotiates WebRTC over its own control channel, not the GA signaling endpoints.
        supports_webrtc=False,
        # Inherited from the OpenAI realtime profile, which Azure delegates to wholesale: Voice Live
        # serves the same models, and they keep talking while a tool call is outstanding.
        supports_async_tool_calls=True,
        # Voice Live's session config takes `modalities: ['text']`, so text output is supported.
        supports_text_output=True,
        supports_tool_return_schema=False,  # no native surface; opted-in schemas go into descriptions
        emits_input_speech_events=True,
        audio_input_sample_rate=24000,
        audio_output_sample_rate=24000,
        supports_thinking=False,
        supported_native_tools=frozenset(),
    )
    agent = Agent(instructions='Reply in a few words.')
    pcm = assets_path.joinpath('marcelo_24khz.pcm').read_bytes()

    events: list[Any] = []
    async with agent.realtime(model).session() as session:
        for start in range(0, len(pcm), 4800):
            await session.send_audio(pcm[start : start + 4800])
        with anyio.fail_after(45):
            async for event in session:  # pragma: no branch - breaks on the recorded terminal event
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == snapshot(['ModelRequest', 'ModelResponse'])
    user_turn = messages[0]
    assert isinstance(user_turn, ModelRequest)
    user_part = user_turn.parts[0]
    assert isinstance(user_part, SpeechPart)
    assert user_part.speaker == 'user'
    assert user_part.transcript == snapshot('Cześć, nazywam się Marcelo.')
    reply = messages[1]
    assert isinstance(reply, ModelResponse)
    assert isinstance(reply.parts[0], SpeechPart)
    # See the note on the text-turn test: an extension attribute can't survive a whole-object snapshot.
    assert session.usage == RunUsage(
        input_tokens=44,
        output_tokens=99,
        input_audio_tokens=30,
        output_audio_tokens=72,
        output_reasoning_tokens=0,
        details=snapshot(
            {
                'input_text_tokens': 14,
                'input_image_tokens': 0,
                'output_text_tokens': 27,
                'audio_tokens': 72,
                'reasoning_tokens': 0,
            }
        ),
        requests=1,
    )


async def test_tool_call_round(
    azure_voice_live_ws_cassette: tuple[AzureProvider, RealtimeCassette],
) -> None:
    """A tool call is executed and its result is folded into standard message history."""
    provider, cassette = azure_voice_live_ws_cassette
    model = AzureRealtimeModel(
        'gpt-realtime',
        provider=provider,
        settings=AzureRealtimeModelSettings(azure_voice_live=True, output_modality='text'),
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
            async for event in session:  # pragma: no branch - breaks on the recorded terminal event
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    assert sent_frames_containing(cassette, 'Look up the weather for a city.') == snapshot(
        [
            {
                'type': 'session.update',
                'session': {
                    'instructions': 'Use the get_weather tool for any weather question, then answer in one short sentence.',
                    'modalities': ['text'],
                    'input_audio_format': 'pcm16',
                    'output_audio_format': 'pcm16',
                    'input_audio_sampling_rate': 24000,
                    'turn_detection': {
                        'type': 'server_vad',
                        'create_response': True,
                        'interrupt_response': True,
                    },
                    'input_audio_transcription': {'model': 'whisper-1'},
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
    call_events = [event for event in events if isinstance(event, FunctionToolCallEvent)]
    result_events = [event for event in events if isinstance(event, FunctionToolResultEvent)]
    assert len(call_events) == 1
    assert call_events[0].part.tool_name == 'get_weather'
    assert call_events[0].part.args_as_dict() == {'city': 'London'}
    assert len(result_events) == 1
    assert isinstance(result_events[0].part, ToolReturnPart)
    assert result_events[0].part.content == 'It is foggy and 12 degrees in London.'

    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == snapshot(
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
    assert isinstance(final_part, TextPart)
    assert 'fog' in final_part.content.lower()
    assert session.usage.requests == 2


async def test_message_history_seeding(
    azure_voice_live_ws_cassette: tuple[AzureProvider, RealtimeCassette],
) -> None:
    """Seeded prior turns are sent on the wire and retained ahead of the new turn."""
    provider, cassette = azure_voice_live_ws_cassette
    model = AzureRealtimeModel(
        'gpt-realtime',
        provider=provider,
        settings=AzureRealtimeModelSettings(azure_voice_live=True, output_modality='text'),
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
            async for event in session:  # pragma: no branch - breaks on the recorded terminal event
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    assert [event for event in events if isinstance(event, RealtimeSessionErrorEvent)] == []
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
    messages = session.all_messages()
    assert messages[:2] == history
    reply = messages[-1]
    assert isinstance(reply, ModelResponse)
    reply_part = reply.parts[0]
    assert isinstance(reply_part, TextPart)
    content = reply_part.content.lower()
    assert 'alice' in content and 'teal' in content


async def test_voice_live_rejects_webrtc_signaling() -> None:
    """Browser WebRTC signaling is not supported for Voice Live yet, so it raises rather than using the GA path.

    A unit test (no cassette): the guard fires before any network call. Voice Live negotiates WebRTC over
    its WebSocket control channel, unlike the GA `/realtime/client_secrets` + `/realtime/calls` flow this
    model inherits, so minting a GA secret for a Voice Live session would hit the wrong endpoint. Tracked
    in https://github.com/pydantic/pydantic-ai/issues/6702.
    """
    provider = AzureProvider(azure_endpoint='https://mock.openai.azure.com/openai/v1', api_key='mock-api-key')
    model = AzureRealtimeModel(
        'gpt-realtime', provider=provider, settings=AzureRealtimeModelSettings(azure_voice_live=True)
    )
    with pytest.raises(UserError, match='not yet supported for Azure AI Voice Live'):
        await model.create_client_secret()
    # `answer_webrtc_offer` mints through `create_client_secret`, so it is rejected too.
    with pytest.raises(UserError, match='not yet supported for Azure AI Voice Live'):
        await model.answer_webrtc_offer('v=0\r\n')
    # `connect_webrtc` (attaching a server sideband to an already-negotiated call) is the third signaling
    # entry point; it is guarded the same way so a Voice Live session can't be sidebanded onto the
    # inherited GA endpoint. The guard fires eagerly, before the call handle or messages are touched.
    webrtc_session = WebRTCSession('azure', session_id='call_mock')
    params = ModelRequestParameters()
    with pytest.raises(UserError, match='not yet supported for Azure AI Voice Live'):
        async with model.connect_webrtc(
            webrtc_session, messages=[], model_settings=None, model_request_parameters=params
        ):
            pass  # pragma: no cover — the guard raises on enter, before the body runs

    # The guard reads *merged* settings, so a per-call `azure_voice_live=True` is rejected on a GA-default
    # model too (not only when it's a model-level default).
    ga_model = AzureRealtimeModel('gpt-realtime', provider=provider)
    per_call = AzureRealtimeModelSettings(azure_voice_live=True)
    with pytest.raises(UserError, match='not yet supported for Azure AI Voice Live'):
        await ga_model.create_client_secret(model_settings=per_call)
    with pytest.raises(UserError, match='not yet supported for Azure AI Voice Live'):
        await ga_model.answer_webrtc_offer('v=0\r\n', model_settings=per_call)
    with pytest.raises(UserError, match='not yet supported for Azure AI Voice Live'):
        async with ga_model.connect_webrtc(
            webrtc_session, messages=[], model_settings=per_call, model_request_parameters=params
        ):
            pass  # pragma: no cover — the guard raises on enter, before the body runs

    # A Voice-Live-only model (e.g. `gpt-5`) auto-routes to Voice Live with no `azure_voice_live` setting
    # at all, so the guard must consult routing, not just the raw setting — otherwise these mint a GA
    # secret / attach a GA sideband for a Voice Live session.
    auto_model = AzureRealtimeModel('gpt-5', provider=provider)
    with pytest.raises(UserError, match='not yet supported for Azure AI Voice Live'):
        await auto_model.create_client_secret()
    with pytest.raises(UserError, match='not yet supported for Azure AI Voice Live'):
        await auto_model.answer_webrtc_offer('v=0\r\n')
    with pytest.raises(UserError, match='not yet supported for Azure AI Voice Live'):
        async with auto_model.connect_webrtc(
            webrtc_session, messages=[], model_settings=None, model_request_parameters=params
        ):
            pass  # pragma: no cover — the guard raises on enter, before the body runs
