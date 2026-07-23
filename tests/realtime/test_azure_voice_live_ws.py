"""Cassette-backed end-to-end test for the Azure AI Voice Live realtime provider."""

from __future__ import annotations as _annotations

from pathlib import Path
from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunUsage
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

from ..conftest import IsDatetime, IsStr, try_import
from .ws_cassettes import RealtimeCassette
from .ws_helpers import collapse_event_types, sent_frames_containing

with try_import() as imports_successful:
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.realtime.azure import AzureRealtimeModel, AzureRealtimeModelSettings

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not imports_successful(), reason='websockets not installed')]


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
    async with agent.realtime_session(model=model, audio_retention='output_audio') as session:
        await session.send('Say a short greeting.')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch - breaks on the recorded terminal event
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
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
        ['PartStartEvent', 'PartDeltaEvent', 'PartEndEvent', 'TurnCompleteEvent']
    )
    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == snapshot(['ModelRequest', 'ModelResponse'])
    assert messages[0] == ModelRequest(parts=[UserPromptPart(content='Say a short greeting.', timestamp=IsDatetime())])
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
    assert session.usage == snapshot(
        RunUsage(
            input_tokens=16,
            output_tokens=45,
            output_audio_tokens=31,
            details={'input_text_tokens': 16, 'input_image_tokens': 0, 'output_text_tokens': 14},
            requests=1,
        )
    )


async def test_audio_in_server_vad_turn(
    azure_voice_live_ws_cassette: tuple[AzureProvider, RealtimeCassette], assets_path: Path
) -> None:
    """A spoken user turn is segmented by server VAD and retained as transcribed history."""
    provider, _ = azure_voice_live_ws_cassette
    model = AzureRealtimeModel(
        'gpt-realtime', provider=provider, settings=AzureRealtimeModelSettings(azure_voice_live=True)
    )
    assert model.profile == RealtimeModelProfile(
        supports_image_input=True,
        supports_manual_turn_control=True,
        supports_interruption=True,
        supports_output_truncation=True,
        supports_session_seeding=True,
        supports_seeding_images=True,
        supports_seeding_audio=True,
        audio_input_sample_rate=24000,
        audio_output_sample_rate=24000,
        supports_thinking=False,
        supported_native_tools=frozenset(),
    )
    agent = Agent(instructions='Reply in a few words.')
    pcm = assets_path.joinpath('marcelo_24khz.pcm').read_bytes()

    events: list[Any] = []
    async with agent.realtime_session(model=model) as session:
        for start in range(0, len(pcm), 4800):
            await session.send_audio(pcm[start : start + 4800])
        with anyio.fail_after(45):
            async for event in session:  # pragma: no branch - breaks on the recorded terminal event
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
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
    assert session.usage == snapshot(
        RunUsage(
            input_tokens=44,
            output_tokens=99,
            input_audio_tokens=30,
            output_audio_tokens=72,
            details={'input_text_tokens': 14, 'input_image_tokens': 0, 'output_text_tokens': 27},
            requests=1,
        )
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
    async with agent.realtime_session(model=model) as session:
        await session.send('What is the weather in London?')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch - breaks on the recorded terminal event
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
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
        parts=[UserPromptPart(content='What is the weather in London?', timestamp=IsDatetime())]
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
    async with agent.realtime_session(model=model, message_history=history) as session:
        await session.send('What is my name and favorite color?')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch - breaks on the recorded terminal event
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
                    break

    assert [event for event in events if isinstance(event, SessionErrorEvent)] == []
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
