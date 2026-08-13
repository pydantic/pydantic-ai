"""Cassette-backed end-to-end test for Azure OpenAI realtime."""

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
    RealtimeSessionErrorEvent,
    SpeechPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.realtime import RealtimeTurnCompleteEvent
from pydantic_ai.usage import RunUsage

from ..conftest import IsDatetime, IsStr, try_import
from .conftest import REAL_SDP_OFFER
from .ws_cassettes import RealtimeCassette
from .ws_helpers import collapse_event_types, sent_frames_containing

with try_import() as imports_successful:
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.realtime.azure import AzureRealtimeModel
    from pydantic_ai.realtime.openai import OpenAIRealtimeModelSettings

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed'),
]


async def test_text_in_audio_out_turn(
    azure_ws_cassette: tuple[AzureProvider, RealtimeCassette],
) -> None:
    """A text turn uses the GA session shape and produces Azure-hosted audio and transcript."""
    provider, cassette = azure_ws_cassette
    model = AzureRealtimeModel('gpt-realtime', provider=provider)
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
    assert response.model_name == 'gpt-realtime'
    part = response.parts[0]
    assert isinstance(part, SpeechPart)
    assert part.speaker == 'assistant'
    assert part.transcript
    assert isinstance(part.audio, BinaryContent)
    assert part.audio.media_type == 'audio/wav'
    assert len(part.audio.data) > 0
    assert session.usage == snapshot(
        RunUsage(
            input_tokens=16,
            output_tokens=98,
            output_audio_tokens=82,
            details={
                'input_text_tokens': 16,
                'input_image_tokens': 0,
                'output_text_tokens': 16,
                'audio_tokens': 82,
            },
            requests=1,
        )
    )


async def test_tool_call_round(azure_ws_cassette: tuple[AzureProvider, RealtimeCassette]) -> None:
    """A tool call is executed by the session and its result folded back into a classic-shaped history."""
    provider, cassette = azure_ws_cassette
    model = AzureRealtimeModel(
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

    # The tool schema is sent on the wire in the GA session shape.
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
    # Text-output modality, so the reply is a `TextPart`, not a `SpeechPart`.
    assert isinstance(final_part, TextPart)
    assert 'fog' in final_part.content.lower()

    # Both provider responses are accounted for: the intermediate function-call-only `response.done`
    # counts its tokens even though it maps to no turn event.
    assert session.usage.requests == 2
    assert session.usage.input_tokens > 0 and session.usage.output_tokens > 0


async def test_message_history_seeding(azure_ws_cassette: tuple[AzureProvider, RealtimeCassette]) -> None:
    """Seeded prior turns are sent on the wire and reflected in the model's reply."""
    provider, cassette = azure_ws_cassette
    model = AzureRealtimeModel(
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

    # A server-side rejection of the seeded items would surface as a `RealtimeSessionErrorEvent`; assert none.
    assert [event for event in events if isinstance(event, RealtimeSessionErrorEvent)] == []

    # The seeded user and assistant turns were sent as `conversation.item.create` frames on the wire.
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

    # `all_messages()` carries the seeded history ahead of this session's turns.
    messages = session.all_messages()
    assert messages[:2] == history
    reply = messages[-1]
    assert isinstance(reply, ModelResponse)
    reply_part = reply.parts[0]
    assert isinstance(reply_part, TextPart)
    content = reply_part.content.lower()
    assert 'alice' in content and 'teal' in content


async def test_audio_in_server_vad_transcription_requires_deployment(
    azure_ws_cassette: tuple[AzureProvider, RealtimeCassette], assets_path: Path
) -> None:
    """Azure keeps a placeholder turn when input transcription lacks a deployed model.

    Unlike OpenAI, where the default `gpt-realtime-whisper` is hosted, Azure GA realtime resolves the
    input-transcription model against the resource's own deployments — so the default fails with
    `DeploymentNotFound` on every turn unless a transcription model is deployed and configured. The
    failure is surfaced while the user turn remains represented in history. This cassette was recorded
    against a resource without a transcription deployment.
    """
    provider, _ = azure_ws_cassette
    model = AzureRealtimeModel('gpt-realtime', provider=provider)
    agent = Agent(instructions='Reply in a few words.')
    pcm = assets_path.joinpath('marcelo_24khz.pcm').read_bytes()

    events: list[Any] = []
    async with agent.realtime(model).session() as session:
        # Stream the clip in ~100 ms chunks like a live mic; the trailing silence lets server VAD end it.
        for start in range(0, len(pcm), 4800):
            await session.send_audio(pcm[start : start + 4800])
        with anyio.fail_after(45):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    assert collapse_event_types(events) == snapshot(
        [
            'RealtimeInputSpeechStartEvent',
            'RealtimeInputSpeechEndEvent',
            'PartStartEvent',
            'PartEndEvent',
            'RealtimeInputTranscriptionErrorEvent',
            'PartStartEvent',
            'PartDeltaEvent',
            'PartEndEvent',
            'RealtimeTurnCompleteEvent',
        ]
    )
    messages = session.new_messages()
    assert isinstance(messages[0], ModelRequest)
    user_part = messages[0].parts[0]
    assert isinstance(user_part, SpeechPart)
    assert user_part.speaker == 'user' and user_part.transcript is None and user_part.audio is None
    assert isinstance(messages[1], ModelResponse)


async def test_audio_in_server_vad_transcribes(
    azure_ws_cassette: tuple[AzureProvider, RealtimeCassette], assets_path: Path
) -> None:
    """Audio-in server-VAD on Azure GA with a *deployed* transcription model transcribes the user turn.

    The companion to `test_audio_in_server_vad_transcription_requires_deployment`: once a transcription
    model (here `gpt-realtime-whisper`, which `input_transcription_model='auto'` resolves to) is deployed
    on the resource, the spoken turn lands in history as a transcribed user `SpeechPart`, exactly like
    OpenAI's hosted default.
    """
    provider, _ = azure_ws_cassette
    model = AzureRealtimeModel('gpt-realtime', provider=provider)
    agent = Agent(instructions='Reply in a few words.')
    pcm = assets_path.joinpath('marcelo_24khz.pcm').read_bytes()

    events: list[Any] = []
    async with agent.realtime(
        model, model_settings=OpenAIRealtimeModelSettings(input_transcription_model='gpt-realtime-whisper')
    ).session() as session:
        for start in range(0, len(pcm), 4800):
            await session.send_audio(pcm[start : start + 4800])
        with anyio.fail_after(45):
            async for event in session:  # pragma: no branch - the loop always breaks on RealtimeTurnCompleteEvent
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    messages = session.all_messages()
    user_turn = messages[0]
    assert isinstance(user_turn, ModelRequest)
    user_part = user_turn.parts[0]
    assert isinstance(user_part, SpeechPart)
    assert user_part.speaker == 'user'
    assert user_part.transcript == snapshot('Hello, my name is Marcelo.')


@pytest.mark.vcr
async def test_webrtc_sideband_text_turn(
    azure_ws_sideband_cassette: tuple[AzureProvider, RealtimeCassette],
) -> None:
    """Azure's secure WebRTC flow end to end: two-step HTTP relay, then run the agent over the sideband.

    The Azure two-step signaling (mint client secret + relay the raw SDP offer) is a VCR cassette; the
    control WebSocket attached by `call_id` is a WS cassette. This is the Azure counterpart to the OpenAI
    `test_webrtc_sideband_text_turn`, proving the inherited sideband runs a turn over Azure's `/openai/v1`
    control URL and `api-key` auth (not just that signaling returns a `call_id`).
    """
    provider, cassette = azure_ws_sideband_cassette
    model = AzureRealtimeModel(
        'gpt-realtime', provider=provider, settings=OpenAIRealtimeModelSettings(output_modality='text')
    )
    agent = Agent(instructions='Answer in two words.')

    answer = await model.answer_webrtc_offer(REAL_SDP_OFFER, instructions='Answer in two words.')
    assert answer.sdp.startswith('v=0')
    assert answer.session.provider_name == 'azure'
    assert answer.session.call_id.startswith('rtc_')

    events: list[Any] = []
    async with agent.realtime(model).session(provider_session=answer.session) as session:
        # The sideband doesn't own the audio transport, so the audio methods are unavailable.
        with pytest.raises(UserError, match='does not own the audio transport'):
            await session.send_audio(b'\x00\x00')

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
    reply = messages[1]
    assert isinstance(reply, ModelResponse)
    assert reply.model_name == 'gpt-realtime'
    assert isinstance(reply.parts[0], TextPart)


async def test_spoken_turn_transcribed_drives_a_tool_and_answers_in_audio(
    azure_ws_cassette: tuple[AzureProvider, RealtimeCassette], assets_path: Path
) -> None:
    """The whole spoken round trip on Azure: heard, transcribed, tool called, answered in speech.

    The other Azure tests each cover one leg — text in/audio out, a text-driven tool call, and audio in
    with transcription *failing* for want of a deployment. This is the combination a browser voice agent
    actually runs, with input transcription pointed at a deployment that exists.

    Only realtime-capable transcription models are accepted here: a classic `whisper` deployment is
    rejected with `DeploymentNotFound` like a missing one, so the deployment has to be of a model such
    as `gpt-4o-transcribe`.
    """
    provider, cassette = azure_ws_cassette
    model = AzureRealtimeModel(
        'gpt-realtime',
        provider=provider,
        settings=OpenAIRealtimeModelSettings(input_transcription_model='gpt-4o-transcribe'),
    )
    agent = Agent(
        instructions=(
            'When someone introduces themselves, call `remember_name` with their name, '
            'then greet them by name in a few words.'
        )
    )

    @agent.tool_plain
    def remember_name(name: str) -> str:
        """Store the name the user introduced themselves with."""
        return f'Stored {name}.'

    pcm = assets_path.joinpath('marcelo_24khz.pcm').read_bytes()

    events: list[Any] = []
    async with agent.realtime(model).session(audio_retention='output_audio') as session:
        # Stream the clip in ~100 ms chunks like a live mic; the trailing silence lets server VAD end it.
        for start in range(0, len(pcm), 4800):
            await session.send_audio(pcm[start : start + 4800])
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    # The deployed transcription model is what goes on the wire, not the unusable default.
    assert sent_frames_containing(cassette, 'gpt-4o-transcribe') == snapshot(
        [
            {
                'type': 'session.update',
                'session': {
                    'type': 'realtime',
                    'instructions': 'When someone introduces themselves, call `remember_name` with their name, then greet them by name in a few words.',
                    'output_modalities': ['audio'],
                    'audio': {
                        'input': {
                            'format': {'type': 'audio/pcm', 'rate': 24000},
                            'turn_detection': {
                                'type': 'server_vad',
                                'create_response': True,
                                'interrupt_response': True,
                            },
                            'transcription': {'model': 'gpt-4o-transcribe'},
                        },
                        'output': {'format': {'type': 'audio/pcm', 'rate': 24000}},
                    },
                    'tools': [
                        {
                            'type': 'function',
                            'name': 'remember_name',
                            'parameters': {
                                'additionalProperties': False,
                                'properties': {'name': {'type': 'string'}},
                                'required': ['name'],
                                'type': 'object',
                            },
                            'description': 'Store the name the user introduced themselves with.',
                        }
                    ],
                },
            }
        ]
    )

    assert [event for event in events if isinstance(event, RealtimeSessionErrorEvent)] == []
    assert collapse_event_types(events) == snapshot(
        [
            'RealtimeInputSpeechStartEvent',
            'RealtimeInputSpeechEndEvent',
            'PartStartEvent',
            'PartDeltaEvent',
            'PartEndEvent',
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

    # The spoken turn was transcribed, so history carries the user's words rather than a bare audio part.
    messages = session.all_messages()
    assert isinstance(messages[0], ModelRequest)
    spoken = messages[0].parts[0]
    assert isinstance(spoken, SpeechPart)
    assert spoken.speaker == 'user'
    assert spoken.transcript == snapshot('Hello, my name is Marcelo.')

    # Those words drove the tool call, with the name taken from the transcribed speech.
    call_events = [e for e in events if isinstance(e, FunctionToolCallEvent)]
    result_events = [e for e in events if isinstance(e, FunctionToolResultEvent)]
    assert len(call_events) == 1
    assert call_events[0].part.tool_name == 'remember_name'
    assert call_events[0].part.args_as_dict() == snapshot({'name': 'Marcelo'})
    assert len(result_events) == 1
    assert isinstance(result_events[0].part, ToolReturnPart)

    # And the answer came back as retained speech, not text.
    final = messages[-1]
    assert isinstance(final, ModelResponse)
    reply = final.parts[-1]
    assert isinstance(reply, SpeechPart)
    assert reply.speaker == 'assistant'
    assert reply.audio is not None and len(reply.audio.data) > 0
    assert 'marcelo' in (reply.transcript or '').lower()
