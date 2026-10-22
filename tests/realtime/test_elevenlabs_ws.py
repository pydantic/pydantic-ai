"""Cassette-backed tests for the ElevenLabs realtime provider, exercising the real WebSocket protocol.

These complement the network-free `test_elevenlabs.py` unit tests: they replay recorded provider frames
end-to-end through [`Agent.realtime`][pydantic_ai.agent.Agent.realtime] to prove the real protocol — the
streamed part events, the tool round-trip, message-history seeding, and reconnect.

Recording requires an `ELEVENLABS_API_KEY`:

    uv run --env-file .env pytest tests/realtime/test_elevenlabs_ws.py --record-mode=rewrite
"""

from __future__ import annotations as _annotations

from pathlib import Path
from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunUsage
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    PartStartEvent,
    RealtimeSessionErrorEvent,
    SpeechPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.realtime import RealtimeSessionReconnectEvent, RealtimeTurnCompleteEvent

from ..conftest import try_import
from .ws_cassettes import CassetteClose, CassetteMessage, RealtimeCassette
from .ws_helpers import collapse_event_types, sent_frames_containing

with try_import() as imports_successful:
    from pydantic_ai.providers.elevenlabs import ElevenLabsProvider
    from pydantic_ai.realtime.elevenlabs import ElevenLabsRealtimeModel, ElevenLabsRealtimeModelSettings

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed'),
]

MODEL = 'eleven_realtime_v1_mini'


async def _until_turn_complete(session: Any, events: list[Any], *, timeout: float = 30) -> None:
    with anyio.fail_after(timeout):
        async for event in session:  # pragma: no branch
            events.append(event)
            if isinstance(event, RealtimeTurnCompleteEvent):
                return


async def test_text_in_audio_out_turn(elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette]) -> None:
    """A text-in turn yields streamed audio+transcript parts, a classic-shaped history, and usage."""
    provider, cassette = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(MODEL, provider=provider)
    agent = Agent(instructions='Answer in two or three words.')

    events: list[Any] = []
    async with agent.realtime(model).session(audio_retention='output_audio') as session:
        await session.send('Say a short greeting.')
        await _until_turn_complete(session, events)

    # The `session.update` the server accepted: the shape every `_session_config` TODO is checked against.
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
                                'type': 'semantic_vad',
                                'create_response': True,
                                'interrupt_response': True,
                            },
                        },
                        'output': {'format': {'type': 'audio/pcm'}},
                    },
                    'model': 'eleven_realtime_v1_mini',
                },
            }
        ]
    )
    assert collapse_event_types(events) == snapshot(
        ['PartStartEvent', 'PartDeltaEvent', 'PartEndEvent', 'RealtimeTurnCompleteEvent']
    )
    messages = session.all_messages()
    assert [type(m).__name__ for m in messages] == snapshot(['ModelRequest', 'ModelResponse'])
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    # The server-reported model, from `session.created`.
    assert response.model_name == snapshot('eleven_realtime_v1_mini')
    part = response.parts[0]
    assert isinstance(part, SpeechPart)
    assert part.transcript == snapshot('Hello there')
    # `response.done` carries no usage.
    assert session.usage == snapshot(RunUsage(requests=1))


async def test_audio_in_semantic_vad_turn(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette], assets_path: Path
) -> None:
    """A spoken user turn (audio in, semantic VAD) is transcribed into a user turn in history."""
    provider, _ = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(MODEL, provider=provider)
    agent = Agent(instructions='Reply in a few words.')
    pcm = assets_path.joinpath('marcelo_24khz.pcm').read_bytes()

    events: list[Any] = []
    async with agent.realtime(model).session() as session:
        # Stream the clip in ~100 ms chunks like a live mic; the trailing silence lets server VAD end it.
        for start in range(0, len(pcm), 4800):
            await session.send_audio(pcm[start : start + 4800])
        await _until_turn_complete(session, events, timeout=45)

    # Speech start -> stop -> user turn -> assistant reply; the user deltas show incremental vs cumulative
    # transcription (see `map_event`).
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
    user_parts = [
        part for message in session.all_messages() if isinstance(message, ModelRequest) for part in message.parts
    ]
    assert len(user_parts) == 1
    user_part = user_parts[0]
    assert isinstance(user_part, SpeechPart) and user_part.speaker == 'user'
    assert user_part.transcript == snapshot('Hello, my name is Marcelo.')


@pytest.mark.realtime_ws_hold_open
async def test_push_to_talk_replies_only_when_asked(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette], assets_path: Path, realtime_recording: bool
) -> None:
    """With turn detection off, committed audio gets a reply only when `create_response()` asks for one.

    This is the test that caught xAI answering a commit by itself and dropping a second `response.create`.
    """
    provider, cassette = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(
        MODEL, provider=provider, settings=ElevenLabsRealtimeModelSettings(turn_detection=False)
    )
    agent = Agent(instructions='Reply in a few words.')
    pcm = assets_path.joinpath('marcelo_24khz.pcm').read_bytes()

    async with agent.realtime(model).session() as session:
        for start in range(0, len(pcm), 4800):
            await session.send_audio(pcm[start : start + 4800])
        await session.commit_audio()
        await session.send('Greet me by name.', respond=False)
        if realtime_recording:  # pragma: no cover
            # Long enough for a reply to the commit to have started, had the server answered it unasked.
            await anyio.sleep(3)
        await session.create_response()
        with anyio.fail_after(30):
            await session.wait_for_reply()
        await session.create_response()
        with anyio.fail_after(30):
            await session.wait_for_reply()

    interactions = [message for message in cassette.interactions if isinstance(message, CassetteMessage)]
    commit_at = next(
        index
        for index, message in enumerate(interactions)
        if message.direction == 'sent' and message.data['type'] == 'input_audio_buffer.commit'
    )
    created_types = [message.data['type'] for message in interactions[commit_at:] if message.direction == 'received']
    assert created_types.count('response.created') == 2
    assert [type(message).__name__ for message in session.all_messages()] == snapshot(
        ['ModelRequest', 'ModelRequest', 'ModelResponse', 'ModelResponse']
    )


async def test_tool_call_round(elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette]) -> None:
    """A tool call is executed by the session and its result folded back into a classic-shaped history."""
    provider, cassette = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(MODEL, provider=provider)
    agent = Agent(instructions='Use the get_weather tool for any weather question, then answer in one short sentence.')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return f'It is foggy and 12 degrees in {city}.'

    events: list[Any] = []
    seen_result = spoke_after_result = False
    async with agent.realtime(model).session() as session:
        await session.send('What is the weather in London?')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, FunctionToolResultEvent):
                    seen_result = True
                elif isinstance(event, PartStartEvent) and seen_result:
                    spoke_after_result = True
                elif isinstance(event, RealtimeTurnCompleteEvent) and spoke_after_result:
                    break

    # The tool schema as sent: check ElevenLabs accepts the strict JSON schema the OpenAI codec produces.
    assert sent_frames_containing(cassette, 'Look up the weather for a city.') == snapshot(
        [
            {
                'type': 'session.update',
                'session': {
                    'type': 'realtime',
                    'instructions': 'Use the get_weather tool for any weather question, then answer in one short sentence.',
                    'output_modalities': ['audio'],
                    'audio': {
                        'input': {
                            'format': {'type': 'audio/pcm', 'rate': 24000},
                            'turn_detection': {
                                'type': 'semantic_vad',
                                'create_response': True,
                                'interrupt_response': True,
                            },
                        },
                        'output': {'format': {'type': 'audio/pcm'}},
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
                    'model': 'eleven_realtime_v1_mini',
                },
            }
        ]
    )
    call_events = [e for e in events if isinstance(e, FunctionToolCallEvent)]
    assert [e.part.args_as_dict() for e in call_events] == [{'city': 'London'}]
    result_events = [e for e in events if isinstance(e, FunctionToolResultEvent)]
    assert len(result_events) == 1 and isinstance(result_events[0].part, ToolReturnPart)
    assert [type(m).__name__ for m in session.all_messages()] == snapshot(
        ['ModelRequest', 'ModelResponse', 'ModelRequest', 'ModelResponse']
    )
    final = session.all_messages()[-1]
    assert isinstance(final, ModelResponse)
    final_part = final.parts[0]
    assert isinstance(final_part, SpeechPart)
    assert 'fog' in (final_part.transcript or '').lower()


async def test_parallel_tool_calls_get_one_reply(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette],
) -> None:
    """Parallel tool calls come as back-to-back responses of one call each, and the reply to both completes the turn."""
    provider, _ = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(MODEL, provider=provider)
    agent = Agent(instructions='Use get_weather for every city asked about, in parallel. Answer in one short sentence.')

    # Run one after the other, so the results go out in the recorded order.
    @agent.tool_plain(sequential=True)
    def get_weather(city: str) -> str:
        """Look up the current weather for a city."""
        return f'It is foggy and 12 degrees in {city}.'

    events: list[Any] = []
    replies: list[ModelResponse] = []
    async with agent.realtime(model).session() as session:
        await session.send('What is the weather in Paris and in Berlin? Check both at once.')
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    final = session.all_messages()[-1]
                    assert isinstance(final, ModelResponse)
                    replies.append(final)
                    if len(replies) == 2:
                        break
                    # The next turn is tracked as usual.
                    await session.send('What is two plus two?')

    assert [e.part.args_as_dict() for e in events if isinstance(e, FunctionToolCallEvent)] == snapshot(
        [{'city': 'Paris'}, {'city': 'Berlin'}]
    )
    assert not any(isinstance(e, RealtimeSessionErrorEvent) for e in events)
    transcripts = [part.transcript for reply in replies for part in reply.parts if isinstance(part, SpeechPart)]
    assert transcripts == snapshot(
        [
            'The weather in Paris is foggy with a temperature of twelve degrees, and the same goes for Berlin, which is also foggy and at twelve degrees.',
            'Two plus two equals four.',
        ]
    )


async def test_message_history_seeding(elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette]) -> None:
    """Seeded prior turns are sent on the wire and reflected in the model's reply."""
    provider, cassette = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(MODEL, provider=provider)
    history = [
        ModelRequest(parts=[UserPromptPart(content='My name is Alice and my favorite color is teal.')]),
        ModelResponse(parts=[TextPart(content='Nice to meet you, Alice!')]),
    ]

    events: list[Any] = []
    async with Agent().realtime(model, message_history=history).session() as session:
        await session.send('What is my name and favorite color?')
        await _until_turn_complete(session, events)

    # A refused seed item (e.g. assistant `output_text`) surfaces as an error event: none may occur.
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
    reply = session.all_messages()[-1]
    assert isinstance(reply, ModelResponse)
    reply_part = reply.parts[0]
    assert isinstance(reply_part, SpeechPart)
    transcript = (reply_part.transcript or '').lower()
    assert 'alice' in transcript and 'teal' in transcript


async def test_reconnect_replays_history_after_drop(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette],
) -> None:
    """A forced WebSocket drop reconnects into a new session seeded with the local history."""
    provider, cassette = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(
        MODEL, provider=provider, settings={'reconnect': {'base_delay': 0.0, 'jitter': False}}
    )
    agent = Agent(instructions='Answer in one short sentence.')

    events: list[Any] = []
    disconnected = sent_followup = False
    async with agent.realtime(model).session() as session:
        await session.send('Remember exactly: the code word is cobalt. Briefly acknowledge it.')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent) and not disconnected:
                    disconnected = True
                    await cassette.disconnect()
                elif isinstance(event, RealtimeSessionReconnectEvent):
                    await session.send('What code word did I ask you to remember?')
                    sent_followup = True
                elif sent_followup and isinstance(event, RealtimeTurnCompleteEvent):
                    break

    assert [event for event in events if isinstance(event, RealtimeSessionReconnectEvent)] == [
        RealtimeSessionReconnectEvent(state_restored=True)
    ]
    close_index = next(
        i for i, interaction in enumerate(cassette.interactions) if isinstance(interaction, CassetteClose)
    )
    replayed_roles = [
        interaction.data['item']['role']
        for interaction in cassette.interactions[close_index + 1 :]
        if isinstance(interaction, CassetteMessage)
        and interaction.direction == 'sent'
        and interaction.data.get('type') == 'conversation.item.create'
        and 'What code word' not in str(interaction.data)
    ]
    assert replayed_roles == snapshot(['user', 'assistant'])
    final = session.all_messages()[-1]
    assert isinstance(final, ModelResponse)
    final_part = final.parts[0]
    assert isinstance(final_part, SpeechPart)
    assert 'cobalt' in (final_part.transcript or '').lower()


async def test_handle_barge_in_over_live_speech(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette], assets_path: Path
) -> None:
    """`handle_barge_in=True`: what goes on the wire when the user speaks over a reply."""
    provider, cassette = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(MODEL, provider=provider)
    agent = Agent(instructions='Reply in a few words.')
    pcm = assets_path.joinpath('marcelo_24khz.pcm').read_bytes()

    async with agent.realtime(model).session(handle_barge_in=True) as session:
        stream = session.stream_audio()
        with anyio.fail_after(90):
            for start in range(0, len(pcm), 4800):
                await session.send_audio(pcm[start : start + 4800])
            # Wait for the reply's audio to start flowing before speaking over it.
            assert len(await anext(stream)) > 0
            for start in range(0, len(pcm), 4800):
                await session.send_audio(pcm[start : start + 4800])
            turns_complete = 0
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    turns_complete += 1
                    if turns_complete == 2:
                        break

    assert sent_frames_containing(cassette, 'conversation.item.truncate') == snapshot(
        [
            {
                'type': 'conversation.item.truncate',
                'item_id': 'item_6801m5d26jvme4mv551f6aj74t5f',
                'content_index': 0,
                'audio_end_ms': 0,
            }
        ]
    )
    responses = [message for message in session.all_messages() if isinstance(message, ModelResponse)]
    assert [response.state for response in responses] == snapshot(['interrupted', 'complete'])
