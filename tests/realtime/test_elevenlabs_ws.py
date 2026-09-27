"""Cassette-backed tests for the ElevenLabs Agents realtime provider, exercising the real protocol.

These complement the network-free `test_elevenlabs.py` unit tests: the fakes there pin the
ElevenLabs-specific preflight and event mapping cheaply, while this replays a recorded agent
conversation end-to-end through [`Agent.realtime`][pydantic_ai.agent.Agent.realtime] to prove the
real protocol: the override initiation, the tool round-trip against a real workspace client tool,
the streamed audio and transcripts, the `agent_response` turn boundary, and the run-level usage.

Unlike the sibling providers, ElevenLabs wraps a *hosted agent*, so recording needs two purpose-built
dev agents (all override toggles enabled, `context_usage` added to `conversation.client_events`,
16 kHz PCM in and out, `turn.turn_timeout` raised to 30 s so the silence a microphone streams after
a short reply does not prompt an "are you still there" turn): one with a `get_weather` client tool
attached for the tool rounds, and one with no tools for the plain turns (the default
`elevenlabs_tool_sync='error'` would otherwise report the attached tool as undefined by the run).
Their ids are baked in below; point `ELEVENLABS_TEST_AGENT_ID` and `ELEVENLABS_TEST_TOOLLESS_AGENT_ID`
at your own agents to re-record. The REST preflight records through ordinary HTTP VCR alongside the
WebSocket cassette:

    uv run --env-file .env pytest --record-mode=rewrite --inline-snapshot=create tests/realtime/test_elevenlabs_ws.py

The `'sync'` mode tests each need a specific agent state at record time, and the mode lists every
workspace tool over REST, so record them one at a time (`-k`) on an otherwise empty workspace, in
this order: `creates` on a fresh toolless agent; delete that agent so its created tool becomes an
unattached orphan; `adopts` on another toolless agent (which attaches the orphan); `updates` on an
agent whose attached `get_weather` carries a stale description; `detaches` on an agent holding the
matching `get_weather`, an extra client tool and a webhook tool. Point the
`ELEVENLABS_TEST_SYNC_*_AGENT_ID` variables at your agents.
"""

from __future__ import annotations as _annotations

import json
import os
import re
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from inline_snapshot import snapshot
from vcr.cassette import Cassette

from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    RealtimeSessionErrorEvent,
    RetryPromptPart,
    SpeechPart,
    SpeechPartDelta,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.realtime import RealtimeTurnCompleteEvent

from ..conftest import IsDatetime, IsStr, try_import
from .conversation import Utterance, load_utterance, speak_continuously
from .ws_cassettes import RealtimeCassette
from .ws_helpers import collapse_event_types, sent_frames_containing

with try_import() as imports_successful:
    from pydantic_ai.providers.elevenlabs import ElevenLabsProvider
    from pydantic_ai.realtime.elevenlabs import ElevenLabsRealtimeModel, ElevenLabsRealtimeModelSettings

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.skipif(not imports_successful(), reason='websockets not installed'),
]

# The dev agents the cassettes were recorded against (deleted after recording); the ids are not
# secrets and pin the recorded REST paths for replay.
AGENT_ID = os.environ.get('ELEVENLABS_TEST_AGENT_ID', 'agent_2001m3hcjcrkfs1awbmpqytnqvew')
TOOLLESS_AGENT_ID = os.environ.get('ELEVENLABS_TEST_TOOLLESS_AGENT_ID', 'agent_1401m3hcjf30eqxtp3g5mczn16sr')
SYNC_CREATE_AGENT_ID = os.environ.get('ELEVENLABS_TEST_SYNC_CREATE_AGENT_ID', 'agent_3801m2dhe625eh8skp0kpebf0e5s')
SYNC_ADOPT_AGENT_ID = os.environ.get('ELEVENLABS_TEST_SYNC_ADOPT_AGENT_ID', 'agent_4901m2dhfaqnfcqrpbs2e1sk8xn6')
SYNC_UPDATE_AGENT_ID = os.environ.get('ELEVENLABS_TEST_SYNC_UPDATE_AGENT_ID', 'agent_0101m2dhhjp2e9h8001eznp7q483')
SYNC_DETACH_AGENT_ID = os.environ.get('ELEVENLABS_TEST_SYNC_DETACH_AGENT_ID', 'agent_1801m2dhhmqwezwaym92h2sp2ya0')

INSTRUCTIONS = 'Answer in one short sentence. Use the get_weather tool for any weather question.'


# The `context_tokens` of the two `context_usage` reports in the recording: the context the pipeline
# LLM held on each turn. Turn 2's is turn 1's plus the follow-up exchange, not a running total, so
# summing the reports into `input_tokens` counts what the LLM actually consumed.
_TURN_ONE_CONTEXT_TOKENS = 451
_TURN_TWO_CONTEXT_TOKENS = 459
_CONTEXT_LIMIT_TOKENS = 1_048_576


@pytest.mark.realtime_ws_hold_open
async def test_tool_round_and_followup_turn(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette], realtime_recording: bool
) -> None:
    """A text-in tool round against the hosted agent, then a follow-up turn.

    Covers, against real frames: the preflight tool comparison passing against the
    server-normalized stored schema (default `elevenlabs_tool_sync='error'`), the prompt-override
    initiation frame, `client_tool_call`/`client_tool_result`, streamed audio with the
    `agent_response` turn boundary, and one `context_usage` report per turn accumulating into
    run-level usage. Each report trails its turn boundary, so the test waits for the second one
    before closing, and pins both counts.
    """
    provider, cassette = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(AGENT_ID, provider=provider)
    agent = Agent(instructions=INSTRUCTIONS)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Look up the current weather for a city.

        Args:
            city: City name
        """
        return f'It is sunny and 21 degrees in {city}.'

    events: list[Any] = []
    turns = 0
    async with agent.realtime(model).session(audio_retention='output_audio') as session:
        await session.send('What is the weather in Berlin?')
        with anyio.fail_after(90):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    turns += 1
                    if turns == 1:
                        await session.send('Thanks, that is all.')
                    else:
                        break
            if realtime_recording:  # pragma: no cover  # only while recording
                await anyio.sleep(5)  # let the second report reach the recording
            while session.usage.input_tokens < _TURN_ONE_CONTEXT_TOKENS + _TURN_TWO_CONTEXT_TOKENS:
                await anyio.sleep(0.05)

    # No server-side rejection (a bad override or tool mismatch closes the socket with 1008 and
    # surfaces as a session error).
    assert [event for event in events if isinstance(event, RealtimeSessionErrorEvent)] == []

    # The initiation frame carried the instructions as the toggle-gated prompt override.
    assert sent_frames_containing(cassette, INSTRUCTIONS) == [
        {
            'type': 'conversation_initiation_client_data',
            'conversation_config_override': {'agent': {'prompt': {'prompt': INSTRUCTIONS}}},
        }
    ]
    # The session executed the workspace client tool's call and answered it on the wire.
    [tool_result_frame] = sent_frames_containing(cassette, 'client_tool_result')
    assert tool_result_frame['result'] == 'It is sunny and 21 degrees in Berlin.'
    assert tool_result_frame['is_error'] is False

    call_events = [event for event in events if isinstance(event, FunctionToolCallEvent)]
    result_events = [event for event in events if isinstance(event, FunctionToolResultEvent)]
    assert len(call_events) == 1
    assert call_events[0].part.tool_name == 'get_weather'
    assert call_events[0].part.args_as_dict() == {'city': 'Berlin'}
    assert len(result_events) == 1
    assert isinstance(result_events[0].part, ToolReturnPart)

    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == [
        'ModelRequest',
        'ModelResponse',
        'ModelRequest',
        'ModelResponse',
        'ModelRequest',
        'ModelResponse',
    ]
    assert messages[0] == ModelRequest(
        parts=[UserPromptPart(content='What is the weather in Berlin?', timestamp=IsDatetime())],
        timestamp=IsDatetime(),
        conversation_id=IsStr(),
        run_id=IsStr(),
    )
    tool_response = messages[1]
    assert isinstance(tool_response, ModelResponse)
    tool_calls = [part for part in tool_response.parts if isinstance(part, ToolCallPart)]
    assert len(tool_calls) == 1 and tool_calls[0].tool_name == 'get_weather'
    tool_return = messages[2]
    assert isinstance(tool_return, ModelRequest)
    assert isinstance(tool_return.parts[0], ToolReturnPart)
    answer = messages[3]
    assert isinstance(answer, ModelResponse)
    answer_part = answer.parts[0]
    assert isinstance(answer_part, SpeechPart)
    assert answer_part.speaker == 'assistant'
    assert answer_part.transcript is not None and 'sunny' in answer_part.transcript.lower()
    assert isinstance(answer_part.audio, BinaryContent)
    assert len(answer_part.audio.data) > 0
    # The conversation id from the handshake rides on the finalized response, so a consumer can
    # reconcile the conversation's cost from persisted history alone.
    assert answer.provider_details == {'conversation_id': IsStr()}

    # ElevenLabs reports LLM context consumption only (no output tokens or credits reach the
    # socket), once per turn *after* the turn boundary, so it accumulates into the run total
    # without attaching to a specific response: the two recorded reports sum into `input_tokens`.
    # The limit stays off the usage (see `ElevenLabsRealtimeConnection.context_limit_tokens`), and
    # the latest report against it is the window fraction.
    assert session.usage.input_tokens == _TURN_ONE_CONTEXT_TOKENS + _TURN_TWO_CONTEXT_TOKENS
    assert session.usage.output_tokens == 0
    assert session.usage.details == {}
    assert session.context_window_used == _TURN_TWO_CONTEXT_TOKENS / _CONTEXT_LIMIT_TOKENS
    assert all(isinstance(message, ModelResponse) and message.usage.input_tokens == 0 for message in messages[1::2])


async def test_tool_sync_off_hands_an_undefined_tool_call_to_the_tool_manager(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette],
) -> None:
    """With the preflight off, the agent calls a tool this run never defined.

    Nothing filters the call in the codec: the session's tool manager answers it with its standard
    unknown-tool error, the call and the error land in history like any other round, the error text
    goes back to the agent as the tool result, and the agent recovers in speech.
    """
    provider, cassette = elevenlabs_ws_cassette
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='off')
    model = ElevenLabsRealtimeModel(AGENT_ID, provider=provider, settings=settings)
    agent = Agent(instructions=INSTRUCTIONS)

    events: list[Any] = []
    async with agent.realtime(model).session() as session:
        await session.send('What is the weather in Berlin?')
        with anyio.fail_after(90):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent) and any(
                    isinstance(seen, FunctionToolResultEvent) for seen in events
                ):
                    break

    assert [event for event in events if isinstance(event, RealtimeSessionErrorEvent)] == []
    results = [event for event in events if isinstance(event, FunctionToolResultEvent)]
    assert len(results) >= 1
    for result in results:
        assert isinstance(result.part, RetryPromptPart)
        assert 'Unknown tool name' in str(result.part.content) and 'get_weather' in str(result.part.content)
    # The tool manager's error text is what the agent received, as an ordinary result (the codec's
    # `ToolResult` carries no error flag).
    wire_results = sent_frames_containing(cassette, 'client_tool_result')
    assert len(wire_results) == len(results)
    assert all('Unknown tool name' in frame['result'] for frame in wire_results)

    messages = session.all_messages()
    calls = [
        part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    assert [call.tool_name for call in calls] == ['get_weather'] * len(results)
    retries = [part for message in messages for part in message.parts if isinstance(part, RetryPromptPart)]
    assert len(retries) == len(results)
    final = messages[-1]
    assert isinstance(final, ModelResponse)
    assert isinstance(final.parts[-1], SpeechPart) and final.parts[-1].transcript


async def test_text_in_audio_out_turn(elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette]) -> None:
    """The simplest session: a text turn answered with streamed audio and a final transcript."""
    provider, cassette = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(TOOLLESS_AGENT_ID, provider=provider)
    agent = Agent(instructions='Answer in two or three words.')

    events: list[Any] = []
    async with agent.realtime(model).session(audio_retention='output_audio') as session:
        await session.send('Say hi.')
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    assert [event for event in events if isinstance(event, RealtimeSessionErrorEvent)] == []
    assert sent_frames_containing(cassette, 'Answer in two or three words.') == [
        {
            'type': 'conversation_initiation_client_data',
            'conversation_config_override': {'agent': {'prompt': {'prompt': 'Answer in two or three words.'}}},
        }
    ]
    assert collapse_event_types(events) == snapshot(
        ['PartStartEvent', 'PartDeltaEvent', 'PartEndEvent', 'RealtimeTurnCompleteEvent']
    )

    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == ['ModelRequest', 'ModelResponse']
    assert messages[0] == ModelRequest(
        parts=[UserPromptPart(content='Say hi.', timestamp=IsDatetime())],
        timestamp=IsDatetime(),
        conversation_id=IsStr(),
        run_id=IsStr(),
    )
    response = messages[1]
    assert isinstance(response, ModelResponse)
    assert response.model_name == TOOLLESS_AGENT_ID
    assert response.provider_details == {'conversation_id': IsStr()}
    [part] = response.parts
    assert isinstance(part, SpeechPart)
    assert part.speaker == 'assistant'
    assert part.transcript == snapshot('Hello there.')
    assert isinstance(part.audio, BinaryContent)
    assert len(part.audio.data) > 0


async def test_audio_in_server_vad_turn(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette], assets_path: Path
) -> None:
    """A spoken user turn: the agent's ASR transcribes it, its turn model ends it, and it answers.

    ElevenLabs owns turn-taking, so there is no commit or VAD event on the client side: the user's
    utterance surfaces as one final transcript once the server has decided the turn is over.
    """
    provider, cassette = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(TOOLLESS_AGENT_ID, provider=provider)
    agent = Agent(instructions='Reply in a few words.')
    # 16 kHz mono PCM16, the agent's configured input format; a second of trailing silence lets the
    # server-side turn model close the utterance.
    pcm = assets_path.joinpath('marcelo_16khz.pcm').read_bytes() + bytes(32_000)

    events: list[Any] = []
    async with agent.realtime(model).session(audio_retention='output_audio') as session:
        # Stream the clip in ~100 ms chunks like a live mic.
        for start in range(0, len(pcm), 3200):
            await session.send_audio(pcm[start : start + 3200])
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    assert [event for event in events if isinstance(event, RealtimeSessionErrorEvent)] == []
    assert len(sent_frames_containing(cassette, 'user_audio_chunk')) > 0
    assert collapse_event_types(events) == snapshot(
        [
            'PartStartEvent',
            'PartDeltaEvent',
            'PartEndEvent',
            'PartStartEvent',
            'PartDeltaEvent',
            'PartEndEvent',
            'RealtimeTurnCompleteEvent',
        ]
    )

    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == snapshot(['ModelRequest', 'ModelResponse'])
    spoken = [
        (part.speaker, part.transcript)
        for message in messages
        for part in message.parts
        if isinstance(part, SpeechPart)
    ]
    assert spoken == snapshot(
        [('user', 'Hello, my name is Marcelo.'), ('assistant', 'Hello Marcelo. How can I help you?')]
    )


# Long enough for the agent to be into its story, short enough that it is still being generated, so
# the interruption lands before `agent_response` closes the turn.
_SILENCE_BEFORE_BARGE_IN = 2.0
_SILENCE_AFTER_BARGE_IN = 8.0


@pytest.mark.realtime_ws_hold_open
async def test_barge_in_while_the_response_is_open_finalizes_the_truncated_reply(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette],
    assets_path: Path,
    realtime_recording: bool,
) -> None:
    """Speaking over a reply that is still being generated.

    Recorded fact: when the barge-in lands before the reply text is complete, the server sends
    `interruption` and then an `agent_response` already cut down to what the user heard (`Elias...`),
    with no `agent_response_correction` at all; the correction frame belongs to a barge-in during
    playback of a finished reply. So the interrupted response is finalized with the truncated text as
    its transcript, marked interrupted, and carries no correction, and the next turn is unaffected.
    """
    provider, cassette = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(TOOLLESS_AGENT_ID, provider=provider)
    agent = Agent(
        instructions='You are a voice assistant. Asked for a story, tell it at length straight away, without questions.'
    )
    rate = 16000
    story = load_utterance(assets_path, Utterance('tell_me_a_long_story', keyword='story'), rate)
    goodbye = load_utterance(assets_path, Utterance('stop_and_say_goodbye', keyword='goodbye'), rate)

    async with agent.realtime(model).session() as session:
        await speak_continuously(
            session,
            [story],
            sample_rate=rate,
            silence_after=_SILENCE_BEFORE_BARGE_IN,
            before_send=cassette.before_audio_send,
            pace=realtime_recording,
        )
        await speak_continuously(
            session,
            [goodbye],
            sample_rate=rate,
            silence_after=_SILENCE_AFTER_BARGE_IN,
            before_send=cassette.before_audio_send,
            pace=realtime_recording,
        )
        with anyio.fail_after(30):
            await session.wait_for_reply()

    messages = session.all_messages()
    responses = [message for message in messages if isinstance(message, ModelResponse)]
    assert len(responses) == 2
    story_response, goodbye_response = responses
    assert story_response.state == 'interrupted'
    assert story_response.provider_details == {'conversation_id': IsStr()}
    story_part = story_response.parts[-1]
    assert isinstance(story_part, SpeechPart)
    assert story_part.transcript == snapshot('Elias...')
    assert goodbye_response.state != 'interrupted'
    assert goodbye_response.provider_details == {'conversation_id': IsStr()}
    goodbye_part = goodbye_response.parts[-1]
    assert isinstance(goodbye_part, SpeechPart) and goodbye_part.transcript == snapshot('Goodbye.')
    spoken = [
        (part.speaker, part.transcript)
        for message in messages
        for part in message.parts
        if isinstance(part, SpeechPart) and part.speaker == 'user'
    ]
    assert len(spoken) == 2
    assert 'story' in (spoken[0][1] or '').lower() and 'goodbye' in (spoken[1][1] or '').lower()


async def test_text_output_modality_returns_text(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette],
) -> None:
    """`output_modality='text'` maps to the toggle-gated text-only override and streams text parts."""
    provider, cassette = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(
        TOOLLESS_AGENT_ID, provider=provider, settings=ElevenLabsRealtimeModelSettings(output_modality='text')
    )
    agent = Agent(instructions='Answer in two or three words.')

    events: list[Any] = []
    async with agent.realtime(model).session() as session:
        await session.send('Say hi.')
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    assert [event for event in events if isinstance(event, RealtimeSessionErrorEvent)] == []
    [initiation] = sent_frames_containing(cassette, 'Answer in two or three words.')
    assert initiation['conversation_config_override'] == {
        'agent': {'prompt': {'prompt': 'Answer in two or three words.'}},
        'conversation': {'text_only': True},
    }
    assert collapse_event_types(events) == snapshot(
        ['PartStartEvent', 'PartDeltaEvent', 'PartEndEvent', 'RealtimeTurnCompleteEvent']
    )

    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == ['ModelRequest', 'ModelResponse']
    response = messages[1]
    assert isinstance(response, ModelResponse)
    [part] = response.parts
    assert isinstance(part, TextPart)
    assert part.content == snapshot('Hello there.')
    assert not any(isinstance(event, PartDeltaEvent) and isinstance(event.delta, SpeechPartDelta) for event in events)


async def test_settings_overrides_reach_the_initiation_frame(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette],
) -> None:
    """Each `elevenlabs_*` override rides in the initiation frame, and the first message is spoken.

    A configured first message makes the agent speak before any user input, so the session's first
    turn is that greeting; a text turn then follows to prove the overridden conversation still works.
    """
    provider, cassette = elevenlabs_ws_cassette
    settings = ElevenLabsRealtimeModelSettings(
        elevenlabs_first_message='Welcome to the demo.',
        elevenlabs_language='en',
        elevenlabs_llm='gemini-2.5-flash',
        elevenlabs_voice_id='EXAVITQu4vr4xnSDxMaL',
        elevenlabs_tts={'stability': 0.5, 'speed': 1.0},
    )
    model = ElevenLabsRealtimeModel(TOOLLESS_AGENT_ID, provider=provider, settings=settings)
    agent = Agent(instructions='Answer in two or three words.')

    events: list[Any] = []
    turns = 0
    async with agent.realtime(model).session(audio_retention='output_audio') as session:
        with anyio.fail_after(90):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    turns += 1
                    if turns == 1:
                        await session.send('Say hi.')
                    else:
                        break

    assert [event for event in events if isinstance(event, RealtimeSessionErrorEvent)] == []
    [initiation] = sent_frames_containing(cassette, 'Welcome to the demo.')
    assert initiation['conversation_config_override'] == snapshot(
        {
            'agent': {
                'prompt': {'prompt': 'Answer in two or three words.', 'llm': 'gemini-2.5-flash'},
                'first_message': 'Welcome to the demo.',
                'language': 'en',
            },
            'tts': {'voice_id': 'EXAVITQu4vr4xnSDxMaL', 'stability': 0.5, 'speed': 1.0},
        }
    )

    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == snapshot(
        ['ModelResponse', 'ModelRequest', 'ModelResponse']
    )
    spoken = [
        (part.speaker, part.transcript)
        for message in messages
        for part in message.parts
        if isinstance(part, SpeechPart)
    ]
    assert spoken == snapshot([('assistant', 'Welcome to the demo.'), ('assistant', 'Hello there.')])


def _weather_agent() -> Agent:
    agent = Agent(instructions=INSTRUCTIONS)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Look up the current weather for a city.

        Args:
            city: City name
        """
        return f'It is sunny and 21 degrees in {city}.'

    return agent


async def _weather_turn(agent: Agent, model: ElevenLabsRealtimeModel) -> tuple[list[Any], list[Any]]:
    """One text turn that triggers `get_weather`; returns the session events and the history."""
    events: list[Any] = []
    async with agent.realtime(model).session(audio_retention='output_audio') as session:
        await session.send('What is the weather in Berlin?')
        with anyio.fail_after(90):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break
    return events, session.all_messages()


def _assert_weather_round(events: list[Any], messages: list[Any], cassette: RealtimeCassette) -> None:
    assert [event for event in events if isinstance(event, RealtimeSessionErrorEvent)] == []
    [call] = [event for event in events if isinstance(event, FunctionToolCallEvent)]
    assert call.part.tool_name == 'get_weather'
    assert call.part.args_as_dict() == {'city': 'Berlin'}
    [tool_result_frame] = sent_frames_containing(cassette, 'client_tool_result')
    assert tool_result_frame['result'] == 'It is sunny and 21 degrees in Berlin.'
    answer = messages[-1]
    assert isinstance(answer, ModelResponse)
    [answer_part] = answer.parts
    assert isinstance(answer_part, SpeechPart)
    assert answer_part.transcript is not None and 'sunny' in answer_part.transcript.lower()


def _rest_calls(vcr: Cassette) -> list[str]:
    """The REST preflight as recorded: method and path, ids replaced so the order is the assertion."""
    calls: list[str] = []
    for request in _requests(vcr):
        path = re.sub(r'/(agent|tool)_[0-9a-z]+', r'/<\1_id>', request.path)
        calls.append(f'{request.method} {path}')
    return calls


def _requests(vcr: Cassette) -> list[Any]:
    return cast('list[Any]', vcr.requests)  # pyright: ignore[reportUnknownMemberType]


def _rest_body(vcr: Cassette, index: int) -> dict[str, Any]:
    body = _requests(vcr)[index].body
    assert isinstance(body, bytes)
    return json.loads(body)


async def test_tool_sync_creates_and_attaches_on_a_toolless_agent(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette], vcr: Cassette
) -> None:
    """`'sync'` on an agent with no tools: the run's tool is created, the agent re-pointed, the round works."""
    provider, cassette = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(
        SYNC_CREATE_AGENT_ID, provider=provider, settings=ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    )
    events, messages = await _weather_turn(_weather_agent(), model)

    assert _rest_calls(vcr) == snapshot(
        [
            'GET /v1/convai/agents/<agent_id>',
            'GET /v1/convai/tools',
            'POST /v1/convai/tools',
            'PATCH /v1/convai/agents/<agent_id>',
            'GET /v1/convai/conversation/get-signed-url',
        ]
    )
    created = _rest_body(vcr, 2)
    assert created == snapshot(
        {
            'tool_config': {
                'type': 'client',
                'name': 'get_weather',
                'description': 'Look up the current weather for a city.',
                'expects_response': True,
                'parameters': {
                    'type': 'object',
                    'properties': {'city': {'type': 'string', 'description': 'City name'}},
                    'required': ['city'],
                },
            }
        }
    )
    repointed = _rest_body(vcr, 3)
    assert repointed == {'conversation_config': {'agent': {'prompt': {'tool_ids': [IsStr(regex=r'tool_[0-9a-z]+')]}}}}
    _assert_weather_round(events, messages, cassette)


async def test_tool_sync_adopts_a_matching_unattached_tool(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette], vcr: Cassette
) -> None:
    """A workspace tool that already matches and no agent depends on is attached instead of duplicated."""
    provider, cassette = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(
        SYNC_ADOPT_AGENT_ID, provider=provider, settings=ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    )
    events, messages = await _weather_turn(_weather_agent(), model)

    # The orphan's dependents are checked, then the agent is re-pointed at it; nothing is created.
    assert _rest_calls(vcr) == snapshot(
        [
            'GET /v1/convai/agents/<agent_id>',
            'GET /v1/convai/tools',
            'GET /v1/convai/tools/<tool_id>/dependent-agents',
            'PATCH /v1/convai/agents/<agent_id>',
            'GET /v1/convai/conversation/get-signed-url',
        ]
    )
    repointed = _rest_body(vcr, 3)
    assert repointed == {'conversation_config': {'agent': {'prompt': {'tool_ids': [IsStr(regex=r'tool_[0-9a-z]+')]}}}}
    _assert_weather_round(events, messages, cassette)


async def test_tool_sync_updates_a_stale_tool_in_place(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette], vcr: Cassette
) -> None:
    """An attached tool whose description drifted is patched under its id; the agent is left alone."""
    provider, cassette = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(
        SYNC_UPDATE_AGENT_ID, provider=provider, settings=ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    )
    events, messages = await _weather_turn(_weather_agent(), model)

    assert _rest_calls(vcr) == snapshot(
        [
            'GET /v1/convai/agents/<agent_id>',
            'GET /v1/convai/tools',
            'PATCH /v1/convai/tools/<tool_id>',
            'GET /v1/convai/conversation/get-signed-url',
        ]
    )
    patched = _rest_body(vcr, 2)
    assert patched['tool_config']['description'] == 'Look up the current weather for a city.'
    _assert_weather_round(events, messages, cassette)


async def test_tool_sync_detaches_extra_client_tools_and_keeps_server_tools(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette], vcr: Cassette
) -> None:
    """A client tool the run does not define is detached; the agent's webhook tool stays attached first."""
    provider, cassette = elevenlabs_ws_cassette
    model = ElevenLabsRealtimeModel(
        SYNC_DETACH_AGENT_ID, provider=provider, settings=ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    )
    events, messages = await _weather_turn(_weather_agent(), model)

    # No create and no tool patch: the matching `get_weather` is kept, `close_widget` is dropped from
    # `tool_ids`, and the webhook tool is preserved ahead of it. Nothing is deleted from the workspace.
    assert _rest_calls(vcr) == snapshot(
        [
            'GET /v1/convai/agents/<agent_id>',
            'GET /v1/convai/tools',
            'PATCH /v1/convai/agents/<agent_id>',
            'GET /v1/convai/conversation/get-signed-url',
        ]
    )
    repointed = _rest_body(vcr, 2)
    assert repointed == {
        'conversation_config': {
            'agent': {'prompt': {'tool_ids': [IsStr(regex=r'tool_[0-9a-z]+'), IsStr(regex=r'tool_[0-9a-z]+')]}}
        }
    }
    _assert_weather_round(events, messages, cassette)
