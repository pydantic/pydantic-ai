"""Cassette-backed tests for the ElevenLabs Agents realtime provider, exercising the real protocol.

These complement the network-free `test_elevenlabs.py` unit tests: the fakes there pin the
ElevenLabs-specific preflight and event mapping cheaply, while this replays a recorded agent
conversation end-to-end through [`Agent.realtime`][pydantic_ai.agent.Agent.realtime] to prove the
real protocol: the override initiation, the tool round-trip against a real workspace client tool,
the streamed audio and transcripts, the `agent_response` turn boundary, and the run-level usage.

Unlike the sibling providers, ElevenLabs wraps a *hosted agent*, so recording needs a purpose-built
dev agent (all override toggles enabled, a `get_weather` client tool attached, `context_usage`
added to `conversation.client_events`) whose id is baked in below; point
`ELEVENLABS_TEST_AGENT_ID` at your own agent to re-record. The REST preflight records through
ordinary HTTP VCR alongside the WebSocket cassette:

    uv run --env-file .env pytest --record-mode=rewrite tests/realtime/test_elevenlabs_ws.py
"""

from __future__ import annotations as _annotations

import os
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    RealtimeSessionErrorEvent,
    SpeechPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.realtime import RealtimeTurnCompleteEvent

from ..conftest import IsDatetime, IsStr, try_import
from .ws_cassettes import RealtimeCassette
from .ws_helpers import sent_frames_containing

with try_import() as imports_successful:
    from pydantic_ai.providers.elevenlabs import ElevenLabsProvider
    from pydantic_ai.realtime.elevenlabs import ElevenLabsRealtimeModel

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.skipif(not imports_successful(), reason='websockets not installed'),
]

# The dev agent the cassettes were recorded against; the id is not a secret and pins the recorded
# REST paths for replay.
AGENT_ID = os.environ.get('ELEVENLABS_TEST_AGENT_ID', 'agent_3701m1hxk8qmer0v2p4grgjb6nre')

INSTRUCTIONS = 'Answer in one short sentence. Use the get_weather tool for any weather question.'


async def test_tool_round_and_followup_turn(
    elevenlabs_ws_cassette: tuple[ElevenLabsProvider, RealtimeCassette],
) -> None:
    """A text-in tool round against the hosted agent, then a follow-up turn.

    Covers, against real frames: the preflight tool comparison passing against the
    server-normalized stored schema (default `elevenlabs_tool_sync='error'`), the prompt-override
    initiation frame, `client_tool_call`/`client_tool_result`, streamed audio with the
    `agent_response` turn boundary, and `context_usage` accumulating into run-level usage (which is
    why the test runs a second turn: usage trails the turn boundary, so turn 1's report is consumed
    while turn 2 streams).
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
    # without attaching to a specific response.
    assert session.usage.input_tokens > 0
    assert session.usage.details.get('context_limit_tokens', 0) > 0
    assert session.usage.output_tokens == 0
