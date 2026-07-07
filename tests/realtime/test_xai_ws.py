"""Cassette-backed tests for the xAI Grok Voice realtime provider, exercising the real WebSocket protocol.

These complement the network-free `test_xai.py` unit tests: the fakes there pin the xAI-specific event
mapping, session config, and handshake cheaply, while these replay recorded provider frames end-to-end
through [`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session] to prove the real protocol —
the streamed part events, the tool round-trip, and message-history seeding.

Recording requires xAI realtime API access (`XAI_API_KEY` with the voice-agent capability); when the
cassette is missing offline the `xai_ws_cassette` fixture skips rather than errors.
"""

from __future__ import annotations as _annotations

from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import (
    AudioWithTranscriptPart,
    BinaryContent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.realtime import PartStartEvent, SessionError, TurnComplete

from ..conftest import IsDatetime, IsStr, try_import
from .ws_cassettes import RealtimeCassette
from .ws_helpers import collapse_event_types, sent_frames_containing

with try_import() as imports_successful:
    from pydantic_ai.providers.xai import XaiProvider
    from pydantic_ai.realtime.xai import XaiRealtimeModel

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='xai-sdk / websockets not installed'),
]

MODEL = 'grok-voice-latest'


async def test_text_in_audio_out_turn(xai_ws_cassette: tuple[XaiProvider, RealtimeCassette]) -> None:
    """A text-in turn yields streamed audio+transcript parts and a classic-shaped history."""
    provider, _ = xai_ws_cassette
    model = XaiRealtimeModel(MODEL, provider=provider)
    agent = Agent(instructions='Answer in two or three words.')

    events: list[Any] = []
    async with agent.realtime_session(model=model, audio_retention='output') as session:
        await session.send_text('Say a short greeting.')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch - the loop always breaks on TurnComplete
                events.append(event)
                if isinstance(event, TurnComplete):
                    break

    messages = session.all_messages()
    assert collapse_event_types(events) == snapshot(
        ['PartStartEvent', 'PartDeltaEvent', 'PartEndEvent', 'TurnComplete']
    )
    assert [type(m).__name__ for m in messages] == snapshot(['ModelRequest', 'ModelResponse'])
    assert messages[0] == ModelRequest(parts=[UserPromptPart(content='Say a short greeting.', timestamp=IsDatetime())])
    response = messages[1]
    assert isinstance(response, ModelResponse)
    assert response.model_name == MODEL
    part = response.parts[0]
    assert isinstance(part, AudioWithTranscriptPart)
    assert part.speaker == 'assistant'
    assert part.transcript == snapshot('Hello there!')
    assert isinstance(part.audio, BinaryContent)
    assert part.audio.media_type == 'audio/pcm'
    assert len(part.audio.data) > 0


async def test_tool_call_round(xai_ws_cassette: tuple[XaiProvider, RealtimeCassette]) -> None:
    """A tool call is executed by the session and its result folded back into a classic-shaped history.

    Unlike OpenAI in text mode, Grok Voice *speaks* before it calls a tool, so the tool call arrives in
    the same (mixed audio + function-call) response that fires the first `TurnComplete`; the model then
    speaks the answer in a second turn. The loop runs until the tool result has come back and the model
    has finished the follow-up turn.
    """
    provider, _ = xai_ws_cassette
    model = XaiRealtimeModel(MODEL, provider=provider)
    agent = Agent(instructions='Use the get_weather tool for any weather question, then answer in one short sentence.')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return f'It is foggy and 12 degrees in {city}.'

    events: list[Any] = []
    seen_result = spoke_after_result = False
    async with agent.realtime_session(model=model) as session:
        await session.send_text('What is the weather in London?')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch - the loop always breaks on TurnComplete
                events.append(event)
                # The tool call rides in the first (mixed) turn, so stop only once the model has spoken
                # a follow-up turn *after* the tool result — the actual answer.
                if isinstance(event, FunctionToolResultEvent):
                    seen_result = True
                elif isinstance(event, PartStartEvent) and seen_result:
                    spoke_after_result = True
                elif isinstance(event, TurnComplete) and spoke_after_result:
                    break

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
    # The tool call rides along with the assistant's spoken intro in the first response.
    tool_response = messages[1]
    assert isinstance(tool_response, ModelResponse)
    tool_calls = [p for p in tool_response.parts if isinstance(p, ToolCallPart)]
    assert tool_calls == [ToolCallPart(tool_name='get_weather', args=IsStr(), tool_call_id=IsStr())]
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
    assert isinstance(final_part, AudioWithTranscriptPart)
    assert final_part.transcript is not None and 'fog' in final_part.transcript.lower()


async def test_message_history_seeding(xai_ws_cassette: tuple[XaiProvider, RealtimeCassette]) -> None:
    """Seeded prior turns are sent on the wire and reflected in the model's reply."""
    provider, cassette = xai_ws_cassette
    model = XaiRealtimeModel(MODEL, provider=provider)
    agent = Agent()

    history = [
        ModelRequest(parts=[UserPromptPart(content='My name is Alice and my favorite color is teal.')]),
        ModelResponse(parts=[TextPart(content='Nice to meet you, Alice!')]),
    ]

    events: list[Any] = []
    async with agent.realtime_session(model=model, message_history=history) as session:
        await session.send_text('What is my name and favorite color?')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch - the loop always breaks on TurnComplete
                events.append(event)
                if isinstance(event, TurnComplete):
                    break

    # A server-side rejection of the seeded items (e.g. a bad content-type shape) surfaces as a
    # `SessionError`; assert none occurred so a broken seed payload fails the test loudly.
    assert [event for event in events if isinstance(event, SessionError)] == []

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
    assert sent_frames_containing(cassette, 'Nice to meet you')

    # `all_messages()` carries the seeded history ahead of this session's turns.
    messages = session.all_messages()
    assert messages[:2] == history
    reply = messages[-1]
    assert isinstance(reply, ModelResponse)
    reply_part = reply.parts[0]
    assert isinstance(reply_part, AudioWithTranscriptPart)
    transcript = (reply_part.transcript or '').lower()
    assert 'alice' in transcript and 'teal' in transcript
