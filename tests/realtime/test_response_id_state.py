"""Responses that overlap on the wire are each assembled and recorded under their own provider id.

These pin the scenarios from review of #8762, where the session assembled one response at a time, so a
late terminal, usage report or content frame for one response landed on whichever response was streaming.

The session core keeps a response per id, and the OpenAI-protocol connection (shared by Azure OpenAI and
xAI) brackets each response from its `response.created` to its `response.done`, so the first six run
over that connection's wire frames. Real OpenAI produces one of these overlaps: the `response.done` of a
reply the user barged in on can arrive after the next reply's `response.created`. The others are
hypothetical (no recording has them), and pin the per-id bookkeeping all the same.

The last one runs over a scripted connection whose lifecycle is inferred (as Gemini Live's and GPT-Live's
are), as only a provider that reports usage apart from its terminal can report usage for a response that
was recorded long ago.
"""

from __future__ import annotations as _annotations

import json
from typing import Any

import pytest

from pydantic_ai.messages import ModelResponse, SpeechPart, ToolCallPart
from pydantic_ai.realtime import RealtimeSession as _RealtimeSession
from pydantic_ai.realtime.codec import OutputTranscript, ResponseDone, SessionUsage
from pydantic_ai.usage import RequestUsage

from ..conftest import try_import
from .test_session import FakeRealtimeConnection, FakeRealtimeModel, RealtimeSession, collect_events

with try_import() as imports_successful:
    from pydantic_ai.realtime.openai import OpenAIRealtimeConnection

    from .test_openai import FakeWebSocket
    from .ws_helpers import collect_session_events

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]


def _summary(session: _RealtimeSession) -> list[tuple[str | None, list[str], int]]:
    """Each recorded response as its id, its spoken text or tool names, and its input tokens."""
    return [
        (
            message.provider_response_id,
            [
                part.transcript or '' if isinstance(part, SpeechPart) else part.tool_name
                for part in message.parts
                if isinstance(part, SpeechPart | ToolCallPart)
            ],
            message.usage.input_tokens,
        )
        for message in session.new_messages()
        if isinstance(message, ModelResponse)
    ]


def _created(response_id: str) -> dict[str, Any]:
    return {'type': 'response.created', 'response': {'id': response_id, 'status': 'in_progress', 'output': []}}


def _transcript(response_id: str, text: str, *, final: bool = True) -> dict[str, Any]:
    common = {'response_id': response_id, 'item_id': f'item_{response_id}', 'output_index': 0, 'content_index': 0}
    if final:
        return {'type': 'response.output_audio_transcript.done', **common, 'transcript': text}
    return {'type': 'response.output_audio_transcript.delta', **common, 'delta': text}


def _call(response_id: str, call_id: str) -> dict[str, Any]:
    return {
        'type': 'response.function_call_arguments.done',
        'response_id': response_id,
        'item_id': f'item_{call_id}',
        'output_index': 1,
        'call_id': call_id,
        'name': 'missing',
        'arguments': '{}',
    }


def _done(
    response_id: str, *, status: str = 'completed', calls: tuple[str, ...] = (), input_tokens: int | None = None
) -> dict[str, Any]:
    output = [
        {
            'id': f'item_{call_id}',
            'type': 'function_call',
            'call_id': call_id,
            'name': 'missing',
            'arguments': '{}',
            'status': 'completed',
        }
        for call_id in calls
    ]
    response: dict[str, Any] = {'id': response_id, 'status': status, 'output': output}
    if status == 'cancelled':
        response['status_details'] = {'type': 'cancelled', 'reason': 'turn_detected'}
    if input_tokens is not None:
        response['usage'] = {'total_tokens': input_tokens, 'input_tokens': input_tokens, 'output_tokens': 0}
    return {'type': 'response.done', 'response': response}


async def _run(frames: list[dict[str, Any]]) -> _RealtimeSession:
    connection = OpenAIRealtimeConnection(FakeWebSocket([json.dumps(frame) for frame in frames]))  # type: ignore[arg-type]
    session = RealtimeSession(connection, model=FakeRealtimeModel(connection, system='openai'))
    async with session:
        await collect_session_events(session)
    return session


async def test_late_usage_of_an_earlier_response_does_not_close_the_next() -> None:
    """A's late terminal neither closes B (which called a tool) nor lands its usage on B.

    The overlap real OpenAI produces: server VAD cancels A on a barge-in, and A's `response.done` arrives after B's
    `response.created`.
    """
    session = await _run(
        [
            _created('resp_a'),
            _created('resp_b'),
            _transcript('resp_b', 'Let me check.'),
            _call('resp_b', 'call_b'),
            _done('resp_a', status='cancelled', input_tokens=1),
            _done('resp_b', calls=('call_b',), input_tokens=10),
        ]
    )
    assert _summary(session) == [('resp_a', [], 1), ('resp_b', ['Let me check.', 'missing'], 10)]
    assert session.usage.input_tokens == 11


async def test_tool_call_response_superseded_mid_assembly_keeps_its_details() -> None:
    """A tool-call response whose terminal lands after the next response started is still recorded whole."""
    session = await _run(
        [
            _created('resp_a'),
            _call('resp_a', 'call_a'),
            _created('resp_b'),
            _transcript('resp_b', 'Meanwhile.'),
            _done('resp_a', calls=('call_a',), input_tokens=3),
            _done('resp_b'),
        ]
    )
    responses = [m for m in session.new_messages() if isinstance(m, ModelResponse)]
    assert [(r.provider_response_id, r.provider_details, r.finish_reason) for r in responses] == [
        ('resp_a', {'status': 'completed'}, 'tool_call'),
        ('resp_b', {'status': 'completed'}, 'stop'),
    ]
    assert _summary(session) == [('resp_a', ['missing'], 3), ('resp_b', ['Meanwhile.'], 0)]


async def test_recorded_responses_trailer_marker_does_not_swallow_the_next_terminal() -> None:
    """A tool-call response ending doesn't make the next response's terminal look redundant."""
    session = await _run(
        [
            _created('resp_a'),
            _call('resp_a', 'call_a'),
            _done('resp_a', calls=('call_a',), input_tokens=0),
            _created('resp_b'),
            _done('resp_b', status='cancelled'),
        ]
    )
    assert [r.provider_response_id for r in session.new_messages() if isinstance(r, ModelResponse)] == [
        'resp_a',
        'resp_b',
    ]


async def test_late_content_of_a_recorded_response_stays_out_of_the_next() -> None:
    """A delta still in flight when its response was recorded doesn't join the next response."""
    session = await _run(
        [
            _created('resp_a'),
            _transcript('resp_a', 'first'),
            _done('resp_a'),
            _created('resp_b'),
            _transcript('resp_b', 'second'),
            _transcript('resp_a', ' leftover', final=False),
            _done('resp_b'),
        ]
    )
    assert _summary(session) == [('resp_a', ['first'], 0), ('resp_b', ['second'], 0)]


async def test_interleaved_responses_are_assembled_separately() -> None:
    """Two responses streaming at once each keep their own content."""
    session = await _run(
        [
            _created('resp_a'),
            _created('resp_b'),
            _transcript('resp_a', 'one ', final=False),
            _transcript('resp_b', 'two', final=False),
            _transcript('resp_a', 'three', final=False),
            _done('resp_a'),
            _done('resp_b'),
        ]
    )
    assert _summary(session) == [('resp_a', ['one three'], 0), ('resp_b', ['two'], 0)]


async def test_late_tool_call_joins_its_own_response() -> None:
    """A tool call for a response still being assembled joins that response, not the one streaming now."""
    session = await _run(
        [
            _created('resp_a'),
            _transcript('resp_a', 'Checking.'),
            _created('resp_b'),
            _transcript('resp_b', 'Hello.'),
            _call('resp_a', 'call_a'),
            _done('resp_a', calls=('call_a',), input_tokens=0),
            _done('resp_b'),
        ]
    )
    assert _summary(session) == [('resp_a', ['Checking.', 'missing'], 0), ('resp_b', ['Hello.'], 0)]


async def test_late_usage_long_after_its_response_starts_nothing() -> None:
    """Usage for a response recorded many responses ago is counted, not taken for a new response."""
    events = [
        event
        for index in range(40)
        for event in (
            OutputTranscript(text=f'reply {index}', is_final=True, response_id=f'resp_{index}'),
            ResponseDone(provider_response_id=f'resp_{index}'),
        )
    ]
    conn = FakeRealtimeConnection(
        [*events, SessionUsage(usage=RequestUsage(input_tokens=7), provider_response_id='resp_0')]
    )
    session = RealtimeSession(conn)
    await collect_events(session)
    assert len([m for m in session.new_messages() if isinstance(m, ModelResponse)]) == 40
    assert session.usage.input_tokens == 7
