"""Responses that overlap on the wire are each assembled and recorded under their own provider id.

These pin the scenarios from review of #8762. The session assembles one response at a time, so a late
terminal, usage report or content event for one response lands on whichever response is streaming.
Each test is expected to fail until the session keeps its response state per response id.
"""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai.messages import ModelResponse, SpeechPart, ToolCallPart
from pydantic_ai.realtime import RealtimeSession as _RealtimeSession
from pydantic_ai.realtime.codec import OutputTranscript, ResponseDone, SessionUsage, ToolCall
from pydantic_ai.usage import RequestUsage

from .test_session import FakeRealtimeConnection, RealtimeSession, collect_events

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.xfail(strict=True, reason='The session assembles one response at a time, not per response id.'),
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


async def test_late_usage_of_an_earlier_response_does_not_close_the_next() -> None:
    """A's late usage neither closes B (awaiting its own usage for a tool call) nor lands in B's usage."""
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='Let me check.', is_final=True, response_id='resp_b'),
            ToolCall(
                tool_call_id='call_b', tool_name='missing', args='{}', response_usage_follows=True, response_id='resp_b'
            ),
            SessionUsage(usage=RequestUsage(input_tokens=1), provider_response_id='resp_a'),
            SessionUsage(usage=RequestUsage(input_tokens=10), provider_response_id='resp_b', finish_reason='tool_call'),
        ]
    )
    session = RealtimeSession(conn)
    await collect_events(session)
    assert _summary(session) == [('resp_b', ['Let me check.', 'missing'], 10)]
    assert session.usage.input_tokens == 11


async def test_tool_call_response_superseded_mid_assembly_keeps_its_details() -> None:
    """A tool-call response whose usage lands after the next response started is still recorded whole."""
    conn = FakeRealtimeConnection(
        [
            ToolCall(
                tool_call_id='call_a', tool_name='missing', args='{}', response_usage_follows=True, response_id='resp_a'
            ),
            OutputTranscript(text='Meanwhile.', is_final=True, response_id='resp_b'),
            SessionUsage(
                usage=RequestUsage(input_tokens=3),
                provider_response_id='resp_a',
                finish_reason='tool_call',
                provider_details={'status': 'completed'},
            ),
            ResponseDone(provider_response_id='resp_b', provider_details={'status': 'completed'}),
        ]
    )
    session = RealtimeSession(conn)
    await collect_events(session)
    responses = [m for m in session.new_messages() if isinstance(m, ModelResponse)]
    assert [(r.provider_response_id, r.provider_details, r.finish_reason) for r in responses] == [
        ('resp_a', {'status': 'completed'}, 'tool_call'),
        ('resp_b', {'status': 'completed'}, 'stop'),
    ]


async def test_recorded_responses_trailer_marker_does_not_swallow_the_next_terminal() -> None:
    """A tool-call response recorded from its usage doesn't make the next response's terminal look redundant."""
    conn = FakeRealtimeConnection(
        [
            ToolCall(
                tool_call_id='call_a', tool_name='missing', args='{}', response_usage_follows=True, response_id='resp_a'
            ),
            SessionUsage(usage=RequestUsage(), provider_response_id='resp_a', finish_reason='tool_call'),
            ResponseDone(interrupted=True, provider_response_id='resp_b', provider_details={'status': 'cancelled'}),
        ]
    )
    session = RealtimeSession(conn)
    await collect_events(session)
    assert [r.provider_response_id for r in session.new_messages() if isinstance(r, ModelResponse)] == [
        'resp_a',
        'resp_b',
    ]


async def test_late_content_of_a_recorded_response_stays_out_of_the_next() -> None:
    """A delta still in flight when its response was recorded doesn't join the next response."""
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='first', is_final=True, response_id='resp_a'),
            ResponseDone(provider_response_id='resp_a'),
            OutputTranscript(text='second', is_final=True, response_id='resp_b'),
            OutputTranscript(text=' leftover', response_id='resp_a'),
            ResponseDone(provider_response_id='resp_b'),
        ]
    )
    session = RealtimeSession(conn)
    await collect_events(session)
    assert _summary(session) == [('resp_a', ['first'], 0), ('resp_b', ['second'], 0)]


async def test_interleaved_responses_are_assembled_separately() -> None:
    """Two responses streaming at once (an out-of-band response, say) each keep their own content."""
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='one ', response_id='resp_a'),
            OutputTranscript(text='two', response_id='resp_b'),
            OutputTranscript(text='three', response_id='resp_a'),
            ResponseDone(provider_response_id='resp_a'),
            ResponseDone(provider_response_id='resp_b'),
        ]
    )
    session = RealtimeSession(conn)
    await collect_events(session)
    assert _summary(session) == [('resp_a', ['one three'], 0), ('resp_b', ['two'], 0)]


async def test_late_tool_call_joins_its_own_response() -> None:
    """A tool call for a response still being assembled joins that response, not the one streaming now."""
    conn = FakeRealtimeConnection(
        [
            OutputTranscript(text='Checking.', is_final=True, response_id='resp_a'),
            OutputTranscript(text='Hello.', is_final=True, response_id='resp_b'),
            ToolCall(
                tool_call_id='call_a', tool_name='missing', args='{}', response_usage_follows=True, response_id='resp_a'
            ),
            SessionUsage(usage=RequestUsage(), provider_response_id='resp_a', finish_reason='tool_call'),
            ResponseDone(provider_response_id='resp_b'),
        ]
    )
    session = RealtimeSession(conn)
    await collect_events(session)
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
