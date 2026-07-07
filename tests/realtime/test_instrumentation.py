"""Tests for `RealtimeSession` OpenTelemetry instrumentation, using a plain in-memory exporter."""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import pytest

from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.realtime import (
    AudioDelta,
    InputTranscript,
    RealtimeConnection,
    RealtimeEvent,
    RealtimeInput,
    RealtimeSession,
    SessionUsage,
    ToolCall,
    Transcript,
    TurnComplete,
)
from pydantic_ai.usage import RequestUsage

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytestmark = pytest.mark.anyio


class _Connection(RealtimeConnection):
    """Replays a fixed list of events; records nothing of interest for these tests."""

    def __init__(self, events: list[RealtimeEvent]) -> None:
        self._events = events

    async def send(self, content: RealtimeInput) -> None:
        pass

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        for event in self._events:
            yield event


def _settings(
    *, include_content: bool = True, use_aggregated_usage_attribute_names: bool = True
) -> tuple[InstrumentationSettings, InMemorySpanExporter]:
    pytest.importorskip('opentelemetry.sdk')  # only installed via the optional `logfire` extra
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    settings = InstrumentationSettings(
        tracer_provider=provider,
        include_content=include_content,
        use_aggregated_usage_attribute_names=use_aggregated_usage_attribute_names,
    )
    return settings, exporter


async def _ok_runner(name: str, args: dict[str, Any], call_id: str) -> str:
    return 'sunny'


async def test_nested_agent_run_nests_under_session_span() -> None:
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    settings, exporter = _settings()
    sub = Agent(TestModel())
    sub.instrument = settings

    async def runner(name: str, args: dict[str, Any], call_id: str) -> str:
        result = await sub.run('hi')
        return str(result.output)

    conn = _Connection([ToolCall(tool_call_id='c', tool_name='analyze', args='{}'), TurnComplete()])
    session = RealtimeSession(conn, runner, instrumentation=settings, model_name='gpt-realtime')
    _ = [e async for e in session]

    by_id = {s.context.span_id: s for s in exporter.get_finished_spans() if s.context is not None}
    session_span = next(s for s in by_id.values() if s.name == 'realtime gpt-realtime')
    tool_span = next(s for s in by_id.values() if s.name == 'execute_tool analyze')
    # the delegated sub-agent run is a real root agent span, nested under the tool span
    agent_span = next(s for s in by_id.values() if s.name.startswith('agent run') or s.name.startswith('invoke_agent'))

    assert session_span.context is not None
    assert tool_span.parent is not None and tool_span.parent.span_id == session_span.context.span_id
    # walk the sub-agent span's ancestry up to the tool span
    ancestor = by_id.get(agent_span.parent.span_id) if agent_span.parent else None
    assert ancestor is tool_span


async def test_session_captures_transcript_messages() -> None:
    # The session span reuses the shared message → gen_ai serialization on the finalized history:
    # the user request lands as an input message, the assistant response as an output message.
    settings, exporter = _settings()
    conn = _Connection(
        [
            InputTranscript(text='hello there', is_final=True),
            Transcript(text='hi, how can I help?', is_final=True),
            TurnComplete(),
        ]
    )
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = [e async for e in session]

    sess = next(s for s in exporter.get_finished_spans() if s.name == 'realtime gpt-realtime')
    assert sess.attributes is not None
    assert json.loads(str(sess.attributes['gen_ai.input.messages'])) == [
        {'role': 'user', 'parts': [{'type': 'text', 'content': 'hello there'}]},
    ]
    assert json.loads(str(sess.attributes['gen_ai.output.messages'])) == [
        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'hi, how can I help?'}]},
    ]


async def test_include_content_false_omits_transcript_messages() -> None:
    settings, exporter = _settings(include_content=False)
    conn = _Connection([InputTranscript(text='secret', is_final=True), TurnComplete()])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = [e async for e in session]
    sess = next(s for s in exporter.get_finished_spans() if s.name == 'realtime gpt-realtime')
    assert sess.attributes is not None
    assert 'gen_ai.input.messages' not in sess.attributes


async def test_session_and_tool_spans_with_usage() -> None:
    settings, exporter = _settings()
    conn = _Connection(
        [
            ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}'),
            SessionUsage(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            TurnComplete(),
        ]
    )
    session = RealtimeSession(
        conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime', agent_name='assistant'
    )
    _ = [e async for e in session]

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {'realtime gpt-realtime', 'chat gpt-realtime', 'execute_tool get_weather'}

    sess = spans['realtime gpt-realtime']
    assert sess.attributes is not None
    assert sess.attributes['gen_ai.operation.name'] == 'realtime'
    assert sess.attributes['gen_ai.request.model'] == 'gpt-realtime'
    assert sess.attributes['gen_ai.agent.name'] == 'assistant'
    # Cumulative usage on the session span uses the aggregated namespace (mirroring the classic
    # agent-run span) so it isn't double-counted against the per-turn `chat` spans' `gen_ai.usage.*`.
    assert sess.attributes['gen_ai.aggregated_usage.input_tokens'] == 10
    assert sess.attributes['gen_ai.aggregated_usage.output_tokens'] == 4

    tool = spans['execute_tool get_weather']
    assert tool.attributes is not None
    assert tool.attributes['gen_ai.tool.name'] == 'get_weather'
    assert tool.attributes['gen_ai.tool.call.id'] == 'c1'
    assert tool.attributes['gen_ai.tool.call.arguments'] == '{"city": "Paris"}'
    assert tool.attributes['gen_ai.tool.call.result'] == 'sunny'
    # Both the `chat` span and the `execute_tool` span are children of the session span (siblings),
    # matching the classic agent-run tree where `execute_tool` follows `chat` rather than nesting in it.
    chat = spans['chat gpt-realtime']
    assert sess.context is not None
    assert chat.parent is not None and chat.parent.span_id == sess.context.span_id
    assert tool.parent is not None and tool.parent.span_id == sess.context.span_id


async def test_include_content_false_omits_args_and_result() -> None:
    settings, exporter = _settings(include_content=False)
    conn = _Connection([ToolCall(tool_call_id='c', tool_name='f', args='{}'), TurnComplete()])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings)
    _ = [e async for e in session]
    tool = next(s for s in exporter.get_finished_spans() if s.name == 'execute_tool f')
    assert tool.attributes is not None
    assert 'gen_ai.tool.call.arguments' not in tool.attributes
    assert 'gen_ai.tool.call.result' not in tool.attributes


async def test_session_span_without_model_or_usage() -> None:
    settings, exporter = _settings()
    conn = _Connection([TurnComplete()])  # no model/agent name, no Usage event
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings)
    _ = [e async for e in session]
    sess = next(s for s in exporter.get_finished_spans() if s.name == 'realtime')
    assert sess.attributes is not None
    assert sess.attributes['gen_ai.operation.name'] == 'realtime'
    assert 'gen_ai.request.model' not in sess.attributes
    assert 'gen_ai.agent.name' not in sess.attributes
    assert 'gen_ai.usage.input_tokens' not in sess.attributes  # zero usage → no token attribute / metric
    # An empty turn produces no assistant `ModelResponse`, so no `chat` span is opened.
    assert not [s for s in exporter.get_finished_spans() if s.name.startswith('chat')]


async def test_chat_span_closed_for_contentless_response() -> None:
    # Audio with no transcript opens a `chat` span (first content) but finalizes with no response
    # parts, so the span closes without attaching messages.
    settings, exporter = _settings()
    conn = _Connection([AudioDelta(data=b'\x00\x01'), TurnComplete()])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = [e async for e in session]
    chat = next(s for s in exporter.get_finished_spans() if s.name == 'chat gpt-realtime')
    assert chat.attributes is not None
    assert 'gen_ai.output.messages' not in chat.attributes


async def test_session_usage_without_aggregated_attribute_names() -> None:
    # With `use_aggregated_usage_attribute_names=False`, cumulative session usage stays under the
    # standard `gen_ai.usage.*` namespace instead of the aggregated one.
    settings, exporter = _settings(use_aggregated_usage_attribute_names=False)
    conn = _Connection(
        [
            InputTranscript(text='hi', is_final=True),
            Transcript(text='hello'),
            SessionUsage(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            TurnComplete(),
        ]
    )
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = [e async for e in session]
    sess = next(s for s in exporter.get_finished_spans() if s.name == 'realtime gpt-realtime')
    assert sess.attributes is not None
    assert sess.attributes['gen_ai.usage.input_tokens'] == 10
    assert 'gen_ai.aggregated_usage.input_tokens' not in sess.attributes


async def test_chat_span_matches_instrumented_model_shape() -> None:
    """One `chat {model}` span per assistant response, with InstrumentedModel-parity attributes.

    The span reuses the same message → gen_ai serialization and response attributes as the classic
    model-request span (`open_model_request_span`): `gen_ai.operation.name='chat'`, request/response
    model, per-response `gen_ai.usage.*`, and input/output messages. Attributes a realtime session
    can't report honestly are omitted (documented on `_ensure_chat_span`), which this pins.
    """
    settings, exporter = _settings()
    conn = _Connection(
        [
            InputTranscript(text='hello there', is_final=True),
            Transcript(text='hi, how can I help?'),
            SessionUsage(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            TurnComplete(),
        ]
    )
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = [e async for e in session]

    chat = next(s for s in exporter.get_finished_spans() if s.name == 'chat gpt-realtime')
    assert chat.attributes is not None
    assert chat.attributes['gen_ai.operation.name'] == 'chat'
    assert chat.attributes['gen_ai.request.model'] == 'gpt-realtime'
    assert chat.attributes['gen_ai.response.model'] == 'gpt-realtime'
    # Per-response usage under the standard (non-aggregated) namespace, exactly as the classic path.
    assert chat.attributes['gen_ai.usage.input_tokens'] == 10
    assert chat.attributes['gen_ai.usage.output_tokens'] == 4
    # Input = the history slice the response replied to; output = the finalized assistant response.
    assert json.loads(str(chat.attributes['gen_ai.input.messages'])) == [
        {'role': 'user', 'parts': [{'type': 'text', 'content': 'hello there'}]},
    ]
    assert json.loads(str(chat.attributes['gen_ai.output.messages'])) == [
        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'hi, how can I help?'}]},
    ]
    assert 'gen_ai.input.messages' in json.loads(str(chat.attributes['logfire.json_schema']))['properties']
    # Honest omissions vs. the classic `chat` span: no provider/system, server address, request
    # parameters/settings, or cost (the session has only a model name, no provider or base URL).
    for omitted in (
        'gen_ai.provider.name',
        'gen_ai.system',
        'server.address',
        'model_request_parameters',
        'gen_ai.request.temperature',
        'operation.cost',
    ):
        assert omitted not in chat.attributes


async def test_chat_spans_split_on_tool_call_are_session_children() -> None:
    """A tool call splits a turn into two assistant responses → two `chat` spans; the tool runs between.

    Mirrors a classic run: the first `chat` span carries the assistant text plus the `ToolCallPart`,
    the `execute_tool` span follows as a sibling under the session, and the second `chat` span carries
    the post-tool response. All three are children of the session span.
    """
    settings, exporter = _settings()
    conn = _Connection(
        [
            InputTranscript(text='weather in Paris?', is_final=True),
            Transcript(text='let me check'),
            ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}'),
            Transcript(text='it is sunny'),
            SessionUsage(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            TurnComplete(),
        ]
    )
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = [e async for e in session]

    finished = exporter.get_finished_spans()
    sess = next(s for s in finished if s.name == 'realtime gpt-realtime')
    chats = [s for s in finished if s.name == 'chat gpt-realtime']
    tool = next(s for s in finished if s.name == 'execute_tool get_weather')
    assert len(chats) == 2
    assert sess.context is not None
    for span in (*chats, tool):
        assert span.parent is not None and span.parent.span_id == sess.context.span_id

    # First `chat` span: assistant text + the tool call it emitted.
    first, second = chats
    assert first.attributes is not None and second.attributes is not None
    assert json.loads(str(first.attributes['gen_ai.output.messages'])) == [
        {
            'role': 'assistant',
            'parts': [
                {'type': 'text', 'content': 'let me check'},
                {'type': 'tool_call', 'id': 'c1', 'name': 'get_weather', 'arguments': '{"city": "Paris"}'},
            ],
        },
    ]
    # Second `chat` span replies with the tool result folded into its input.
    assert {'type': 'tool_call_response', 'id': 'c1', 'name': 'get_weather', 'result': 'sunny'} in json.loads(
        str(second.attributes['gen_ai.input.messages'])
    )[-1]['parts']
    assert json.loads(str(second.attributes['gen_ai.output.messages'])) == [
        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'it is sunny'}]},
    ]


async def test_include_content_false_redacts_chat_span_messages() -> None:
    """With `include_content=False`, the `chat` span keeps the message envelope but drops content.

    This is the same redaction the classic model `chat` span applies (via the shared
    `handle_messages`): the message roles/structure remain for observability, but transcripts and
    tool arguments are omitted.
    """
    settings, exporter = _settings(include_content=False)
    conn = _Connection(
        [
            InputTranscript(text='my secret', is_final=True),
            Transcript(text='secret answer'),
            TurnComplete(),
        ]
    )
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = [e async for e in session]
    chat = next(s for s in exporter.get_finished_spans() if s.name == 'chat gpt-realtime')
    assert chat.attributes is not None
    # Envelope present, content redacted (no `content` key on the text parts).
    assert json.loads(str(chat.attributes['gen_ai.input.messages'])) == [
        {'role': 'user', 'parts': [{'type': 'text'}]},
    ]
    assert json.loads(str(chat.attributes['gen_ai.output.messages'])) == [
        {'role': 'assistant', 'parts': [{'type': 'text'}]},
    ]
    assert chat.attributes['gen_ai.response.model'] == 'gpt-realtime'
