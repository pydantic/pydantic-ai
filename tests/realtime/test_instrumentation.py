"""Tests for `RealtimeSession` OpenTelemetry instrumentation, using a plain in-memory exporter."""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.realtime import (
    InputTranscript,
    RealtimeConnection,
    RealtimeEvent,
    RealtimeInput,
    RealtimeSession,
    ToolCall,
    Transcript,
    TurnComplete,
    Usage,
)
from pydantic_ai.usage import RequestUsage

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


def _settings(*, include_content: bool = True) -> tuple[InstrumentationSettings, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return InstrumentationSettings(tracer_provider=provider, include_content=include_content), exporter


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
    settings, exporter = _settings()
    conn = _Connection(
        [
            InputTranscript(text='hello there', is_final=True),
            Transcript(text='hi, how can I help?', is_final=True),
            InputTranscript(text='par', is_final=False),  # non-final → not captured
            TurnComplete(),
        ]
    )
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = [e async for e in session]

    sess = next(s for s in exporter.get_finished_spans() if s.name == 'realtime gpt-realtime')
    assert sess.attributes is not None
    messages = json.loads(str(sess.attributes['gen_ai.input.messages']))
    assert messages == [
        {'role': 'user', 'parts': [{'type': 'text', 'content': 'hello there'}]},
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
            Usage(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            TurnComplete(),
        ]
    )
    session = RealtimeSession(
        conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime', agent_name='assistant'
    )
    _ = [e async for e in session]

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {'realtime gpt-realtime', 'execute_tool get_weather'}

    sess = spans['realtime gpt-realtime']
    assert sess.attributes is not None
    assert sess.attributes['gen_ai.operation.name'] == 'realtime'
    assert sess.attributes['gen_ai.request.model'] == 'gpt-realtime'
    assert sess.attributes['gen_ai.agent.name'] == 'assistant'
    assert sess.attributes['gen_ai.usage.input_tokens'] == 10
    assert sess.attributes['gen_ai.usage.output_tokens'] == 4

    tool = spans['execute_tool get_weather']
    assert tool.attributes is not None
    assert tool.attributes['gen_ai.tool.name'] == 'get_weather'
    assert tool.attributes['gen_ai.tool.call.id'] == 'c1'
    assert tool.attributes['gen_ai.tool.call.arguments'] == '{"city": "Paris"}'
    assert tool.attributes['gen_ai.tool.call.result'] == 'sunny'
    # the tool span is a child of the session span
    assert tool.parent is not None
    assert sess.context is not None
    assert tool.parent.span_id == sess.context.span_id


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
