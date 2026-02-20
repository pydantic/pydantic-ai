"""Tests for realtime model OpenTelemetry instrumentation."""

from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire

from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.realtime import (
    InstrumentedRealtimeModel,
    ToolCall,
    ToolCallCompleted,
    ToolCallStarted,
    TurnComplete,
    instrument_realtime_model,
)
from pydantic_ai.tools import ToolDefinition

from .conftest import FakeRealtimeConnection, FakeRealtimeModel

pytestmark = pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')


@pytest.mark.anyio
async def test_session_span_attributes(capfire: CaptureLogfire) -> None:
    """Session span is created with correct name and OTel attributes."""
    conn = FakeRealtimeConnection([TurnComplete()])
    model = FakeRealtimeModel(conn)
    instrumented = InstrumentedRealtimeModel(model, InstrumentationSettings())

    async with instrumented.connect(instructions='Be helpful') as connection:
        events = [e async for e in connection]

    assert len(events) == 1
    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(s for s in spans if 'realtime' in s['name'])

    assert span['name'] == 'realtime fake-realtime'
    attrs = span['attributes']
    assert attrs['gen_ai.operation.name'] == 'realtime'
    assert attrs['gen_ai.provider.name'] == 'fake'
    assert attrs['gen_ai.system'] == 'fake'
    assert attrs['gen_ai.request.model'] == 'fake-realtime'


@pytest.mark.anyio
async def test_tool_definitions_in_span(capfire: CaptureLogfire) -> None:
    """Tool definitions appear in span attributes as JSON."""
    conn = FakeRealtimeConnection([TurnComplete()])
    model = FakeRealtimeModel(conn)
    instrumented = InstrumentedRealtimeModel(model, InstrumentationSettings())

    tools = [
        ToolDefinition(
            name='get_weather',
            description='Get the weather',
            parameters_json_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}},
        )
    ]

    async with instrumented.connect(instructions='', tools=tools) as connection:
        _ = [e async for e in connection]

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(s for s in spans if 'realtime' in s['name'])
    tool_defs = span['attributes']['gen_ai.tool.definitions']

    assert len(tool_defs) == 1
    assert tool_defs[0]['name'] == 'get_weather'
    assert tool_defs[0]['description'] == 'Get the weather'


@pytest.mark.anyio
async def test_system_instructions_included(capfire: CaptureLogfire) -> None:
    """System instructions appear when `include_content=True` (default)."""
    conn = FakeRealtimeConnection([TurnComplete()])
    model = FakeRealtimeModel(conn)
    instrumented = InstrumentedRealtimeModel(model, InstrumentationSettings(include_content=True))

    async with instrumented.connect(instructions='Talk like a pirate') as connection:
        _ = [e async for e in connection]

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(s for s in spans if 'realtime' in s['name'])
    assert span['attributes']['gen_ai.system_instructions'] == 'Talk like a pirate'


@pytest.mark.anyio
async def test_system_instructions_omitted_when_disabled(capfire: CaptureLogfire) -> None:
    """System instructions are omitted when `include_content=False`."""
    conn = FakeRealtimeConnection([TurnComplete()])
    model = FakeRealtimeModel(conn)
    instrumented = InstrumentedRealtimeModel(model, InstrumentationSettings(include_content=False))

    async with instrumented.connect(instructions='Secret instructions') as connection:
        _ = [e async for e in connection]

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(s for s in spans if 'realtime' in s['name'])
    assert 'gen_ai.system_instructions' not in span['attributes']


@pytest.mark.anyio
async def test_instrument_realtime_model_idempotent() -> None:
    """Wrapping an already-instrumented model returns it unchanged."""
    conn = FakeRealtimeConnection([])
    model = FakeRealtimeModel(conn)
    settings = InstrumentationSettings()

    wrapped = instrument_realtime_model(model, settings)
    assert isinstance(wrapped, InstrumentedRealtimeModel)

    double_wrapped = instrument_realtime_model(wrapped, settings)
    assert double_wrapped is wrapped


@pytest.mark.anyio
async def test_instrument_realtime_model_bool_creates_defaults() -> None:
    """`instrument_realtime_model(model, True)` creates default settings."""
    conn = FakeRealtimeConnection([])
    model = FakeRealtimeModel(conn)

    wrapped = instrument_realtime_model(model, True)
    assert isinstance(wrapped, InstrumentedRealtimeModel)


@pytest.mark.anyio
async def test_instrument_realtime_model_false_noop() -> None:
    """`instrument_realtime_model(model, False)` returns the model unchanged."""
    conn = FakeRealtimeConnection([])
    model = FakeRealtimeModel(conn)

    result = instrument_realtime_model(model, False)
    assert result is model


@pytest.mark.anyio
async def test_session_exception_recorded(capfire: CaptureLogfire) -> None:
    """Exceptions during the session are recorded on the span."""
    conn = FakeRealtimeConnection([])
    model = FakeRealtimeModel(conn)
    instrumented = InstrumentedRealtimeModel(model, InstrumentationSettings())

    with pytest.raises(ValueError, match='boom'):
        async with instrumented.connect(instructions='') as _connection:
            raise ValueError('boom')

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(s for s in spans if 'realtime' in s['name'])
    events = span.get('events', [])
    assert any(e['name'] == 'exception' for e in events)
    exception_event = next(e for e in events if e['name'] == 'exception')
    assert exception_event['attributes']['exception.type'] == 'ValueError'


@pytest.mark.anyio
async def test_agent_realtime_session_instruments(capfire: CaptureLogfire) -> None:
    """Agent.realtime_session() instruments when `instrument=True`."""
    agent: Agent[None, str] = Agent(instrument=True)

    conn = FakeRealtimeConnection([TurnComplete()])
    model = FakeRealtimeModel(conn)

    async with agent.realtime_session(model=model) as session:
        _ = [e async for e in session]

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    realtime_spans = [s for s in spans if 'realtime' in s['name']]
    assert len(realtime_spans) == 1
    assert realtime_spans[0]['attributes']['gen_ai.operation.name'] == 'realtime'


@pytest.mark.anyio
async def test_agent_instrument_all_enables_realtime(capfire: CaptureLogfire) -> None:
    """`Agent.instrument_all()` enables instrumentation for realtime sessions."""
    Agent.instrument_all()
    try:
        agent: Agent[None, str] = Agent()

        conn = FakeRealtimeConnection([TurnComplete()])
        model = FakeRealtimeModel(conn)

        async with agent.realtime_session(model=model) as session:
            _ = [e async for e in session]

        spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
        realtime_spans = [s for s in spans if 'realtime' in s['name']]
        assert len(realtime_spans) == 1
    finally:
        Agent.instrument_all(False)


@pytest.mark.anyio
async def test_agent_realtime_session_tools_in_span(capfire: CaptureLogfire) -> None:
    """Tool definitions from the agent appear in the instrumentation span."""
    agent: Agent[None, str] = Agent(instrument=True)

    @agent.tool_plain
    def greet(name: str) -> str:
        """Greet someone."""
        return f'Hello {name}!'

    tool_call = ToolCall(tool_call_id='tc_1', tool_name='greet', args='{"name": "Alice"}')
    conn = FakeRealtimeConnection([tool_call, TurnComplete()])
    model = FakeRealtimeModel(conn)

    async with agent.realtime_session(model=model) as session:
        events = [e async for e in session]

    assert isinstance(events[0], ToolCallStarted)
    assert isinstance(events[1], ToolCallCompleted)

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(s for s in spans if 'realtime' in s['name'])
    tool_defs = span['attributes']['gen_ai.tool.definitions']
    assert any(t['name'] == 'greet' for t in tool_defs)
