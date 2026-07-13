"""Tests for realtime OpenTelemetry instrumentation.

Two span sources meet here:

- The session-level `realtime` span and per-response `chat` spans are hand-managed by
  [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession]; tests that only exercise those construct
  a `RealtimeSession` directly with `instrumentation=`.
- The per-tool `execute_tool` span is owned by the `Instrumentation` capability's `wrap_tool_execute`
  hook, which [`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session] injects into the
  tool runner's `ToolManager` (mirroring a classic run). Tests that assert on tool spans go through
  `Agent.realtime_session` so the capability produces them.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.realtime import (
    AudioDelta,
    InputTranscript,
    RealtimeConnection,
    RealtimeEvent,
    RealtimeInput,
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeSession,
    SessionUsageEvent,
    ToolCall,
    Transcript,
    TurnCompleteEvent,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
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


class _Model(RealtimeModel):
    """A realtime model that yields a pre-built connection, for `Agent.realtime_session` tests."""

    def __init__(self, connection: RealtimeConnection) -> None:
        self._connection = connection

    @property
    def model_name(self) -> str:
        return 'gpt-realtime'

    @property
    def profile(self) -> RealtimeModelProfile:
        return RealtimeModelProfile(
            supports_image_input=True,
            supports_manual_turn_control=True,
            supports_interruption=True,
            supports_output_truncation=False,
            supports_session_seeding=True,
            supported_native_tools=frozenset(),
        )

    @asynccontextmanager
    async def connect(
        self,
        *,
        instructions: str,
        tools: list[ToolDefinition] | None = None,
        native_tools: list[AbstractNativeTool] | None = None,
        model_settings: ModelSettings | None = None,
        messages: Sequence[ModelMessage] | None = None,
    ) -> AsyncGenerator[RealtimeConnection]:
        yield self._connection


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


def _weather_agent(*, name: str | None = None, capabilities: list[Instrumentation] | None = None) -> Agent[None, str]:
    """An agent whose one tool mirrors the `_ok_runner` used by the direct-session tests."""
    agent: Agent[None, str] = Agent(name=name, capabilities=capabilities or [])

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return 'sunny'

    return agent


# --- tool spans: owned by the `Instrumentation` capability via `Agent.realtime_session` ----------


async def test_nested_agent_run_nests_under_session_span() -> None:
    settings, exporter = _settings()
    sub = Agent(TestModel())
    sub.instrument = settings

    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    async def analyze() -> str:
        result = await sub.run('hi')
        return str(result.output)

    agent.instrument = settings
    conn = _Connection([ToolCall(tool_call_id='c', tool_name='analyze', args='{}'), TurnCompleteEvent()])
    async with agent.realtime_session(model=_Model(conn)) as session:
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


async def test_session_and_tool_spans_with_usage() -> None:
    settings, exporter = _settings()
    agent = _weather_agent(name='assistant')
    agent.instrument = settings
    conn = _Connection(
        [
            ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}'),
            SessionUsageEvent(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            TurnCompleteEvent(),
        ]
    )
    async with agent.realtime_session(model=_Model(conn)) as session:
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
    # Arguments come from the capability's `wrap_tool_execute`, which serializes the validated dict
    # via `ToolCallPart.args_as_json_str()` — canonical compact JSON, not the raw provider string.
    assert tool.attributes['gen_ai.tool.call.arguments'] == '{"city":"Paris"}'
    assert tool.attributes['gen_ai.tool.call.result'] == 'sunny'
    # Both the `chat` span and the `execute_tool` span are children of the session span (siblings),
    # matching the classic agent-run tree where `execute_tool` follows `chat` rather than nesting in it.
    chat = spans['chat gpt-realtime']
    assert sess.context is not None
    assert chat.parent is not None and chat.parent.span_id == sess.context.span_id
    assert tool.parent is not None and tool.parent.span_id == sess.context.span_id


async def test_include_content_false_omits_args_and_result() -> None:
    settings, exporter = _settings(include_content=False)
    agent = _weather_agent()
    agent.instrument = settings
    conn = _Connection(
        [ToolCall(tool_call_id='c', tool_name='get_weather', args='{"city": "Paris"}'), TurnCompleteEvent()]
    )
    async with agent.realtime_session(model=_Model(conn)) as session:
        _ = [e async for e in session]
    tool = next(s for s in exporter.get_finished_spans() if s.name == 'execute_tool get_weather')
    assert tool.attributes is not None
    assert 'gen_ai.tool.call.arguments' not in tool.attributes
    assert 'gen_ai.tool.call.result' not in tool.attributes


async def test_chat_spans_split_on_tool_call_are_session_children() -> None:
    """A tool call splits a turn into two assistant responses → two `chat` spans; the tool runs between.

    Mirrors a classic run: the first `chat` span carries the assistant text plus the `ToolCallPart`,
    the capability's `execute_tool` span follows as a sibling under the session, and the second `chat`
    span carries the post-tool response. All three are children of the session span.
    """
    settings, exporter = _settings()
    agent = _weather_agent()
    agent.instrument = settings
    conn = _Connection(
        [
            InputTranscript(text='weather in Paris?', is_final=True),
            Transcript(text='let me check'),
            ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}'),
            Transcript(text='it is sunny'),
            SessionUsageEvent(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            TurnCompleteEvent(),
        ]
    )
    async with agent.realtime_session(model=_Model(conn)) as session:
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


# --- capability precedence: injected `instrument=` vs. explicit `Instrumentation` capability ------


async def test_instrument_and_explicit_capability_no_double_tool_spans() -> None:
    """`instrument=` plus an explicit `Instrumentation` capability must not double the tool span.

    The tool span has a single owner — the capability's `wrap_tool_execute` — so exactly one
    `execute_tool` span is produced even when both instrumentation entry points are configured.
    """
    settings, exporter = _settings()
    agent = _weather_agent(capabilities=[Instrumentation(settings=settings)])
    agent.instrument = settings
    conn = _Connection(
        [ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}'), TurnCompleteEvent()]
    )
    async with agent.realtime_session(model=_Model(conn)) as session:
        _ = [e async for e in session]
    tool_spans = [s for s in exporter.get_finished_spans() if s.name == 'execute_tool get_weather']
    assert len(tool_spans) == 1


async def test_explicit_capability_produces_session_chat_and_tool_spans() -> None:
    """An explicit `Instrumentation` capability alone (no `instrument=`) drives all realtime spans.

    Without the injection wiring, `instrument=` being unset would leave the session/`chat` spans
    absent while the capability still produced a tool span — inconsistent. The session span reads its
    settings from the explicit capability, so all three spans are emitted with those settings.
    """
    settings, exporter = _settings()
    agent = _weather_agent(name='assistant', capabilities=[Instrumentation(settings=settings)])
    conn = _Connection(
        [
            ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}'),
            SessionUsageEvent(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            TurnCompleteEvent(),
        ]
    )
    async with agent.realtime_session(model=_Model(conn)) as session:
        _ = [e async for e in session]
    spans = {s.name for s in exporter.get_finished_spans()}
    assert spans == {'realtime gpt-realtime', 'chat gpt-realtime', 'execute_tool get_weather'}


async def test_explicit_capability_settings_win_over_instrument() -> None:
    """Explicit capability settings take precedence over `instrument=`, mirroring classic runs.

    The two instrumentation entry points point at different exporters; when both are configured the
    explicit capability wins, so every realtime span (session, chat, tool) lands in the capability's
    exporter and none in the `instrument=` one.
    """
    cap_settings, cap_exporter = _settings()
    inst_settings, inst_exporter = _settings()
    agent = _weather_agent(name='assistant', capabilities=[Instrumentation(settings=cap_settings)])
    agent.instrument = inst_settings
    conn = _Connection(
        [
            ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}'),
            SessionUsageEvent(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            TurnCompleteEvent(),
        ]
    )
    async with agent.realtime_session(model=_Model(conn)) as session:
        _ = [e async for e in session]
    assert {s.name for s in cap_exporter.get_finished_spans()} == {
        'realtime gpt-realtime',
        'chat gpt-realtime',
        'execute_tool get_weather',
    }
    assert not inst_exporter.get_finished_spans()


# --- session + chat spans: hand-managed by `RealtimeSession` --------------------------------------


async def test_session_captures_transcript_messages() -> None:
    # The session span reuses the shared message → gen_ai serialization on the finalized history:
    # the user request lands as an input message, the assistant response as an output message.
    settings, exporter = _settings()
    conn = _Connection(
        [
            InputTranscript(text='hello there', is_final=True),
            Transcript(text='hi, how can I help?', is_final=True),
            TurnCompleteEvent(),
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
    conn = _Connection([InputTranscript(text='secret', is_final=True), TurnCompleteEvent()])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = [e async for e in session]
    sess = next(s for s in exporter.get_finished_spans() if s.name == 'realtime gpt-realtime')
    assert sess.attributes is not None
    assert 'gen_ai.input.messages' not in sess.attributes


async def test_session_span_sets_conversation_id() -> None:
    # `conversation_id` lands on the session span under the same key the classic agent-run span uses
    # (`gen_ai.conversation.id`), so a realtime session can be correlated with other runs.
    settings, exporter = _settings()
    agent = _weather_agent()
    agent.instrument = settings
    conn = _Connection([TurnCompleteEvent()])
    async with agent.realtime_session(model=_Model(conn), conversation_id='conv-123') as session:
        _ = [e async for e in session]
    sess = next(s for s in exporter.get_finished_spans() if s.name == 'realtime gpt-realtime')
    assert sess.attributes is not None
    assert sess.attributes['gen_ai.conversation.id'] == 'conv-123'


async def test_session_span_omits_conversation_id_when_unset() -> None:
    settings, exporter = _settings()
    conn = _Connection([TurnCompleteEvent()])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = [e async for e in session]
    sess = next(s for s in exporter.get_finished_spans() if s.name == 'realtime gpt-realtime')
    assert sess.attributes is not None
    assert 'gen_ai.conversation.id' not in sess.attributes


async def test_session_span_without_model_or_usage() -> None:
    settings, exporter = _settings()
    conn = _Connection([TurnCompleteEvent()])  # no model/agent name, no Usage event
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
    conn = _Connection([AudioDelta(data=b'\x00\x01'), TurnCompleteEvent()])
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
            SessionUsageEvent(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            TurnCompleteEvent(),
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
            SessionUsageEvent(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            TurnCompleteEvent(),
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
            TurnCompleteEvent(),
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


async def test_direct_session_runs_tool_via_runner() -> None:
    """A direct `RealtimeSession` executes a tool call through its `tool_runner` and folds the result in.

    The hand-managed path has no `Instrumentation` capability, so no `execute_tool` span is produced;
    the runner's result surfaces as the `tool_call_response` folded into the post-tool `chat` span.
    """
    settings, exporter = _settings()
    conn = _Connection(
        [
            InputTranscript(text='weather in Paris?', is_final=True),
            Transcript(text='let me check'),
            ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}'),
            Transcript(text='it is sunny'),
            TurnCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = [e async for e in session]

    finished = exporter.get_finished_spans()
    assert not [s for s in finished if s.name.startswith('execute_tool')]  # tool spans are capability-owned
    chats = [s for s in finished if s.name == 'chat gpt-realtime']
    assert len(chats) == 2
    _, second = chats
    assert second.attributes is not None
    # The runner returned 'sunny', which is folded into the post-tool response's input messages.
    assert {'type': 'tool_call_response', 'id': 'c1', 'name': 'get_weather', 'result': 'sunny'} in json.loads(
        str(second.attributes['gen_ai.input.messages'])
    )[-1]['parts']


async def test_chat_span_without_model_name() -> None:
    """Without a model name, the `chat` span is named just `chat` and omits `gen_ai.request.model`."""
    settings, exporter = _settings()
    conn = _Connection([Transcript(text='hello'), TurnCompleteEvent()])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings)  # no model_name
    _ = [e async for e in session]
    chat = next(s for s in exporter.get_finished_spans() if s.name == 'chat')
    assert chat.attributes is not None
    assert 'gen_ai.request.model' not in chat.attributes


async def test_early_break_finishes_chat_span() -> None:
    """Breaking mid-turn still finishes the in-flight `chat` span, so it doesn't outlive the session span.

    A response opens a `chat` span on its first content but only finishes it on `TurnCompleteEvent` /
    a tool-call boundary. If the consumer breaks or cancels mid-turn, the span used to leak unfinished.
    """
    settings, exporter = _settings()
    conn = _Connection([AudioDelta(data=b'\x00'), AudioDelta(data=b'\x01')])  # no TurnCompleteEvent
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    agen = cast(AsyncGenerator[Any], session.__aiter__())
    await agen.__anext__()  # first audio delta opens the assistant `chat` span (still in-flight)
    await agen.aclose()  # break before the turn completes
    # Every started span must be finished — including the `chat` child, which used to be left open.
    assert {s.name for s in exporter.get_finished_spans()} == {'realtime gpt-realtime', 'chat gpt-realtime'}
