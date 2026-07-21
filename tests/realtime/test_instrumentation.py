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

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import pytest
from inline_snapshot import snapshot

pytest.importorskip('opentelemetry.sdk')  # only installed via the optional `logfire` extra

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.realtime import (
    AudioDelta,
    InputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeEvent,
    RealtimeInput,
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
    RealtimeSession as _RealtimeSession,
    SessionUsageEvent,
    ToolCall,
    Transcript,
    TurnCompleteEvent,
)
from pydantic_ai.usage import RequestUsage

from .test_session import make_tool_manager

pytestmark = pytest.mark.anyio


def RealtimeSession(connection: RealtimeConnection, runner: Any, **kwargs: Any) -> _RealtimeSession:
    return _RealtimeSession(connection, make_tool_manager(runner), **kwargs)


async def collect_events(session: _RealtimeSession) -> list[RealtimeEvent]:
    async with session:
        return [event async for event in session]


def _span_tree(exporter: InMemorySpanExporter) -> list[dict[str, Any]]:
    """Render the finished spans as nested `{name: [children]}` dicts, ordered by start time.

    A readable view of the whole session's span tree, so a test can pin the parent/child shape
    (session span parenting its `chat` and `execute_tool` spans) in one assertion.
    """
    spans = exporter.get_finished_spans()
    by_id = {span.context.span_id: span for span in spans if span.context is not None}
    children: dict[int | None, list[Any]] = {}
    for span in spans:
        parent_id = span.parent.span_id if span.parent is not None and span.parent.span_id in by_id else None
        children.setdefault(parent_id, []).append(span)

    def render(span: Any) -> dict[str, Any]:
        kids = sorted(children.get(span.context.span_id, []), key=lambda child: child.start_time)
        return {span.name: [render(child) for child in kids]}

    return [render(root) for root in sorted(children.get(None, []), key=lambda span: span.start_time)]


class _Connection(RealtimeConnection):
    """Replays a fixed list of events; records nothing of interest for these tests."""

    def __init__(self, events: list[RealtimeCodecEvent]) -> None:
        self._events = events

    async def send(self, content: RealtimeInput) -> None:
        pass

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
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
    def system(self) -> str:
        return 'openai'

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
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[RealtimeConnection]:
        yield self._connection


def _settings(
    *, include_content: bool = True, use_aggregated_usage_attribute_names: bool = True
) -> tuple[InstrumentationSettings, InMemorySpanExporter]:
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
    # The semconv operation-name enum has no realtime value, so the session span reports the
    # session as an agent invocation like the classic agent-run span.
    assert sess.attributes['gen_ai.operation.name'] == 'invoke_agent'
    assert sess.attributes['gen_ai.request.model'] == 'gpt-realtime'
    assert sess.attributes['gen_ai.agent.name'] == 'assistant'
    # `gen_ai.output.type` reports the configured output modality; the default is spoken audio,
    # which the semconv enum calls `speech`. Set on both the session span and the `chat` spans.
    assert sess.attributes['gen_ai.output.type'] == 'speech'
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
    assert chat.attributes is not None
    assert chat.attributes['gen_ai.output.type'] == 'speech'
    assert sess.context is not None
    assert chat.parent is not None and chat.parent.span_id == sess.context.span_id
    assert tool.parent is not None and tool.parent.span_id == sess.context.span_id


async def test_output_type_reflects_text_modality() -> None:
    # With `output_modality='text'` the model replies as plain text rather than speech, and the
    # session and `chat` spans report `gen_ai.output.type='text'` (threaded from the model settings
    # by `Agent.realtime_session`).
    settings, exporter = _settings()
    agent = _weather_agent(name='assistant')
    agent.instrument = settings
    conn = _Connection([Transcript(text='hi', is_final=True), TurnCompleteEvent()])
    async with agent.realtime_session(
        model=_Model(conn), model_settings=RealtimeModelSettings(output_modality='text')
    ) as session:
        _ = [e async for e in session]
    spans = {s.name: s for s in exporter.get_finished_spans()}
    for name in ('realtime gpt-realtime', 'chat gpt-realtime'):
        attributes = spans[name].attributes
        assert attributes is not None
        assert attributes['gen_ai.output.type'] == 'text'


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
    # The synthetic connection emits the post-tool response without yielding, so the concurrent tool
    # has not finished when the second `chat` span opens. Its input therefore ends at the tool call;
    # the late result is still inserted next to the call in session history after it completes.
    assert json.loads(str(second.attributes['gen_ai.input.messages']))[-1]['parts'] == [
        {'type': 'text', 'content': 'let me check'},
        {'type': 'tool_call', 'id': 'c1', 'name': 'get_weather', 'arguments': '{"city": "Paris"}'},
    ]
    assert json.loads(str(second.attributes['gen_ai.output.messages'])) == [
        {
            'role': 'assistant',
            'parts': [{'type': 'text', 'content': 'it is sunny'}],
            'finish_reason': 'stop',
        },
    ]


async def test_conversation_span_tree() -> None:
    """The whole span tree for a realistic session, in one view.

    A first user turn where the assistant speaks, calls a tool, then answers (two `chat` spans around
    one `execute_tool` span), followed by a second spoken turn (a third `chat` span). Every `chat` and
    `execute_tool` span is a direct child of the single `realtime` session span.
    """
    settings, exporter = _settings()
    agent = _weather_agent(name='assistant')
    agent.instrument = settings
    conn = _Connection(
        [
            InputTranscript(text='weather in Paris?', is_final=True),
            Transcript(text='let me check'),
            ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}'),
            Transcript(text='it is sunny'),
            SessionUsageEvent(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            TurnCompleteEvent(),
            InputTranscript(text='and tomorrow?', is_final=True),
            Transcript(text='also sunny'),
            TurnCompleteEvent(),
        ]
    )
    async with agent.realtime_session(model=_Model(conn)) as session:
        _ = [e async for e in session]

    # Two turns → three `chat` spans (the first turn splits around the tool call) plus one
    # `execute_tool` span, all direct children of the one session span. Children are ordered by start
    # time: the tool span starts last here because the synthetic connection emits every event without
    # yielding, so the concurrent tool only runs after the pump has drained and opened all `chat` spans.
    assert _span_tree(exporter) == snapshot(
        [
            {
                'realtime gpt-realtime': [
                    {'chat gpt-realtime': []},
                    {'chat gpt-realtime': []},
                    {'chat gpt-realtime': []},
                    {'execute_tool get_weather': []},
                ]
            }
        ]
    )


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
    # The session span mirrors the classic agent-run span's end-of-run contract: the full
    # conversation, in order, under `pydantic_ai.all_messages`, with `logfire.json_schema` marking
    # it as a JSON array so the Logfire UI renders it as a conversation.
    settings, exporter = _settings()
    conn = _Connection(
        [
            InputTranscript(text='hello there', is_final=True),
            Transcript(text='hi, how can I help?', is_final=True),
            TurnCompleteEvent(),
        ]
    )
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = await collect_events(session)

    sess = next(s for s in exporter.get_finished_spans() if s.name == 'realtime gpt-realtime')
    assert sess.attributes is not None
    assert json.loads(str(sess.attributes['pydantic_ai.all_messages'])) == [
        {'role': 'user', 'parts': [{'type': 'text', 'content': 'hello there'}]},
        {
            'role': 'assistant',
            'parts': [{'type': 'text', 'content': 'hi, how can I help?'}],
            'finish_reason': 'stop',
        },
    ]
    assert json.loads(str(sess.attributes['logfire.json_schema']))['properties'] == {
        'pydantic_ai.all_messages': {'type': 'array'},
    }
    # No seeded history, so there is no prior-messages boundary to mark.
    assert 'pydantic_ai.new_message_index' not in sess.attributes


async def test_session_span_includes_resolved_run_attributes() -> None:
    settings, exporter = _settings()
    agent: Agent[None, str] = Agent(
        name='assistant',
        description='Handles realtime conversations.',
        instructions='Keep answers concise.',
    )
    agent.instrument = settings
    conn = _Connection([Transcript(text='hello', is_final=True), TurnCompleteEvent()])

    async with agent.realtime_session(model=_Model(conn), metadata={'tier': 'gold'}) as session:
        _ = [event async for event in session]

    sess = next(s for s in exporter.get_finished_spans() if s.name == 'realtime gpt-realtime')
    assert sess.attributes is not None
    assert sess.attributes['gen_ai.agent.description'] == 'Handles realtime conversations.'
    assert json.loads(str(sess.attributes['gen_ai.system_instructions'])) == [
        {'type': 'text', 'content': 'Keep answers concise.'}
    ]
    assert json.loads(str(sess.attributes['metadata'])) == {'tier': 'gold'}
    assert json.loads(str(sess.attributes['logfire.json_schema']))['properties'] == {
        'gen_ai.system_instructions': {'type': 'array'},
        'pydantic_ai.all_messages': {'type': 'array'},
        'metadata': {},
    }


async def test_session_span_marks_seeded_history_boundary() -> None:
    # A session seeded with `message_history=` includes the seeded messages in
    # `pydantic_ai.all_messages` and marks where this session's own messages begin with
    # `pydantic_ai.new_message_index`, exactly like a classic run given `message_history=`.
    settings, exporter = _settings()
    conn = _Connection([InputTranscript(text='and now?', is_final=True), TurnCompleteEvent()])
    seeded: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='earlier question')]),
        ModelResponse(parts=[TextPart(content='earlier answer')]),
    ]
    session = RealtimeSession(
        conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime', message_history=seeded
    )
    _ = await collect_events(session)

    sess = next(s for s in exporter.get_finished_spans() if s.name == 'realtime gpt-realtime')
    assert sess.attributes is not None
    assert sess.attributes['pydantic_ai.new_message_index'] == 2
    all_messages = json.loads(str(sess.attributes['pydantic_ai.all_messages']))
    assert [m['role'] for m in all_messages] == ['user', 'assistant', 'user']
    assert all_messages[2]['parts'] == [{'type': 'text', 'content': 'and now?'}]


async def test_include_content_false_redacts_transcript_messages() -> None:
    # With `include_content=False` the conversation *structure* is still emitted (matching the
    # classic agent-run span); per-part content is redacted by `otel_message_parts`.
    settings, exporter = _settings(include_content=False)
    conn = _Connection([InputTranscript(text='secret', is_final=True), TurnCompleteEvent()])
    session = RealtimeSession(
        conn,
        _ok_runner,
        instrumentation=settings,
        model_name='gpt-realtime',
        instructions='secret instructions',
        metadata={'tier': 'gold'},
    )
    _ = await collect_events(session)
    sess = next(s for s in exporter.get_finished_spans() if s.name == 'realtime gpt-realtime')
    assert sess.attributes is not None
    assert json.loads(str(sess.attributes['pydantic_ai.all_messages'])) == [
        {'role': 'user', 'parts': [{'type': 'text'}]},
    ]
    assert 'secret' not in str(sess.attributes['pydantic_ai.all_messages'])
    assert 'gen_ai.system_instructions' not in sess.attributes
    assert json.loads(str(sess.attributes['metadata'])) == {'tier': 'gold'}


async def test_session_span_sets_conversation_id() -> None:
    # `conversation_id` lands on the session span under the same key the classic agent-run span uses
    # (`gen_ai.conversation.id`), so a realtime session can be correlated with other runs.
    settings, exporter = _settings()
    agent = _weather_agent()
    agent.instrument = settings
    conn = _Connection([TurnCompleteEvent()])
    async with agent.realtime_session(model=_Model(conn), conversation_id='conv-123') as session:
        _ = [event async for event in session]
    sess = next(s for s in exporter.get_finished_spans() if s.name == 'realtime gpt-realtime')
    assert sess.attributes is not None
    assert sess.attributes['gen_ai.conversation.id'] == 'conv-123'


async def test_session_span_omits_conversation_id_when_unset() -> None:
    settings, exporter = _settings()
    conn = _Connection([TurnCompleteEvent()])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = await collect_events(session)
    sess = next(s for s in exporter.get_finished_spans() if s.name == 'realtime gpt-realtime')
    assert sess.attributes is not None
    assert 'gen_ai.conversation.id' not in sess.attributes


async def test_session_span_without_model_or_usage() -> None:
    settings, exporter = _settings()
    conn = _Connection([TurnCompleteEvent()])  # no model/agent name, no Usage event
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings)
    _ = await collect_events(session)
    sess = next(s for s in exporter.get_finished_spans() if s.name == 'realtime')
    assert sess.attributes is not None
    assert sess.attributes['gen_ai.operation.name'] == 'invoke_agent'
    assert 'gen_ai.request.model' not in sess.attributes
    assert 'gen_ai.agent.name' not in sess.attributes
    assert 'gen_ai.agent.description' not in sess.attributes
    assert 'gen_ai.system_instructions' not in sess.attributes
    assert 'metadata' not in sess.attributes
    assert 'gen_ai.usage.input_tokens' not in sess.attributes  # zero usage → no token attribute / metric
    # An empty turn produces no assistant `ModelResponse`, so no `chat` span is opened.
    assert not [s for s in exporter.get_finished_spans() if s.name.startswith('chat')]


async def test_chat_span_closed_for_contentless_response() -> None:
    # Audio with no transcript opens a `chat` span (first content) but finalizes with no response
    # parts, so the span closes without attaching messages.
    settings, exporter = _settings()
    conn = _Connection([AudioDelta(data=b'\x00\x01'), TurnCompleteEvent()])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = await collect_events(session)
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
    _ = await collect_events(session)
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
            SessionUsageEvent(
                usage=RequestUsage(input_tokens=10, output_tokens=4),
                provider_response_id='resp-1',
                finish_reason='stop',
            ),
            TurnCompleteEvent(provider_response_id='resp-1', finish_reason='stop'),
        ]
    )
    session = RealtimeSession(
        conn,
        _ok_runner,
        instrumentation=settings,
        model_name='gpt-realtime',
        provider_name='openai',
    )
    _ = await collect_events(session)

    chat = next(s for s in exporter.get_finished_spans() if s.name == 'chat gpt-realtime')
    assert chat.attributes is not None
    assert chat.attributes['gen_ai.operation.name'] == 'chat'
    assert chat.attributes['gen_ai.request.model'] == 'gpt-realtime'
    assert chat.attributes['gen_ai.response.model'] == 'gpt-realtime'
    assert chat.attributes['gen_ai.provider.name'] == 'openai'
    assert chat.attributes['gen_ai.system'] == 'openai'
    assert chat.attributes['gen_ai.response.id'] == 'resp-1'
    assert chat.attributes['gen_ai.response.finish_reasons'] == ('stop',)
    # Per-response usage under the standard (non-aggregated) namespace, exactly as the classic path.
    assert chat.attributes['gen_ai.usage.input_tokens'] == 10
    assert chat.attributes['gen_ai.usage.output_tokens'] == 4
    # Input = the history slice the response replied to; output = the finalized assistant response.
    assert json.loads(str(chat.attributes['gen_ai.input.messages'])) == [
        {'role': 'user', 'parts': [{'type': 'text', 'content': 'hello there'}]},
    ]
    assert json.loads(str(chat.attributes['gen_ai.output.messages'])) == [
        {
            'role': 'assistant',
            'parts': [{'type': 'text', 'content': 'hi, how can I help?'}],
            'finish_reason': 'stop',
        },
    ]
    assert 'gen_ai.input.messages' in json.loads(str(chat.attributes['logfire.json_schema']))['properties']
    # Honest omissions vs. the classic `chat` span: no server address, request parameters/settings,
    # or cost (the session has no provider base URL and does not calculate per-response cost).
    for omitted in (
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
    _ = await collect_events(session)
    chat = next(s for s in exporter.get_finished_spans() if s.name == 'chat gpt-realtime')
    assert chat.attributes is not None
    # Envelope present, content redacted (no `content` key on the text parts).
    assert json.loads(str(chat.attributes['gen_ai.input.messages'])) == [
        {'role': 'user', 'parts': [{'type': 'text'}]},
    ]
    assert json.loads(str(chat.attributes['gen_ai.output.messages'])) == [
        {'role': 'assistant', 'parts': [{'type': 'text'}], 'finish_reason': 'stop'},
    ]
    assert chat.attributes['gen_ai.response.model'] == 'gpt-realtime'


async def test_direct_session_runs_tool_via_runner() -> None:
    """A direct `RealtimeSession` executes a tool call concurrently through its `tool_runner`.

    The hand-managed path has no `Instrumentation` capability, so no `execute_tool` span is produced;
    the runner's result is inserted into history when it completes.
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
    _ = await collect_events(session)

    # The runner actually ran and its result was inserted into history as a `ToolReturnPart` — the
    # point of the direct-session tool-runner path, independent of the span assertions below.
    tool_returns = [
        part
        for message in session.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert [(part.tool_name, part.content, part.tool_call_id) for part in tool_returns] == [
        ('get_weather', 'sunny', 'c1')
    ]

    finished = exporter.get_finished_spans()
    assert not [s for s in finished if s.name.startswith('execute_tool')]  # tool spans are capability-owned
    chats = [s for s in finished if s.name == 'chat gpt-realtime']
    assert len(chats) == 2
    _, second = chats
    assert second.attributes is not None
    # The connection does not yield between the call and response, so the concurrent tool finishes
    # after this span opens and is not yet present in its input attributes.
    assert json.loads(str(second.attributes['gen_ai.input.messages']))[-1]['parts'] == [
        {'type': 'text', 'content': 'let me check'},
        {'type': 'tool_call', 'id': 'c1', 'name': 'get_weather', 'arguments': '{"city": "Paris"}'},
    ]


async def test_chat_span_without_model_name() -> None:
    """Without a model name, the `chat` span is named just `chat` and omits `gen_ai.request.model`."""
    settings, exporter = _settings()
    conn = _Connection([Transcript(text='hello'), TurnCompleteEvent()])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings)  # no model_name
    _ = await collect_events(session)
    chat = next(s for s in exporter.get_finished_spans() if s.name == 'chat')
    assert chat.attributes is not None
    assert 'gen_ai.request.model' not in chat.attributes


async def test_early_break_finishes_chat_span(caplog: pytest.LogCaptureFixture) -> None:
    """The documented early-break shape synchronously finishes spans in the owner's OTel context."""
    settings, exporter = _settings()
    agent = _weather_agent(capabilities=[Instrumentation(settings=settings)])
    conn = _Connection(
        [AudioDelta(data=b'\x00'), AudioDelta(data=b'\x01'), AudioDelta(data=b'\x02')]
    )  # no TurnCompleteEvent

    with caplog.at_level(logging.ERROR, logger='opentelemetry'):
        async with agent.realtime_session(model=_Model(conn)) as session:
            async for _ in session:
                break

    # These are all spans this path starts. They must be exported before the owner block returns,
    # without GC or extra event-loop turns, and the session remains the explicit parent of `chat`.
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {'realtime gpt-realtime', 'chat gpt-realtime'}
    session_span = spans['realtime gpt-realtime']
    chat_span = spans['chat gpt-realtime']
    assert session_span.context is not None
    assert chat_span.parent is not None and chat_span.parent.span_id == session_span.context.span_id
    assert not any(
        'Failed to detach context' in record.getMessage() or 'different Context' in record.getMessage()
        for record in caplog.records
    )


async def test_early_break_finishes_running_tool_span(caplog: pytest.LogCaptureFixture) -> None:
    """Owner exit cancels a running tool and finishes every span before returning."""

    class _IdleAfterTool(RealtimeConnection):
        async def send(self, content: RealtimeInput) -> None:  # pragma: no cover - tool is cancelled first
            raise AssertionError

        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            yield ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}')
            await asyncio.Event().wait()

    settings, exporter = _settings()
    agent: Agent[None, str] = Agent(capabilities=[Instrumentation(settings=settings)])
    blocked = asyncio.Event()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    tool_task: asyncio.Task[Any] | None = None

    @agent.tool_plain
    async def get_weather(city: str) -> str:
        nonlocal tool_task
        tool_task = asyncio.current_task()
        started.set()
        try:
            await blocked.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return f'sunny in {city}'  # pragma: no cover

    with caplog.at_level(logging.ERROR, logger='opentelemetry'):
        async with agent.realtime_session(model=_Model(_IdleAfterTool())) as session:
            async for _ in session:
                await started.wait()
                break

    assert tool_task is not None and tool_task.done() and tool_task.cancelled()
    assert cancelled.is_set()
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {'realtime gpt-realtime', 'chat gpt-realtime', 'execute_tool get_weather'}
    session_span = spans['realtime gpt-realtime']
    assert session_span.context is not None
    for child_name in ('chat gpt-realtime', 'execute_tool get_weather'):
        parent = spans[child_name].parent
        assert parent is not None and parent.span_id == session_span.context.span_id
    assert not any(
        'Failed to detach context' in record.getMessage() or 'different Context' in record.getMessage()
        for record in caplog.records
    )
