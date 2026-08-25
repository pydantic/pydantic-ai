"""Tests for realtime OpenTelemetry instrumentation.

Two span sources meet here:

- The session-level `realtime` span and per-response `chat` spans are hand-managed by
  [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession]; tests that only exercise those construct
  a `RealtimeSession` directly with `instrumentation=`.
- The per-tool `execute_tool` span is owned by the `Instrumentation` capability's `wrap_tool_execute`
  hook, which [`Agent.realtime`][pydantic_ai.agent.Agent.realtime] injects into the
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

from opentelemetry.context import Context
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import Histogram, InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

from pydantic_ai import Agent
from pydantic_ai._instrumentation import provider_attributes
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SpeechPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.realtime import (
    RealtimeEvent,
    RealtimeInputSpeechEndEvent,
    RealtimeInputSpeechStartEvent,
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
    RealtimeOutputSpeechEndEvent,
    RealtimeOutputSpeechStartEvent,
    RealtimeSession as _RealtimeSession,
    RealtimeSessionReconnectEvent,
)
from pydantic_ai.realtime.codec import (
    AudioDelta,
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    ResponseDone,
    SessionUsage,
    ToolCall,
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from .test_session import FakeRealtimeModel, make_tool_manager

pytestmark = pytest.mark.anyio


def RealtimeSession(connection: RealtimeConnection, runner: Any, **kwargs: Any) -> _RealtimeSession:
    if any(name in kwargs for name in ('model_name', 'provider_name', 'provider_url')):
        kwargs['model'] = FakeRealtimeModel(
            connection,
            model_name=kwargs.pop('model_name', None),
            system=kwargs.pop('provider_name', None),
            base_url=kwargs.pop('provider_url', None),
        )
    return _RealtimeSession(connection, tool_manager=make_tool_manager(runner), **kwargs)


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
    def base_url(self) -> str:
        return 'https://api.openai.com/v1'

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
    *,
    include_content: bool = True,
    include_binary_content: bool = True,
    use_aggregated_usage_attribute_names: bool = True,
) -> tuple[InstrumentationSettings, InMemorySpanExporter]:
    settings, exporter, _ = _settings_with_metrics(
        include_content=include_content,
        include_binary_content=include_binary_content,
        use_aggregated_usage_attribute_names=use_aggregated_usage_attribute_names,
    )
    return settings, exporter


def _settings_with_metrics(
    *,
    include_content: bool = True,
    include_binary_content: bool = True,
    use_aggregated_usage_attribute_names: bool = True,
) -> tuple[InstrumentationSettings, InMemorySpanExporter, InMemoryMetricReader]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    metric_reader = InMemoryMetricReader()
    settings = InstrumentationSettings(
        tracer_provider=provider,
        meter_provider=MeterProvider(metric_readers=[metric_reader]),
        include_content=include_content,
        include_binary_content=include_binary_content,
        use_aggregated_usage_attribute_names=use_aggregated_usage_attribute_names,
    )
    return settings, exporter, metric_reader


async def _ok_runner(name: str, args: dict[str, Any], call_id: str) -> str:
    return 'sunny'


async def test_owner_error_marks_active_chat_and_session_spans() -> None:
    settings, exporter = _settings()
    session = RealtimeSession(
        _Connection([OutputTranscript(text='partial')]),
        _ok_runner,
        instrumentation=settings,
        model_name='gpt-realtime',
    )

    with pytest.raises(RuntimeError, match='application failed'):
        async with session:
            stream = session.__aiter__()
            await anext(stream)
            raise RuntimeError('application failed')

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans['chat gpt-realtime'].status.is_ok is False
    assert spans['invoke_agent agent'].status.is_ok is False
    assert all(span.events for span in spans.values())


async def test_playback_boundary_opens_a_speak_span_outlasting_the_response() -> None:
    """A `speak` span measures audibility, which a WebRTC sideband's turn spans can't.

    The provider generates audio far ahead of playing it, so `model turn complete` lands while the listener
    still has seconds of speech to hear. Without this span the trace claims the model stopped talking
    long before it did.
    """
    settings, exporter = _settings()
    session = RealtimeSession(
        _Connection(
            [
                RealtimeOutputSpeechStartEvent(),
                # A repeat, which some providers send: one utterance is one span, not two.
                RealtimeOutputSpeechStartEvent(),
                ResponseDone(),
                RealtimeOutputSpeechEndEvent(),
            ]
        ),
        _ok_runner,
        instrumentation=settings,
        model_name='gpt-realtime',
    )

    async with session:
        events = [event async for event in session]

    # The events reach the caller as well, so a UI can drive a 'speaking' indicator from them.
    assert [type(event).__name__ for event in events] == [
        'RealtimeOutputSpeechStartEvent',
        'RealtimeOutputSpeechStartEvent',
        'RealtimeTurnCompleteEvent',
        'RealtimeOutputSpeechEndEvent',
    ]
    assert len([span for span in exporter.get_finished_spans() if span.name == 'speak gpt-realtime']) == 1
    spans = {span.name: span for span in exporter.get_finished_spans()}
    speak = spans['speak gpt-realtime']
    assert speak.end_time is not None and speak.start_time is not None
    # It brackets the turn rather than nesting inside it: playback outlives the response.
    assert speak.end_time >= spans['model turn complete'].end_time  # type: ignore[operator]


async def test_no_speak_span_without_playback_reporting() -> None:
    """An ordinary session never reports playback, so its trace is unchanged."""
    settings, exporter = _settings()
    session = RealtimeSession(
        _Connection([OutputTranscript(text='hi', is_final=True), ResponseDone()]),
        _ok_runner,
        instrumentation=settings,
        model_name='gpt-realtime',
    )

    async with session:
        _ = [event async for event in session]

    assert not [span for span in exporter.get_finished_spans() if span.name.startswith('speak')]


async def test_reconnect_without_state_restored_ends_the_speak_span() -> None:
    """A reconnect that didn't restore state closes the open `speak` span, so it doesn't swallow the next utterance.

    The audio that was playing won't resume on the fresh connection. If the span stays open, the next
    utterance's `RealtimeOutputSpeechStartEvent` no-ops (a span is already set) and merges into it, so
    the two utterances land in one `speak` span that also spans the dead-air reconnect. Ending it at
    the reconnect keeps one span per utterance. Regression for the sixth `#7392` finding.
    """
    settings, exporter = _settings()
    session = RealtimeSession(
        _Connection(
            [
                RealtimeOutputSpeechStartEvent(),
                # The default connection restores in-flight state, but a `state_restored=False` reconnect
                # is still a disruption — the playing audio is cut regardless.
                RealtimeSessionReconnectEvent(state_restored=False),
                RealtimeOutputSpeechStartEvent(),
                ResponseDone(),
                RealtimeOutputSpeechEndEvent(),
            ]
        ),
        _ok_runner,
        instrumentation=settings,
        model_name='gpt-realtime',
    )

    async with session:
        _ = [event async for event in session]

    # Two distinct utterances bracketed a reconnect, so two `speak` spans -- not one merged across it.
    assert len([span for span in exporter.get_finished_spans() if span.name == 'speak gpt-realtime']) == 2


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
    conn = _Connection([ToolCall(tool_call_id='c', tool_name='analyze', args='{}'), ResponseDone()])
    async with agent.realtime(_Model(conn)).session() as session:
        _ = [e async for e in session]

    by_id = {s.context.span_id: s for s in exporter.get_finished_spans() if s.context is not None}
    session_span = next(s for s in by_id.values() if s.name == 'invoke_agent agent')
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
            SessionUsage(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            ResponseDone(),
        ]
    )
    async with agent.realtime(_Model(conn), run_id='session-run').session() as session:
        _ = [e async for e in session]

    spans = {s.name: s for s in exporter.get_finished_spans()}
    # No `model turn complete` span: the tool round is all this connection yields, so the model's follow-up
    # response — where the exchange would end — never arrives.
    assert set(spans) == {'invoke_agent assistant', 'chat gpt-realtime', 'execute_tool get_weather'}

    sess = spans['invoke_agent assistant']
    assert sess.attributes is not None
    # The semconv operation-name enum has no realtime value, so the session span reports the
    # session as an agent invocation like the classic agent-run span.
    assert sess.attributes['gen_ai.operation.name'] == 'invoke_agent'
    # The session span reports the model under `model_name` like the classic agent-run span (not
    # `gen_ai.request.model`, which stays on the child `chat`/turn spans).
    assert sess.attributes['model_name'] == 'gpt-realtime'
    assert 'gen_ai.request.model' not in sess.attributes
    assert sess.attributes['gen_ai.agent.name'] == 'assistant'
    # The session is one run, so it reports its `run_id` under the same key the classic agent-run span
    # uses — letting a session and the runs around it be correlated the same way.
    assert sess.attributes['gen_ai.agent.call.id'] == 'session-run'
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
    # Every span in a session tree carries the realtime marker — the tool span is created by the
    # shared `Instrumentation` capability, which marks it only when the active model is realtime.
    assert all(s.attributes is not None and s.attributes.get('pydantic_ai.realtime') is True for s in spans.values())
    # The capability receives the raw call, matching the standard tool-manager path.
    assert tool.attributes['gen_ai.tool.call.arguments'] == '{"city": "Paris"}'
    assert tool.attributes['gen_ai.tool.call.result'] == 'sunny'
    # Both the `chat` span and the `execute_tool` span are children of the session span (siblings),
    # matching the classic agent-run tree where `execute_tool` follows `chat` rather than nesting in it.
    chat = spans['chat gpt-realtime']
    assert chat.attributes is not None
    assert chat.attributes['gen_ai.output.type'] == 'speech'
    assert chat.attributes['gen_ai.provider.name'] == 'openai'
    assert chat.attributes['server.address'] == 'api.openai.com'
    assert sess.context is not None
    assert chat.parent is not None and chat.parent.span_id == sess.context.span_id
    assert tool.parent is not None and tool.parent.span_id == sess.context.span_id


async def test_session_and_chat_spans_carry_request_config() -> None:
    # `model_request_parameters` (native tools included) and `model_settings` are sent once at connect, so
    # they're set on the session span and duplicated onto each per-turn `chat` span — matching where the
    # classic path carries them, so Logfire renders native tools / tool definitions per step.
    settings, exporter = _settings()
    conn = _Connection(
        [
            ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}'),
            ResponseDone(),
        ]
    )
    session = RealtimeSession(
        conn,
        _ok_runner,
        instrumentation=settings,
        model_name='gpt-realtime',
        provider_name='openai',
        provider_url='https://api.openai.com/v1',
        model_request_parameters=ModelRequestParameters(
            function_tools=[ToolDefinition(name='get_weather')],
            native_tools=[WebSearchTool()],
        ),
        model_settings=RealtimeModelSettings(max_tokens=4096),
    )
    async with session:
        _ = [e async for e in session]

    spans = {s.name: s for s in exporter.get_finished_spans()}
    # The config is identical on the session span and every per-turn `chat` span.
    for name in ('invoke_agent agent', 'chat gpt-realtime'):
        attributes = spans[name].attributes
        assert attributes is not None
        # `model_request_parameters` is serialized whole, so the session's configured native tools are
        # inspectable — the reason for surfacing it (e.g. to see which native tools the API was given).
        params = json.loads(str(attributes['model_request_parameters']))
        assert [t['kind'] for t in params['native_tools']] == ['web_search']
        assert [t['name'] for t in params['function_tools']] == ['get_weather']
        # The realtime settings vocabulary is serialized as-is, including settings with no OTel-spec home.
        assert json.loads(str(attributes['model_settings'])) == {'max_tokens': 4096}
        # Function/output tools are also emitted as `gen_ai.tool.definitions`, like the classic `chat` span.
        assert json.loads(str(attributes['gen_ai.tool.definitions'])) == [
            {'type': 'function', 'name': 'get_weather', 'parameters': {'type': 'object', 'properties': {}}}
        ]
        # `max_tokens` is the one realtime setting with an OTel-spec `gen_ai.request.*` home.
        assert attributes['gen_ai.request.max_tokens'] == 4096
        # `model_request_parameters` is declared an object so Logfire renders it richly (not a raw string).
        assert json.loads(str(attributes['logfire.json_schema']))['properties']['model_request_parameters'] == {
            'type': 'object'
        }

    # The `chat` span keeps the semconv `chat` operation and `chat {model}` span name, but renders (via
    # `logfire.msg`) as `response {model}`: it covers one `ModelResponse`, and no request was sent. It is
    # deliberately not called a "turn" — a turn that calls tools produces several of these spans, and the
    # turn boundary is the separate `model turn complete` span.
    chat_attributes = spans['chat gpt-realtime'].attributes
    assert chat_attributes is not None
    assert chat_attributes['gen_ai.operation.name'] == 'chat'
    assert chat_attributes['logfire.msg'] == 'response gpt-realtime'


async def test_tool_call_otel_metadata_comes_from_definition() -> None:
    settings, exporter = _settings()
    conn = _Connection([ToolCall(tool_call_id='c1', tool_name='run_code', args='{"python_code":"1 + 1"}')])
    session = RealtimeSession(
        conn,
        _ok_runner,
        instrumentation=settings,
        model_request_parameters=ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name='run_code',
                    metadata={'code_arg_name': 'python_code', 'code_arg_language': 'python'},
                )
            ]
        ),
    )

    async with session:
        _ = [event async for event in session]

    response = next(message for message in session.new_messages() if isinstance(message, ModelResponse))
    call = next(part for part in response.parts if isinstance(part, ToolCallPart))
    assert call.otel_metadata == {'code_arg_name': 'python_code', 'code_arg_language': 'python'}
    chat = next(span for span in exporter.get_finished_spans() if span.name == 'chat')
    assert chat.attributes is not None
    output_messages = json.loads(str(chat.attributes['gen_ai.output.messages']))
    assert output_messages == snapshot(
        [
            {
                'role': 'assistant',
                'parts': [
                    {
                        'type': 'tool_call',
                        'id': 'c1',
                        'name': 'run_code',
                        'code_arg_name': 'python_code',
                        'code_arg_language': 'python',
                        'arguments': '{"python_code":"1 + 1"}',
                    }
                ],
            }
        ]
    )


async def test_request_config_respects_include_model_request_parameters() -> None:
    # `include_model_request_parameters=False` drops only the serialized config blobs. The tool definitions
    # and `gen_ai.request.max_tokens` remain, matching the classic instrumented-model path.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    settings = InstrumentationSettings(tracer_provider=provider, include_model_request_parameters=False)
    conn = _Connection([ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}'), ResponseDone()])
    session = RealtimeSession(
        conn,
        _ok_runner,
        instrumentation=settings,
        model_name='gpt-realtime',
        model_request_parameters=ModelRequestParameters(
            function_tools=[ToolDefinition(name='get_weather')], native_tools=[WebSearchTool()]
        ),
        model_settings=RealtimeModelSettings(max_tokens=4096),
    )
    async with session:
        _ = [e async for e in session]

    spans = {s.name: s for s in exporter.get_finished_spans()}
    for name in ('invoke_agent agent', 'chat gpt-realtime'):
        attributes = spans[name].attributes
        assert attributes is not None
        assert 'model_request_parameters' not in attributes
        assert 'model_settings' not in attributes
        assert json.loads(str(attributes['gen_ai.tool.definitions'])) == [
            {'type': 'function', 'name': 'get_weather', 'parameters': {'type': 'object', 'properties': {}}}
        ]
        # The spec-standard setting is still emitted when the serialized blobs are gated off.
        assert attributes['gen_ai.request.max_tokens'] == 4096


async def test_session_span_records_lifecycle_spans() -> None:
    # Barge-ins and turn boundaries have no span of their own, so they surface as zero-duration child
    # spans under the session span, making the stream's progression visible. Names are lowercase;
    # `interrupted` is attached only when true (a clean turn carries no null attribute). The user's
    # speech is the exception: it has a real duration, so it gets a `listen` span rather than a marker.
    settings, exporter = _settings()
    conn = _Connection(
        [
            RealtimeInputSpeechStartEvent(),
            OutputTranscript(text='wait', is_final=True),
            ResponseDone(interrupted=True),
        ]
    )
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = await collect_events(session)

    spans = {s.name: s for s in exporter.get_finished_spans()}
    session_span = spans['invoke_agent agent']
    assert dict(spans['model turn complete'].attributes or {}) == {
        'pydantic_ai.realtime': True,
        'logfire.msg': 'model turn complete (interrupted)',
        'interrupted': True,
    }
    # No `user speech` span here: this stream never reports the end of speech, and its length is not
    # something to guess at.
    assert 'user speech' not in spans
    # They nest under the session span, not the `chat` span.
    assert session_span.context is not None
    for name in ('model turn complete',):
        parent = spans[name].parent
        assert parent is not None and parent.span_id == session_span.context.span_id


async def test_user_speech_span_covers_the_spoken_segment() -> None:
    """The span measures how long the user talked: onset to end, backdated so the duration is real.

    It has to close before the model's own turn rather than run alongside it, or a trace can't show
    the hand-off a voice conversation is made of.
    """
    settings, exporter = _settings()
    conn = _Connection(
        [
            RealtimeInputSpeechStartEvent(),
            RealtimeInputSpeechEndEvent(),
            OutputTranscript(text='hi', is_final=True),
            ResponseDone(),
        ]
    )
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = await collect_events(session)

    spans = {s.name: s for s in exporter.get_finished_spans()}
    speech, chat = spans['user speech'], spans['chat gpt-realtime']
    assert speech.start_time is not None and speech.end_time is not None
    assert speech.end_time > speech.start_time, 'backdated to the onset, so it has a real duration'
    assert chat.start_time is not None and speech.end_time <= chat.start_time


async def test_no_user_speech_span_without_a_speech_end_event() -> None:
    """Gemini Live reports speech onset but never its end, so no span is recorded at all.

    Inferring the end from the model's reply would report a duration nobody measured; leaving the
    span open would report a user who talked until the session closed. Both are worse than silence.
    """
    settings, exporter = _settings()
    conn = _Connection([RealtimeInputSpeechStartEvent(), OutputTranscript(text='hi', is_final=True), ResponseDone()])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = await collect_events(session)

    assert [s.name for s in exporter.get_finished_spans() if s.name == 'user speech'] == []


async def test_no_user_speech_span_when_the_session_closes_mid_sentence() -> None:
    """A session torn down mid-sentence never learns how long the sentence was, so it records none."""
    settings, exporter = _settings()
    session = RealtimeSession(
        _Connection([RealtimeInputSpeechStartEvent()]), _ok_runner, instrumentation=settings, model_name='gpt-realtime'
    )
    _ = await collect_events(session)

    assert [s.name for s in exporter.get_finished_spans() if s.name == 'user speech'] == []


async def test_session_span_turn_complete_omits_interrupted_when_false() -> None:
    # A clean (uninterrupted) turn records the `model turn complete` span with no `interrupted` attribute,
    # rather than a null one.
    settings, exporter = _settings()
    conn = _Connection([OutputTranscript(text='hi', is_final=True), ResponseDone()])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = await collect_events(session)

    turn_complete = next(s for s in exporter.get_finished_spans() if s.name == 'model turn complete')
    assert dict(turn_complete.attributes or {}) == {'pydantic_ai.realtime': True}


async def test_session_span_name_follows_instrumentation_version() -> None:
    # The session span follows the configured instrumentation version's agent-run naming: semconv
    # `invoke_agent {name}` from v3 on, and a bare operation name on v2 (where the classic agent-run
    # span is likewise a bare `agent run`).
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with pytest.warns(PydanticAIDeprecationWarning, match='versions 2, 3, and 4 are deprecated'):
        settings = InstrumentationSettings(tracer_provider=provider, version=2)
    conn = _Connection([ResponseDone()])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = await collect_events(session)

    assert [s.name for s in exporter.get_finished_spans() if s.name != 'model turn complete'] == snapshot(['realtime'])


async def test_chat_span_records_interrupted_response_state() -> None:
    # A response cut off by a barge-in is recorded on the span covering that response, so a reader
    # can see *which* response was cut off rather than only that an interruption happened somewhere.
    # A response that ends normally carries no state attribute.
    settings, exporter = _settings()
    conn = _Connection([OutputTranscript(text='hello there', is_final=False), ResponseDone(interrupted=True)])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = await collect_events(session)

    chat = next(s for s in exporter.get_finished_spans() if s.name == 'chat gpt-realtime')
    assert chat.attributes is not None
    assert chat.attributes['pydantic_ai.response.state'] == 'interrupted'

    settings, exporter = _settings()
    conn = _Connection([OutputTranscript(text='hello there', is_final=False), ResponseDone()])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = await collect_events(session)

    chat = next(s for s in exporter.get_finished_spans() if s.name == 'chat gpt-realtime')
    assert chat.attributes is not None
    assert 'pydantic_ai.response.state' not in chat.attributes


async def test_interrupt_records_lifecycle_span_with_audio_offset() -> None:
    # A barge-in records an `interrupt` lifecycle span; when the caller passes `played_ms` (the ms of
    # output audio actually played before truncating), it's recorded so the trace shows how far the
    # response got before the user cut in.
    settings, exporter = _settings()
    session = RealtimeSession(
        _Connection([ResponseDone()]), _ok_runner, instrumentation=settings, model_name='gpt-realtime'
    )
    async with session:
        await session.interrupt(played_ms=1500)
        _ = [event async for event in session]

    interrupt = next(s for s in exporter.get_finished_spans() if s.name == 'interrupt')
    assert dict(interrupt.attributes or {}) == {'pydantic_ai.realtime': True, 'played_ms': 1500}


async def test_interrupt_without_offset_records_bare_lifecycle_span() -> None:
    # A cancel without truncation (no `played_ms`) still records the `interrupt` marker, with no
    # null attribute — just the realtime marker every session span carries.
    settings, exporter = _settings()
    session = RealtimeSession(
        _Connection([ResponseDone()]), _ok_runner, instrumentation=settings, model_name='gpt-realtime'
    )
    async with session:
        await session.interrupt()
        _ = [event async for event in session]

    interrupt = next(s for s in exporter.get_finished_spans() if s.name == 'interrupt')
    assert dict(interrupt.attributes or {}) == {'pydantic_ai.realtime': True}


async def test_unnamed_agent_session_span_defaults_agent_name() -> None:
    # An agent with no `name=` still gets `agent_name='agent'` on its session span (both the semconv
    # `gen_ai.agent.name` and legacy `agent_name` keys), mirroring the classic run span. Backends that
    # group runs by `agent_name` (e.g. Logfire's Runs view) would otherwise skip an unnamed session.
    settings, exporter = _settings()
    agent: Agent[None, str] = Agent()
    agent.instrument = settings
    conn = _Connection([OutputTranscript(text='hi', is_final=True), ResponseDone()])

    async with agent.realtime(_Model(conn)).session() as session:
        _ = [event async for event in session]

    sess = next(s for s in exporter.get_finished_spans() if s.name == 'invoke_agent agent')
    assert sess.attributes is not None
    assert sess.attributes['gen_ai.agent.name'] == 'agent'
    assert sess.attributes['agent_name'] == 'agent'


async def test_output_type_reflects_text_modality() -> None:
    # With `output_modality='text'` the model replies as plain text rather than speech, and the
    # session and `chat` spans report `gen_ai.output.type='text'` (threaded from the model settings
    # by `Agent.realtime_session`).
    settings, exporter = _settings()
    agent = _weather_agent(name='assistant')
    agent.instrument = settings
    conn = _Connection([OutputTranscript(text='hi', is_final=True, output_text=True), ResponseDone()])
    async with agent.realtime(
        _Model(conn), model_settings=RealtimeModelSettings(output_modality='text')
    ).session() as session:
        _ = [e async for e in session]
    spans = {s.name: s for s in exporter.get_finished_spans()}
    for name in ('invoke_agent assistant', 'chat gpt-realtime'):
        attributes = spans[name].attributes
        assert attributes is not None
        assert attributes['gen_ai.output.type'] == 'text'


async def test_output_type_reflects_actual_speech_despite_text_request() -> None:
    settings, exporter = _settings()
    agent = _weather_agent(name='assistant')
    agent.instrument = settings
    # xAI ignores `output_modality='text'`; an audio transcript is definitive evidence that the
    # provider produced speech, so both session and response telemetry must report speech.
    conn = _Connection([OutputTranscript(text='hi', is_final=True), ResponseDone()])
    async with agent.realtime(
        _Model(conn), model_settings=RealtimeModelSettings(output_modality='text')
    ).session() as session:
        _ = [e async for e in session]
    spans = {s.name: s for s in exporter.get_finished_spans()}
    for name in ('invoke_agent assistant', 'chat gpt-realtime'):
        attributes = spans[name].attributes
        assert attributes is not None
        assert attributes['gen_ai.output.type'] == 'speech'


async def test_include_content_false_omits_args_and_result() -> None:
    settings, exporter = _settings(include_content=False)
    agent = _weather_agent()
    agent.instrument = settings
    conn = _Connection([ToolCall(tool_call_id='c', tool_name='get_weather', args='{"city": "Paris"}'), ResponseDone()])
    async with agent.realtime(_Model(conn)).session() as session:
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
            OutputTranscript(text='let me check'),
            ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}'),
            OutputTranscript(text='it is sunny'),
            SessionUsage(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            ResponseDone(),
        ]
    )
    async with agent.realtime(_Model(conn)).session() as session:
        _ = [e async for e in session]

    finished = exporter.get_finished_spans()
    sess = next(s for s in finished if s.name == 'invoke_agent agent')
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
            OutputTranscript(text='let me check'),
            ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}'),
            OutputTranscript(text='it is sunny'),
            SessionUsage(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            ResponseDone(),
            InputTranscript(text='and tomorrow?', is_final=True),
            OutputTranscript(text='also sunny'),
            ResponseDone(),
        ]
    )
    async with agent.realtime(_Model(conn)).session() as session:
        _ = [e async for e in session]

    # Two turns → three `chat` spans (the first turn splits around the tool call) plus one
    # `execute_tool` span, all direct children of the one session span. Children are ordered by start
    # time: validation completes before the call event is surfaced, so the tool begins after the first
    # `chat` span and remains a sibling of the later response spans.
    assert _span_tree(exporter) == snapshot(
        [
            {
                'invoke_agent assistant': [
                    {'chat gpt-realtime': []},
                    {'execute_tool get_weather': []},
                    {'chat gpt-realtime': []},
                    {'chat gpt-realtime': []},
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
    conn = _Connection([ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}'), ResponseDone()])
    async with agent.realtime(_Model(conn)).session() as session:
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
            SessionUsage(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            ResponseDone(),
        ]
    )
    async with agent.realtime(_Model(conn)).session() as session:
        _ = [e async for e in session]
    spans = {s.name for s in exporter.get_finished_spans()}
    # No `model turn complete` span: the tool round is all this connection yields, so the model's follow-up
    # response — where the exchange would end — never arrives.
    assert spans == {'invoke_agent assistant', 'chat gpt-realtime', 'execute_tool get_weather'}


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
            SessionUsage(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            ResponseDone(),
        ]
    )
    async with agent.realtime(_Model(conn)).session() as session:
        _ = [e async for e in session]
    assert {s.name for s in cap_exporter.get_finished_spans()} == {
        'invoke_agent assistant',
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
            OutputTranscript(text='hi, how can I help?', is_final=True),
            ResponseDone(),
        ]
    )
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = await collect_events(session)

    sess = next(s for s in exporter.get_finished_spans() if s.name == 'invoke_agent agent')
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
    # `final_result` mirrors the classic run span: the most recent assistant reply, which the Logfire UI
    # renders as the run's final response.
    assert sess.attributes['final_result'] == 'hi, how can I help?'


async def test_session_span_counts_dropped_audio_chunks() -> None:
    settings, exporter = _settings()
    chunks = [bytes([index]) for index in range(40)]
    session = RealtimeSession(
        _Connection([AudioDelta(chunk) for chunk in chunks]),
        _ok_runner,
        instrumentation=settings,
        model_name='gpt-realtime',
    )

    async with session:
        assert [chunk async for chunk in session.stream_audio()] == chunks[-32:]

    sess = next(s for s in exporter.get_finished_spans() if s.name == 'invoke_agent agent')
    assert sess.attributes is not None
    assert sess.attributes['pydantic_ai.audio_chunks_dropped'] == 8
    assert sess.attributes['pydantic_ai.transcript_items_dropped'] == 0


async def test_session_span_counts_dropped_transcript_items() -> None:
    settings, exporter = _settings()
    transcripts = [str(index) for index in range(520)]
    session = RealtimeSession(
        _Connection([InputTranscript(text=transcript, is_final=True) for transcript in transcripts]),
        _ok_runner,
        instrumentation=settings,
        model_name='gpt-realtime',
    )

    async with session:
        parts = [part async for part in session.stream_transcripts()]
        assert [part.transcript for part in parts] == transcripts[-512:]

    sess = next(s for s in exporter.get_finished_spans() if s.name == 'invoke_agent agent')
    assert sess.attributes is not None
    assert sess.attributes['pydantic_ai.audio_chunks_dropped'] == 0
    assert sess.attributes['pydantic_ai.transcript_items_dropped'] == 8


async def test_session_span_includes_resolved_run_attributes() -> None:
    settings, exporter = _settings()
    agent: Agent[None, str] = Agent(
        name='assistant',
        description='Handles realtime conversations.',
        instructions='Keep answers concise.',
    )
    agent.instrument = settings
    conn = _Connection([OutputTranscript(text='hello', is_final=True), ResponseDone()])

    async with agent.realtime(_Model(conn), metadata={'tier': 'gold'}).session() as session:
        _ = [event async for event in session]

    sess = next(s for s in exporter.get_finished_spans() if s.name == 'invoke_agent assistant')
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
        # The agent always resolves a (possibly empty) `ModelRequestParameters`, declared here so the
        # session span renders it as an object.
        'model_request_parameters': {'type': 'object'},
    }


async def test_session_span_marks_seeded_history_boundary() -> None:
    # A session seeded with `message_history=` includes the seeded messages in
    # `pydantic_ai.all_messages` and marks where this session's own messages begin with
    # `pydantic_ai.new_message_index`, exactly like a classic run given `message_history=`.
    settings, exporter = _settings()
    conn = _Connection([InputTranscript(text='and now?', is_final=True), ResponseDone()])
    seeded: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='earlier question')]),
        ModelResponse(parts=[TextPart(content='earlier answer')]),
    ]
    session = RealtimeSession(
        conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime', message_history=seeded
    )
    _ = await collect_events(session)

    sess = next(s for s in exporter.get_finished_spans() if s.name == 'invoke_agent agent')
    assert sess.attributes is not None
    assert sess.attributes['pydantic_ai.new_message_index'] == 2
    all_messages = json.loads(str(sess.attributes['pydantic_ai.all_messages']))
    assert [m['role'] for m in all_messages] == ['user', 'assistant', 'user']
    assert all_messages[2]['parts'] == [{'type': 'text', 'content': 'and now?'}]


async def test_include_content_false_redacts_transcript_messages() -> None:
    # With `include_content=False` the conversation *structure* is still emitted (matching the
    # classic agent-run span); per-part content is redacted by `otel_message_parts`.
    settings, exporter = _settings(include_content=False)
    conn = _Connection(
        [
            InputTranscript(text='secret', is_final=True),
            OutputTranscript(text='secret reply', is_final=True),
            ResponseDone(),
        ]
    )
    session = RealtimeSession(
        conn,
        _ok_runner,
        instrumentation=settings,
        model_name='gpt-realtime',
        instructions='secret instructions',
        metadata={'tier': 'gold'},
    )
    _ = await collect_events(session)
    sess = next(s for s in exporter.get_finished_spans() if s.name == 'invoke_agent agent')
    assert sess.attributes is not None
    assert json.loads(str(sess.attributes['pydantic_ai.all_messages'])) == [
        {'role': 'user', 'parts': [{'type': 'text'}]},
        {'role': 'assistant', 'parts': [{'type': 'text'}], 'finish_reason': 'stop'},
    ]
    assert 'secret' not in str(sess.attributes['pydantic_ai.all_messages'])
    assert 'gen_ai.system_instructions' not in sess.attributes
    assert json.loads(str(sess.attributes['metadata'])) == {'tier': 'gold'}
    # `final_result` carries reply content, so it is redacted with `include_content=False`.
    assert 'final_result' not in sess.attributes


@pytest.mark.parametrize('include_binary_content', [False, True])
async def test_session_span_respects_include_binary_content(include_binary_content: bool) -> None:
    settings, exporter = _settings(include_binary_content=include_binary_content)
    audio = BinaryContent(data=b'secret-audio', media_type='audio/wav')
    seeded: list[ModelMessage] = [ModelRequest(parts=[SpeechPart(speaker='user', audio=audio)])]
    session = RealtimeSession(
        _Connection([]),
        _ok_runner,
        instrumentation=settings,
        message_history=seeded,
        metadata={'sample': audio},
    )
    _ = await collect_events(session)

    session_span = next(span for span in exporter.get_finished_spans() if span.name == 'invoke_agent agent')
    assert session_span.attributes is not None
    serialized = str(session_span.attributes['pydantic_ai.all_messages'])
    assert ('c2VjcmV0LWF1ZGlv' in serialized) is include_binary_content
    # The user-supplied session `metadata` is redacted the same way, matching the classic run span.
    assert ('c2VjcmV0LWF1ZGlv' in str(session_span.attributes['metadata'])) is include_binary_content


async def test_session_span_sets_conversation_id() -> None:
    # `conversation_id` lands on the session span under the same key the classic agent-run span uses
    # (`gen_ai.conversation.id`), so a realtime session can be correlated with other runs.
    settings, exporter = _settings()
    agent = _weather_agent()
    agent.instrument = settings
    conn = _Connection([ResponseDone()])
    async with agent.realtime(_Model(conn), conversation_id='conv-123').session() as session:
        _ = [event async for event in session]
    sess = next(s for s in exporter.get_finished_spans() if s.name == 'invoke_agent agent')
    assert sess.attributes is not None
    assert sess.attributes['gen_ai.conversation.id'] == 'conv-123'


async def test_session_span_omits_conversation_id_when_unset() -> None:
    settings, exporter = _settings()
    conn = _Connection([ResponseDone()])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = await collect_events(session)
    sess = next(s for s in exporter.get_finished_spans() if s.name == 'invoke_agent agent')
    assert sess.attributes is not None
    assert 'gen_ai.conversation.id' not in sess.attributes


async def test_session_span_without_model_or_usage() -> None:
    settings, exporter = _settings()
    conn = _Connection([ResponseDone()])  # no model name, no Usage event
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings)
    _ = await collect_events(session)
    sess = next(s for s in exporter.get_finished_spans() if s.name == 'invoke_agent agent')
    assert sess.attributes is not None
    assert sess.attributes['gen_ai.operation.name'] == 'invoke_agent'
    assert 'model_name' not in sess.attributes
    # The agent name defaults to `'agent'` even without an explicit name, mirroring the classic run span.
    assert sess.attributes['gen_ai.agent.name'] == 'agent'
    assert 'gen_ai.agent.description' not in sess.attributes
    assert 'gen_ai.system_instructions' not in sess.attributes
    assert 'metadata' not in sess.attributes
    assert 'gen_ai.usage.input_tokens' not in sess.attributes  # zero usage → no token attribute / metric
    # An empty turn produces no assistant `ModelResponse`, so no `chat` span is opened.
    assert not [s for s in exporter.get_finished_spans() if s.name.startswith('chat')]


async def test_non_recording_spans_skip_events_but_still_record_metrics() -> None:
    metric_reader = InMemoryMetricReader()
    settings = InstrumentationSettings(
        tracer_provider=TracerProvider(sampler=ALWAYS_OFF),
        meter_provider=MeterProvider(metric_readers=[metric_reader]),
    )
    session = RealtimeSession(
        _Connection([OutputTranscript(text='hello'), ResponseDone()]),
        _ok_runner,
        instrumentation=settings,
        model_name='gpt-realtime',
    )

    _ = await collect_events(session)
    span = settings.tracer.start_span('not-recorded')
    assert not span.is_recording()
    session._session_instrumentation.record_error(span, RuntimeError('not recorded'))  # pyright: ignore[reportPrivateUsage]

    metrics = metric_reader.get_metrics_data()
    assert metrics is not None


def test_chat_span_without_response_ends_without_metrics() -> None:
    settings, exporter, metric_reader = _settings_with_metrics()
    session = RealtimeSession(_Connection([]), _ok_runner, instrumentation=settings)
    session._session_instrumentation.context = Context()  # pyright: ignore[reportPrivateUsage]
    session._ensure_chat_span()  # pyright: ignore[reportPrivateUsage]

    session._session_instrumentation.end_chat_span([], None)  # pyright: ignore[reportPrivateUsage]

    assert [span.name for span in exporter.get_finished_spans()] == ['chat']
    assert metric_reader.get_metrics_data() is None


async def test_chat_span_closed_for_contentless_response() -> None:
    # Audio with no retained content still proves a response happened by opening a `chat` span, so
    # history and instrumentation retain its empty response envelope.
    settings, exporter = _settings()
    conn = _Connection([AudioDelta(data=b'\x00\x01'), ResponseDone()])
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = await collect_events(session)
    chat = next(s for s in exporter.get_finished_spans() if s.name == 'chat gpt-realtime')
    assert chat.attributes is not None
    assert json.loads(str(chat.attributes['gen_ai.output.messages'])) == [
        {'role': 'assistant', 'parts': [], 'finish_reason': 'stop'}
    ]
    response = session.new_messages()[0]
    assert isinstance(response, ModelResponse)
    assert response.parts == [SpeechPart(speaker='assistant')]
    assert session.usage.requests == 1


async def test_session_usage_without_aggregated_attribute_names() -> None:
    # With `use_aggregated_usage_attribute_names=False`, cumulative session usage stays under the
    # standard `gen_ai.usage.*` namespace instead of the aggregated one.
    settings, exporter = _settings(use_aggregated_usage_attribute_names=False)
    conn = _Connection(
        [
            InputTranscript(text='hi', is_final=True),
            OutputTranscript(text='hello'),
            SessionUsage(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            ResponseDone(),
        ]
    )
    session = RealtimeSession(conn, _ok_runner, instrumentation=settings, model_name='gpt-realtime')
    _ = await collect_events(session)
    sess = next(s for s in exporter.get_finished_spans() if s.name == 'invoke_agent agent')
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
            OutputTranscript(text='hi, how can I help?'),
            SessionUsage(
                usage=RequestUsage(input_tokens=10, output_tokens=4),
                provider_response_id='resp-1',
                finish_reason='stop',
            ),
            ResponseDone(provider_response_id='resp-1', finish_reason='stop'),
        ]
    )
    session = RealtimeSession(
        conn,
        _ok_runner,
        instrumentation=settings,
        model_name='gpt-realtime',
        provider_name='openai',
        provider_url='https://api.openai.com:8443/v1',
    )
    _ = await collect_events(session)

    chat = next(s for s in exporter.get_finished_spans() if s.name == 'chat gpt-realtime')
    assert chat.attributes is not None
    assert chat.attributes['gen_ai.operation.name'] == 'chat'
    assert chat.attributes['gen_ai.request.model'] == 'gpt-realtime'
    assert chat.attributes['gen_ai.response.model'] == 'gpt-realtime'
    assert chat.attributes['gen_ai.provider.name'] == 'openai'
    assert chat.attributes['gen_ai.system'] == 'openai'
    assert chat.attributes['server.address'] == 'api.openai.com'
    assert chat.attributes['server.port'] == 8443
    assert chat.attributes['gen_ai.response.id'] == 'resp-1'
    assert chat.attributes['gen_ai.response.finish_reasons'] == ('stop',)
    cost = chat.attributes['operation.cost']
    assert isinstance(cost, (int, float)) and cost > 0
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
    # Honest omissions vs. the classic `chat` span: no per-turn request parameters/settings.
    for omitted in (
        'model_request_parameters',
        'gen_ai.request.temperature',
    ):
        assert omitted not in chat.attributes


async def test_per_response_token_metrics_match_classic_dimensions() -> None:
    settings, _, metric_reader = _settings_with_metrics()
    conn = _Connection(
        [
            InputTranscript(text='first', is_final=True),
            OutputTranscript(text='one'),
            SessionUsage(usage=RequestUsage(input_tokens=10, output_tokens=4)),
            ResponseDone(),
            InputTranscript(text='second', is_final=True),
            OutputTranscript(text='two'),
            SessionUsage(usage=RequestUsage(input_tokens=7, output_tokens=3)),
            ResponseDone(),
        ]
    )
    session = RealtimeSession(
        conn,
        _ok_runner,
        instrumentation=settings,
        model_name='gpt-realtime',
        provider_name='openai',
        provider_url='https://api.openai.com/v1',
    )
    _ = await collect_events(session)

    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is not None
    token_metric = next(
        metric
        for resource in metrics_data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == 'gen_ai.client.token.usage'
    )
    assert isinstance(token_metric.data, Histogram)
    points: dict[str, dict[str, Any]] = {}
    for point in token_metric.data.data_points:
        attributes = dict(point.attributes or {})
        token_type = attributes['gen_ai.token.type']
        assert isinstance(token_type, str)
        points[token_type] = {
            'attributes': attributes,
            'count': point.count,
            'sum': point.sum,
        }
    assert points == {
        'input': {
            'attributes': {
                'gen_ai.provider.name': 'openai',
                'gen_ai.system': 'openai',
                'gen_ai.operation.name': 'chat',
                'gen_ai.request.model': 'gpt-realtime',
                'gen_ai.response.model': 'gpt-realtime',
                'gen_ai.token.type': 'input',
            },
            'count': 2,
            'sum': 17,
        },
        'output': {
            'attributes': {
                'gen_ai.provider.name': 'openai',
                'gen_ai.system': 'openai',
                'gen_ai.operation.name': 'chat',
                'gen_ai.request.model': 'gpt-realtime',
                'gen_ai.response.model': 'gpt-realtime',
                'gen_ai.token.type': 'output',
            },
            'count': 2,
            'sum': 7,
        },
    }


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
            OutputTranscript(text='secret answer'),
            ResponseDone(),
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
            OutputTranscript(text='let me check'),
            ToolCall(tool_call_id='c1', tool_name='get_weather', args='{"city": "Paris"}'),
            OutputTranscript(text='it is sunny'),
            ResponseDone(),
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
    conn = _Connection([OutputTranscript(text='hello'), ResponseDone()])
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
    )  # no ResponseDone

    with caplog.at_level(logging.ERROR, logger='opentelemetry'):
        async with agent.realtime(_Model(conn)).session() as session:
            async for _ in session:
                break

    # These are all spans this path starts. They must be exported before the owner block returns,
    # without GC or extra event-loop turns, and the session remains the explicit parent of `chat`.
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {'invoke_agent agent', 'chat gpt-realtime'}
    session_span = spans['invoke_agent agent']
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
        # Tool is cancelled first.
        async def send(self, content: RealtimeInput) -> None:  # pragma: no cover
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
        async with agent.realtime(_Model(_IdleAfterTool())).session() as session:
            async for _ in session:
                await started.wait()
                break

    assert tool_task is not None and tool_task.done() and tool_task.cancelled()
    assert cancelled.is_set()
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {'invoke_agent agent', 'chat gpt-realtime', 'execute_tool get_weather'}
    session_span = spans['invoke_agent agent']
    assert session_span.context is not None
    for child_name in ('chat gpt-realtime', 'execute_tool get_weather'):
        parent = spans[child_name].parent
        assert parent is not None and parent.span_id == session_span.context.span_id
    assert not any(
        'Failed to detach context' in record.getMessage() or 'different Context' in record.getMessage()
        for record in caplog.records
    )


def test_provider_attributes_degrade_on_malformed_port() -> None:
    # `urlparse` only re-parses the port on access, so a malformed one must cost the server
    # attributes rather than crash span setup.
    attributes = provider_attributes('openai', 'https://host:not-a-port/v1')
    assert attributes['gen_ai.provider.name'] == 'openai'
    assert 'server.address' not in attributes
    assert 'server.port' not in attributes
