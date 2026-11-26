"""`pydantic_ai.telemetry`: spans that follow the right run's tracer and content policy.

These are unit tests, not VCR tests: what they check is span attributes and the `INVALID_SPAN` and
`NoOpTracer` branches, which no provider response can show, so the runs use `TestModel` or no model.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from inline_snapshot import snapshot
from opentelemetry.trace import INVALID_SPAN, NoOpTracer
from opentelemetry.util.types import AttributeValue

from pydantic_ai import Agent, RunContext
from pydantic_ai._telemetry import run_telemetry_scope
from pydantic_ai.capabilities import AbstractCapability, Instrumentation, ValidatedToolArgs
from pydantic_ai.durable_exec import DurableRunContext
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.telemetry import (
    RunTelemetry,
    current_run_telemetry,
    current_span,
    open_span,
    record_decision,
    records_content,
    set_content_attribute,
)
from pydantic_ai.tools import ToolDefinition

from .conftest import Instrumented, try_import

with try_import() as otel_sdk_installed:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytestmark = pytest.mark.skipif(not otel_sdk_installed(), reason='opentelemetry-sdk not installed')


@dataclass
class TraceWhenPrepared(AbstractCapability[object]):
    """Records a span and a decision from `for_run`, which runs before the run publishes its telemetry."""

    async def for_run(self, ctx: RunContext[object]) -> AbstractCapability[object]:
        with open_span(
            ctx, 'example prepare', attributes={'example.kind': 'setup'}, content={'example.input': 'a'}
        ) as span:
            set_content_attribute(ctx, span, 'example.output', 'b')
        record_decision(ctx, 'example decided', attributes={'example.kind': 'setup'}, content={'example.detail': 'c'})
        return self


def example_spans(exporter: InMemorySpanExporter) -> list[dict[str, AttributeValue]]:
    return [
        {'name': span.name, **(span.attributes or {})}
        for span in exporter.get_finished_spans()
        if span.name.startswith('example ')
    ]


async def test_spans_in_a_sub_agents_for_run_follow_the_sub_agent(instrumented: Instrumented):
    """A sub-agent's `for_run` traces to the sub-agent's tracer with its content policy, not its parent's.

    `for_run` runs before the sub-agent publishes its own telemetry, so only its `ctx` can say which run
    it belongs to: the value published at that point is the parent's.
    """
    parent_instrumentation, parent_exporter = instrumented(False)
    child_instrumentation, child_exporter = instrumented(True)
    child = Agent(TestModel(), capabilities=[TraceWhenPrepared(), child_instrumentation])
    parent = Agent(TestModel(call_tools=['delegate']), capabilities=[parent_instrumentation])

    @parent.tool_plain
    async def delegate() -> str:
        return (await child.run('go')).output

    await parent.run('go')

    assert example_spans(parent_exporter) == []
    assert example_spans(child_exporter) == snapshot(
        [
            {'name': 'example prepare', 'example.kind': 'setup', 'example.input': 'a', 'example.output': 'b'},
            {'name': 'example decided', 'example.kind': 'setup', 'example.detail': 'c'},
        ]
    )


async def test_content_is_left_out_without_include_content(instrumented: Instrumented):
    instrumentation, exporter = instrumented(False)
    await Agent(TestModel(), capabilities=[TraceWhenPrepared(), instrumentation]).run('go')

    assert example_spans(exporter) == snapshot(
        [{'name': 'example prepare', 'example.kind': 'setup'}, {'name': 'example decided', 'example.kind': 'setup'}]
    )


@pytest.mark.parametrize(('parent_content', 'child_content'), [(True, False), (False, True), (True, True)])
async def test_a_sub_agent_on_its_parents_tracer_follows_its_own_content_setting(
    parent_content: bool, child_content: bool
):
    """Agents on one tracer provider share tracers, but a sub-agent's `for_run` still follows the sub-agent's setting."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    child = Agent(
        TestModel(),
        capabilities=[
            TraceWhenPrepared(),
            Instrumentation(settings=InstrumentationSettings(tracer_provider=provider, include_content=child_content)),
        ],
    )
    parent = Agent(
        TestModel(call_tools=['ask_child']),
        capabilities=[
            Instrumentation(settings=InstrumentationSettings(tracer_provider=provider, include_content=parent_content))
        ],
    )

    @parent.tool_plain
    async def ask_child() -> str:
        return (await child.run('go')).output

    await parent.run('go')

    prepare: dict[str, AttributeValue] = {'name': 'example prepare', 'example.kind': 'setup'}
    decided: dict[str, AttributeValue] = {'name': 'example decided', 'example.kind': 'setup'}
    if child_content:
        prepare |= {'example.input': 'a', 'example.output': 'b'}
        decided |= {'example.detail': 'c'}
    assert example_spans(exporter) == [prepare, decided]


@dataclass
class MarkToolCall(AbstractCapability[object]):
    async def before_tool_execute(
        self, ctx: RunContext[object], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        current_span(ctx).set_attribute('example.marked', True)
        return args


def agent_with_tool(*capabilities: AbstractCapability[object]) -> Agent[None, str]:
    agent = Agent(TestModel(call_tools=['noop']), capabilities=[MarkToolCall(), *capabilities])

    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    return agent


async def test_current_span_in_a_tool_hook_is_the_calls_execute_tool_span(instrumented: Instrumented):
    instrumentation, exporter = instrumented(False)
    await agent_with_tool(instrumentation).run('go')

    assert [span.name for span in exporter.get_finished_spans() if (span.attributes or {}).get('example.marked')] == [
        'execute_tool noop'
    ]


async def test_current_span_leaves_the_applications_span_alone_in_an_uninstrumented_run():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer('app').start_as_current_span('app request'):
        await agent_with_tool().run('go')

    assert [(span.name, dict(span.attributes or {})) for span in exporter.get_finished_spans()] == [('app request', {})]


async def test_current_span_in_an_uninstrumented_sub_agent_leaves_the_parents_span_alone(instrumented: Instrumented):
    """The parent's telemetry is published around the sub-agent, but the sub-agent's own run isn't traced."""
    instrumentation, exporter = instrumented(False)
    child = agent_with_tool()
    parent = Agent(TestModel(call_tools=['delegate']), capabilities=[instrumentation])

    @parent.tool_plain
    async def delegate() -> str:
        return (await child.run('go')).output

    await parent.run('go')

    assert [span.name for span in exporter.get_finished_spans() if (span.attributes or {}).get('example.marked')] == []


@dataclass
class MarkWhenPrepared(AbstractCapability[object]):
    async def for_run(self, ctx: RunContext[object]) -> AbstractCapability[object]:
        set_content_attribute(ctx, current_span(ctx), 'example.prompt', 'secret')
        return self


async def run_marking_sub_agent(parent: AbstractCapability[object], child: AbstractCapability[object]) -> None:
    """Run a parent whose tool runs a sub-agent that writes a content attribute to `current_span` in `for_run`."""
    sub_agent = Agent(TestModel(), capabilities=[MarkWhenPrepared(), child])
    agent = Agent(TestModel(call_tools=['delegate']), capabilities=[parent])

    @agent.tool_plain
    async def delegate() -> str:
        return (await sub_agent.run('go')).output

    await agent.run('go')


def marked_spans(exporter: InMemorySpanExporter) -> list[str]:
    return [span.name for span in exporter.get_finished_spans() if 'example.prompt' in (span.attributes or {})]


async def test_a_for_run_does_not_write_to_the_applications_span(instrumented: Instrumented):
    """A top-level run's `for_run` runs before the run publishes, while the application's own span is open."""
    instrumentation, exporter = instrumented(True)
    app_exporter = InMemorySpanExporter()
    app_provider = TracerProvider()
    app_provider.add_span_processor(SimpleSpanProcessor(app_exporter))
    with app_provider.get_tracer('app').start_as_current_span('app request'):
        await Agent(TestModel(), capabilities=[MarkWhenPrepared(), instrumentation]).run('go')

    assert marked_spans(app_exporter) == []
    assert marked_spans(exporter) == []


async def test_a_sub_agents_for_run_does_not_write_to_its_parents_span(instrumented: Instrumented):
    """In a sub-agent's `for_run` the open span is the parent's tool call, which records no content here."""
    parent, parent_exporter = instrumented(False)
    child, _ = instrumented(True)
    await run_marking_sub_agent(parent, child)

    assert marked_spans(parent_exporter) == []


async def test_a_sub_agent_on_its_parents_tracer_and_setting_writes_to_the_parents_span():
    """With the parent's tracer and content setting, the sub-agent's `for_run` gets the parent's span, which allows it."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    settings = InstrumentationSettings(tracer_provider=provider, include_content=True)
    await run_marking_sub_agent(Instrumentation(settings=settings), Instrumentation(settings=settings))

    assert marked_spans(exporter) == ['execute_tool delegate']


def test_a_durable_unit_that_did_not_carry_the_tracer_is_untraced():
    """Reading an uncarried field of a `DurableRunContext` raises, so telemetry treats the unit as untraced."""
    ctx = DurableRunContext[None](deps=None)

    with open_span(ctx, 'example work') as span:
        set_content_attribute(ctx, span, 'example.detail', 'a')
    record_decision(ctx, 'example decided')

    assert span is INVALID_SPAN
    assert current_span(ctx) is INVALID_SPAN
    assert records_content(ctx) is False


async def test_code_without_a_ctx_traces_to_the_calling_run(instrumented: Instrumented):
    """A tool's helper with no `RunContext` reads the run's telemetry; outside a run there is none."""
    instrumentation, exporter = instrumented(False)
    agent = Agent(TestModel(call_tools=['noop']), capabilities=[instrumentation])

    @agent.tool_plain
    def noop() -> str:
        with open_span(current_run_telemetry(), 'example helper', content={'example.input': 'a'}):
            pass
        return 'ok'

    await agent.run('go')

    assert example_spans(exporter) == [{'name': 'example helper'}]
    with open_span(current_run_telemetry(), 'example outside a run') as span:
        assert span is INVALID_SPAN


@pytest.mark.parametrize('include_content', [True, False])
@pytest.mark.parametrize(
    ('error', 'not_errors', 'status'),
    [
        (asyncio.CancelledError('private'), (), 'ERROR'),
        (asyncio.CancelledError('private'), (asyncio.CancelledError,), 'UNSET'),
        (ValueError('private'), (), 'ERROR'),
        (FileNotFoundError('private'), (FileNotFoundError,), 'UNSET'),
    ],
)
def test_an_escaping_exception_is_recorded_by_its_type(
    error: BaseException, not_errors: tuple[type[BaseException], ...], status: str, include_content: bool
):
    """A cancellation is an error like any other unless the caller expects it; the message needs content."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    with pytest.raises(type(error)):
        with open_span(
            RunTelemetry(provider.get_tracer('test'), include_content=include_content),
            'example work',
            not_errors=not_errors,
        ):
            raise error

    [span] = exporter.get_finished_spans()
    assert span.status.status_code.name == status
    [event] = span.events
    assert str((event.attributes or {})['exception.type']).endswith(type(error).__qualname__)
    recorded = f'{span.status.description} {event.attributes}'
    assert ('private' in recorded) is include_content


def test_a_telemetry_scope_publishes_for_its_block_only():
    telemetry = RunTelemetry(TracerProvider().get_tracer('test'), include_content=False)

    with run_telemetry_scope(telemetry):
        assert current_run_telemetry() is telemetry
        with run_telemetry_scope(RunTelemetry(NoOpTracer(), include_content=True)):
            assert current_run_telemetry() is None
    assert current_run_telemetry() is None
