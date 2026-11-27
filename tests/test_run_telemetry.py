"""`open_span`, `record_decision` and `tool_call_span`: spans that follow the right run's tracer and content policy.

These are unit tests, not VCR tests: what they check is span attributes and the `INVALID_SPAN` and
`NoOpTracer` branches, which no provider response can show, so the runs use `TestModel` or no model.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from inline_snapshot import snapshot
from opentelemetry.trace import INVALID_SPAN, Tracer
from opentelemetry.util.types import AttributeValue

from pydantic_ai import Agent, AgentRunResult, RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    Instrumentation,
    RawToolArgs,
    ValidatedToolArgs,
    WrapRunHandler,
    WrapToolExecuteHandler,
    open_span,
    record_decision,
    tool_call_span,
)
from pydantic_ai.durable_exec import DurableRunContext
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.realtime.codec import ResponseDone, ToolCall
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.usage import RunUsage

from .conftest import Instrumented, try_import
from .realtime.test_session import FakeRealtimeConnection, FakeRealtimeModel

with try_import() as otel_sdk_installed:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytestmark = pytest.mark.skipif(not otel_sdk_installed(), reason='opentelemetry-sdk not installed')


@dataclass
class TraceBeforeRun(AbstractCapability[object]):
    """Records a span and a decision from `before_run`, which runs inside the run's span."""

    async def before_run(self, ctx: RunContext[object]) -> None:
        with open_span(
            ctx, 'example prepare', attributes={'example.kind': 'setup'}, content={'example.input': 'a'}
        ) as span:
            if ctx.trace_include_content:
                span.set_attribute('example.output', 'b')
        record_decision(ctx, 'example decided', attributes={'example.kind': 'setup'}, content={'example.detail': 'c'})


def example_spans(exporter: InMemorySpanExporter) -> list[dict[str, AttributeValue]]:
    return [
        {'name': span.name, **(span.attributes or {})}
        for span in exporter.get_finished_spans()
        if span.name.startswith('example ')
    ]


async def test_spans_in_a_sub_agents_run_hook_follow_the_sub_agent(instrumented: Instrumented):
    """A sub-agent's `before_run` traces to the sub-agent's tracer with its content policy, not its parent's."""
    parent_instrumentation, parent_exporter = instrumented(False)
    child_instrumentation, child_exporter = instrumented(True)
    child = Agent(TestModel(), capabilities=[TraceBeforeRun(), child_instrumentation])
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
    await Agent(TestModel(), capabilities=[TraceBeforeRun(), instrumentation]).run('go')

    assert example_spans(exporter) == snapshot(
        [{'name': 'example prepare', 'example.kind': 'setup'}, {'name': 'example decided', 'example.kind': 'setup'}]
    )


@pytest.mark.parametrize(('parent_content', 'child_content'), [(True, False), (False, True), (True, True)])
async def test_a_sub_agent_on_its_parents_tracer_follows_its_own_content_setting(
    parent_content: bool, child_content: bool
):
    """Agents on one tracer provider share tracers, but a sub-agent's `before_run` still follows its own setting."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    child = Agent(
        TestModel(),
        capabilities=[
            TraceBeforeRun(),
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
        tool_call_span(ctx).set_attribute('example.marked', True)
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[object],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        tool_call_span(ctx).set_attribute('example.marked_after', True)
        return result


def agent_with_tool(*capabilities: AbstractCapability[object]) -> Agent[None, str]:
    agent = Agent(TestModel(call_tools=['noop']), capabilities=[MarkToolCall(), *capabilities])

    @agent.tool
    def noop(ctx: RunContext[object]) -> str:
        tool_call_span(ctx).set_attribute('example.marked_in_body', True)
        return 'ok'

    return agent


def spans_with(exporter: InMemorySpanExporter, attribute: str) -> list[str]:
    return [span.name for span in exporter.get_finished_spans() if (span.attributes or {}).get(attribute)]


async def test_tool_call_span_in_a_tool_hook_and_body_is_the_calls_execute_tool_span(instrumented: Instrumented):
    instrumentation, exporter = instrumented(False)
    await agent_with_tool(instrumentation).run('go')

    assert spans_with(exporter, 'example.marked') == ['execute_tool noop']
    assert spans_with(exporter, 'example.marked_after') == ['execute_tool noop']
    assert spans_with(exporter, 'example.marked_in_body') == ['execute_tool noop']


class MarkingToolset(WrapperToolset[object]):
    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
    ) -> Any:
        tool_call_span(ctx).set_attribute('example.marked_in_call_tool', True)
        return await super().call_tool(name, tool_args, ctx, tool)


async def test_tool_call_span_in_a_toolsets_call_tool_is_the_calls_execute_tool_span(instrumented: Instrumented):
    instrumentation, exporter = instrumented(False)

    def noop() -> str:
        return 'ok'

    toolset = MarkingToolset(FunctionToolset([noop]))
    await Agent(TestModel(call_tools=['noop']), toolsets=[toolset], capabilities=[instrumentation]).run('go')

    assert spans_with(exporter, 'example.marked_in_call_tool') == ['execute_tool noop']


async def test_tool_call_span_leaves_the_applications_span_alone_in_an_uninstrumented_run():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer('app').start_as_current_span('app request'):
        await agent_with_tool().run('go')

    assert [(span.name, dict(span.attributes or {})) for span in exporter.get_finished_spans()] == [('app request', {})]


@dataclass
class MarkWhenPrepared(AbstractCapability[object]):
    async def for_run(self, ctx: RunContext[object]) -> AbstractCapability[object]:
        tool_call_span(ctx).set_attribute('example.prompt', 'secret')
        return self


@dataclass
class MarkAroundTheRun(AbstractCapability[object]):
    def get_ordering(self) -> CapabilityOrdering:
        # `Instrumentation` is outermost too; listed first, this one wraps it.
        return CapabilityOrdering(position='outermost')

    async def wrap_run(self, ctx: RunContext[object], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        tool_call_span(ctx).set_attribute('example.prompt', 'secret')
        return await handler()


async def test_a_wrap_run_outside_instrumentation_does_not_write_to_the_span_around_the_run(
    instrumented: Instrumented,
):
    """Ordered before `Instrumentation`, `wrap_run` runs before the run's span opens, around the application's span."""
    instrumentation, exporter = instrumented(True)
    app_exporter = InMemorySpanExporter()
    app_provider = TracerProvider()
    app_provider.add_span_processor(SimpleSpanProcessor(app_exporter))
    with app_provider.get_tracer('app').start_as_current_span('app request'):
        await Agent(TestModel(), capabilities=[MarkAroundTheRun(), instrumentation]).run('go')

    assert marked_spans(app_exporter) == []
    assert marked_spans(exporter) == []


async def run_marking_sub_agent(parent: AbstractCapability[object], child: AbstractCapability[object]) -> None:
    """Run a parent whose tool runs a sub-agent that writes an attribute to `tool_call_span` in `for_run`."""
    sub_agent = Agent(TestModel(), capabilities=[MarkWhenPrepared(), child])
    agent = Agent(TestModel(call_tools=['delegate']), capabilities=[parent])

    @agent.tool_plain
    async def delegate() -> str:
        return (await sub_agent.run('go')).output

    await agent.run('go')


def marked_spans(exporter: InMemorySpanExporter) -> list[str]:
    return [span.name for span in exporter.get_finished_spans() if 'example.prompt' in (span.attributes or {})]


async def test_a_sub_agents_for_run_does_not_write_to_its_parents_span():
    """Even on the parent's tracer and content setting: setup runs before the sub-agent's run publishes."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    settings = InstrumentationSettings(tracer_provider=provider, include_content=True)
    await run_marking_sub_agent(Instrumentation(settings=settings), Instrumentation(settings=settings))

    assert marked_spans(exporter) == []


def test_a_durable_unit_that_did_not_carry_the_tracer_is_untraced():
    """Reading an uncarried field of a `DurableRunContext` raises, so telemetry treats the unit as untraced."""
    ctx = DurableRunContext[None](deps=None)

    with open_span(ctx, 'example work') as span:
        pass
    record_decision(ctx, 'example decided')

    assert span is INVALID_SPAN
    assert tool_call_span(ctx) is INVALID_SPAN
    # Defaulted rather than guarded, so a capability that reads it records no content instead of raising.
    assert ctx.trace_include_content is False


def traced_context(tracer: Tracer, include_content: bool) -> RunContext[None]:
    return RunContext(
        deps=None, model=TestModel(), usage=RunUsage(), tracer=tracer, trace_include_content=include_content
    )


@pytest.mark.parametrize('include_content', [True, False])
def test_an_escaping_exception_is_recorded_by_its_type(include_content: bool):
    """The message can quote content, so it is recorded only when the run captures content."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    with pytest.raises(ValueError):
        with open_span(traced_context(provider.get_tracer('test'), include_content), 'example work'):
            raise ValueError('private')

    [span] = exporter.get_finished_spans()
    assert span.status.status_code.name == 'ERROR'
    assert (span.attributes or {})['error.type'] == 'ValueError'
    [event] = span.events
    assert (event.attributes or {})['exception.type'] == 'ValueError'
    recorded = f'{span.status.description} {event.attributes}'
    assert ('private' in recorded) is include_content


def test_a_cancellation_is_not_an_error():
    """As in OpenTelemetry's own spans, a cancellation leaves the status unset and records no exception."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    with pytest.raises(asyncio.CancelledError):
        with open_span(traced_context(provider.get_tracer('test'), True), 'example work'):
            raise asyncio.CancelledError

    [span] = exporter.get_finished_spans()
    assert (span.status.status_code.name, span.events) == ('UNSET', ())


class ExampleError(Exception):
    pass


def test_an_errored_span_sets_error_type_without_content():
    """`error.type` is the exception's fully-qualified type, set even when the run doesn't capture content."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    with pytest.raises(ExampleError):
        with open_span(traced_context(provider.get_tracer('test'), False), 'example work'):
            raise ExampleError('private')

    [span] = exporter.get_finished_spans()
    assert dict(span.attributes or {}) == {'error.type': 'tests.test_run_telemetry.ExampleError'}


@dataclass
class RecordToolCallSpanOutsideToolCalls(AbstractCapability[object]):
    """Records what `tool_call_span` returns in hooks that run outside any tool call."""

    seen: dict[str, object]

    async def for_run(self, ctx: RunContext[object]) -> AbstractCapability[object]:
        self.seen['for_run'] = tool_call_span(ctx)
        return self

    async def before_run(self, ctx: RunContext[object]) -> None:
        self.seen['before_run'] = tool_call_span(ctx)

    async def wrap_run(self, ctx: RunContext[object], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        self.seen['wrap_run'] = tool_call_span(ctx)
        return await handler()


async def test_tool_call_span_is_invalid_outside_a_tool_call(instrumented: Instrumented):
    """`for_run`, `before_run` and `wrap_run` run outside any tool call: no span, not the run's or the application's."""
    instrumentation, _ = instrumented(True)
    seen: dict[str, object] = {}
    with TracerProvider().get_tracer('app').start_as_current_span('app request'):
        await Agent(TestModel(), capabilities=[RecordToolCallSpanOutsideToolCalls(seen), instrumentation]).run('go')

    assert seen == {'for_run': INVALID_SPAN, 'before_run': INVALID_SPAN, 'wrap_run': INVALID_SPAN}


async def test_tool_call_span_is_invalid_in_a_sub_agent_run_inside_a_parent_tool(instrumented: Instrumented):
    """A sub-agent run inside a parent's tool call never gets the parent's `execute_tool` span."""
    instrumentation, _ = instrumented(True)
    seen: dict[str, object] = {}
    child = Agent(TestModel(), capabilities=[RecordToolCallSpanOutsideToolCalls(seen), instrumentation])
    parent = Agent(TestModel(call_tools=['delegate']), capabilities=[instrumentation])

    @parent.tool_plain
    async def delegate() -> str:
        await child.run('go')
        return 'ok'

    await parent.run('go')

    assert seen == {'for_run': INVALID_SPAN, 'before_run': INVALID_SPAN, 'wrap_run': INVALID_SPAN}


async def test_tool_call_span_is_invalid_in_a_realtime_sessions_before_run(instrumented: Instrumented):
    """A realtime session's `before_run` runs outside any tool call; a tool call in the session still gets its span."""
    instrumentation, exporter = instrumented(False)
    seen: dict[str, object] = {}

    def noop() -> str:
        return 'ok'

    connection = FakeRealtimeConnection([ToolCall(tool_call_id='call-1', tool_name='noop', args='{}'), ResponseDone()])
    realtime = Agent(capabilities=[RecordToolCallSpanOutsideToolCalls(seen), MarkToolCall(), instrumentation]).realtime(
        FakeRealtimeModel(connection), toolsets=[FunctionToolset([noop])]
    )
    with TracerProvider().get_tracer('app').start_as_current_span('app request'):
        async with realtime.session() as session:
            _ = [event async for event in session]

    assert seen['before_run'] is INVALID_SPAN
    assert spans_with(exporter, 'example.marked') == ['execute_tool noop']


@dataclass
class RecordToolCallSpanAroundTheCall(AbstractCapability[object]):
    """Records what `tool_call_span` returns where `Instrumentation` doesn't have the call's span open."""

    seen: dict[str, object]

    def get_ordering(self) -> CapabilityOrdering:
        # `Instrumentation` is outermost too; listed first, this one's `wrap_tool_execute` wraps it.
        return CapabilityOrdering(position='outermost')

    async def before_tool_validate(
        self, ctx: RunContext[object], *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs
    ) -> RawToolArgs:
        self.seen['before_tool_validate'] = tool_call_span(ctx)
        return args

    async def wrap_tool_execute(
        self,
        ctx: RunContext[object],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        self.seen['wrap_tool_execute outside Instrumentation'] = tool_call_span(ctx)
        return await handler(args)


async def test_tool_call_span_is_invalid_where_the_calls_span_is_not_open(instrumented: Instrumented):
    """`before_tool_validate` and an outer `wrap_tool_execute` run before the span opens; a late task after it ends."""
    instrumentation, _ = instrumented(False)
    seen: dict[str, object] = {}
    tool_returned = asyncio.Event()
    late_tasks: list[asyncio.Task[None]] = []
    agent = Agent(TestModel(call_tools=['noop']), capabilities=[RecordToolCallSpanAroundTheCall(seen), instrumentation])

    async def after_the_call(ctx: RunContext[object]) -> None:
        await tool_returned.wait()
        seen['task that outlives the call'] = tool_call_span(ctx)

    @agent.tool
    async def noop(ctx: RunContext[object]) -> str:
        late_tasks.append(asyncio.create_task(after_the_call(ctx)))
        return 'ok'

    await agent.run('go')
    tool_returned.set()
    await asyncio.gather(*late_tasks)

    assert seen == {
        'before_tool_validate': INVALID_SPAN,
        'wrap_tool_execute outside Instrumentation': INVALID_SPAN,
        'task that outlives the call': INVALID_SPAN,
    }
