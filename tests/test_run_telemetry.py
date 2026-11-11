"""`pydantic_ai._telemetry`: spans that follow the right run's tracer and content policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai._telemetry import current_span, open_span, record_decision, set_content_attribute
from pydantic_ai.capabilities import AbstractCapability, Instrumentation, ValidatedToolArgs
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from .conftest import try_import

with try_import() as otel_sdk_installed:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytestmark = pytest.mark.skipif(not otel_sdk_installed(), reason='opentelemetry-sdk not installed')


def instrumented(include_content: bool) -> tuple[Instrumentation, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return Instrumentation(
        settings=InstrumentationSettings(tracer_provider=provider, include_content=include_content)
    ), exporter


@dataclass
class TraceWhenPrepared(AbstractCapability[Any]):
    """Records a span and a decision from `for_run`, which runs before the run publishes its telemetry."""

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        with open_span(
            ctx, 'example prepare', attributes={'example.kind': 'setup'}, content={'example.input': 'a'}
        ) as span:
            set_content_attribute(ctx, span, 'example.output', 'b')
        record_decision(ctx, 'example decided', attributes={'example.kind': 'setup'}, content={'example.detail': 'c'})
        return self


def example_spans(exporter: InMemorySpanExporter) -> list[dict[str, Any]]:
    return [
        {'name': span.name, **(span.attributes or {})}
        for span in exporter.get_finished_spans()
        if span.name.startswith('example ')
    ]


async def test_spans_in_a_sub_agents_for_run_follow_the_sub_agent():
    """A sub-agent's `for_run` traces to the sub-agent's tracer with its content policy, not its parent's.

    `for_run` runs before the sub-agent publishes its own telemetry, so only its `ctx` can say which run
    it belongs to: the value published at that point is the parent's.
    """
    parent_instrumentation, parent_exporter = instrumented(include_content=False)
    child_instrumentation, child_exporter = instrumented(include_content=True)
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


async def test_content_is_left_out_without_include_content():
    instrumentation, exporter = instrumented(include_content=False)
    await Agent(TestModel(), capabilities=[TraceWhenPrepared(), instrumentation]).run('go')

    assert example_spans(exporter) == snapshot(
        [{'name': 'example prepare', 'example.kind': 'setup'}, {'name': 'example decided', 'example.kind': 'setup'}]
    )


@dataclass
class MarkToolCall(AbstractCapability[Any]):
    async def before_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        current_span(ctx).set_attribute('example.marked', True)
        return args


def agent_with_tool(*capabilities: AbstractCapability[Any]) -> Agent[None, str]:
    agent = Agent(TestModel(call_tools=['noop']), capabilities=[MarkToolCall(), *capabilities])

    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    return agent


async def test_current_span_in_a_tool_hook_is_the_calls_execute_tool_span():
    instrumentation, exporter = instrumented(include_content=False)
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


async def test_current_span_in_an_uninstrumented_sub_agent_leaves_the_parents_span_alone():
    """The parent's telemetry is published around the sub-agent, but the sub-agent's own run isn't traced."""
    instrumentation, exporter = instrumented(include_content=False)
    child = agent_with_tool()
    parent = Agent(TestModel(call_tools=['delegate']), capabilities=[instrumentation])

    @parent.tool_plain
    async def delegate() -> str:
        return (await child.run('go')).output

    await parent.run('go')

    assert [span.name for span in exporter.get_finished_spans() if (span.attributes or {}).get('example.marked')] == []
