"""`pydantic_ai._telemetry`: spans that follow the right run's tracer and content policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai._telemetry import open_span, record_decision, set_content_attribute
from pydantic_ai.capabilities import AbstractCapability, Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel

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
