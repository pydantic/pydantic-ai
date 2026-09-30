"""Public tool tracing across the Render JSON boundary."""

from __future__ import annotations

import inspect

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from render.workflows import TaskContext, Workflows

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.render import RenderWorkflows

from .conftest import RecordingTaskContext


@pytest.mark.parametrize('instrumentation', ['disabled', 'agent', 'global', 'model'])
async def test_child_tool_tracer_uses_worker_instrumentation(instrumentation: str) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    settings = InstrumentationSettings(tracer_provider=provider, include_content=False)
    model = TestModel(call_tools=['traced_tool'])
    runtime = RenderWorkflows[None](Workflows(), models={'plain': model})
    agent = Agent(
        InstrumentedModel(model, settings) if instrumentation == 'model' else model,
        name='traced-agent',
        deps_type=type(None),
        capabilities=[runtime],
    )
    if instrumentation == 'agent':
        agent.instrument = settings
    elif instrumentation == 'disabled':
        agent.instrument = False

    recording: list[bool] = []

    @agent.tool
    async def traced_tool(ctx: RunContext[None]) -> str:
        with ctx.tracer.start_as_current_span('tool.custom') as span:
            recording.append(span.is_recording())
            span.set_attribute('tool.check', 'worker')
        assert ctx.trace_include_content is False
        return 'traced'

    @runtime.task
    async def run_agent(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('trace')).output

    if instrumentation == 'global':
        Agent.instrument_all(settings)
    try:
        context = RecordingTaskContext()
        pending = run_agent.func(context)
        assert inspect.isawaitable(pending)
        assert 'traced' in await pending
        enabled = instrumentation != 'disabled'
        assert recording == [enabled]
        spans = [span for span in exporter.get_finished_spans() if span.name == 'tool.custom']
        assert len(spans) == int(enabled)
        if spans:
            assert spans[0].attributes is not None
            assert spans[0].attributes['tool.check'] == 'worker'
        assert 'traced-agent__function_toolset__<agent>.call_tool' in context.task_names
    finally:
        if instrumentation == 'global':
            Agent.instrument_all(False)
        provider.shutdown()
