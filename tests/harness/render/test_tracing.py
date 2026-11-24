"""Public tool tracing across the Render JSON boundary."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from render.workflows import TaskContext, TaskDefinition, Workflows

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Instrumentation, LocalWorkspace
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.render import RenderWorkflows
from tests.harness.conftest import Instrumented

from .conftest import (
    P,
    R,
    RecordingTaskContext,
    TaskBoundary,
    json_round_trip,
    rewrite_tool_context,
    run_agent_in_task,
)


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


@pytest.mark.parametrize('instrumented', [False, True])
async def test_wire_tracer_flag_uses_only_worker_configuration(instrumented: bool) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    settings = InstrumentationSettings(tracer_provider=provider)
    model = TestModel()
    runtime = RenderWorkflows[None](Workflows(), models={'plain': model})
    agent = Agent(
        InstrumentedModel(model, settings) if instrumented else model,
        name='tracer-fallback',
        deps_type=type(None),
        capabilities=[runtime],
    )
    recording: list[bool] = []

    @agent.tool
    async def inspect_tracer(ctx: RunContext[None]) -> str:
        with ctx.tracer.start_as_current_span('fallback') as span:
            recording.append(span.is_recording())
        return 'checked'

    def rewrite(fields: dict[str, object]) -> None:
        fields['_model_id'] = 'unknown-to-worker'
        fields['tracer_enabled'] = True

    try:
        await run_agent_in_task(agent, runtime, TaskBoundary(tamper_request=rewrite_tool_context(rewrite)))
        assert recording == [instrumented]
    finally:
        provider.shutdown()


class WorkerTaskContext(RecordingTaskContext):
    """Run each child task in a fresh context, as a worker process does, so nothing leaks in from the workflow."""

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        self.task_names.append(task.name)

        async def run_task() -> R:
            result = task.func(self, *json_round_trip(args), **json_round_trip(kwargs))
            if inspect.isawaitable(result):
                return await result
            # Every task these tests run is async.
            return result  # pragma: no cover

        return json_round_trip(await asyncio.create_task(run_task(), context=contextvars.Context()))


async def test_child_workspace_calls_are_traced_on_the_worker(tmp_path: Path, instrumented: Instrumented) -> None:
    tracing, exporter = instrumented(False)
    model = TestModel(call_tools=['look'])
    runtime = RenderWorkflows[None](Workflows(), models={'plain': model})
    agent = Agent(model, name='workspace-agent', deps_type=type(None), capabilities=[LocalWorkspace(tmp_path), runtime])
    agent.instrument = tracing.settings

    @agent.tool
    async def look(ctx: RunContext[None]) -> bool:
        return await ctx.workspace.exists('notes.txt')

    @runtime.task
    async def run_agent(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('go')).output

    pending = run_agent.func(WorkerTaskContext())
    assert inspect.isawaitable(pending)
    await pending

    # Both units run on the worker: the run-start `ensure` and the tool. A worker starts with no span of the
    # workflow's, so each call is traced as a root of its own.
    assert [
        (span.name, (span.attributes or {}).get('pydantic_ai.workspace.provider'))
        for span in exporter.get_finished_spans()
        if span.name.startswith('workspace ')
    ] == [('workspace working_dir', 'local'), ('workspace exists', 'local')]


@pytest.mark.parametrize('worker_content', [False, True])
async def test_workspace_content_needs_the_workers_content_setting_too(tmp_path: Path, worker_content: bool) -> None:
    """A serialized context asking for content can't turn it on for a worker configured without it."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    model = TestModel(call_tools=['look'])
    runtime = RenderWorkflows[None](Workflows(), models={'plain': model})
    agent = Agent(model, name='content-agent', deps_type=type(None), capabilities=[LocalWorkspace(tmp_path), runtime])
    agent.instrument = InstrumentationSettings(tracer_provider=provider, include_content=worker_content)

    @agent.tool
    async def look(ctx: RunContext[None]) -> bool:
        return await ctx.workspace.exists('notes.txt')

    def rewrite(fields: dict[str, object]) -> None:
        fields['trace_include_content'] = True

    try:
        await run_agent_in_task(agent, runtime, TaskBoundary(tamper_request=rewrite_tool_context(rewrite)))
        [exists] = [span for span in exporter.get_finished_spans() if span.name == 'workspace exists']
        assert ('file.path' in (exists.attributes or {})) is worker_content
    finally:
        provider.shutdown()


@pytest.mark.parametrize('worker_content', [False, True])
@pytest.mark.parametrize('configured_on', ['model', 'capability'])
async def test_a_task_records_content_only_if_the_worker_allows_it(worker_content: bool, configured_on: str) -> None:
    """A run that allows content still records none in a task whose worker doesn't, since the task's spans go to it.

    An `Instrumentation` capability on the worker's agent wins over `agent.instrument`, as it does in the run.
    """
    provider = TracerProvider()
    settings = InstrumentationSettings(tracer_provider=provider, include_content=worker_content)
    model = TestModel()
    runtime = RenderWorkflows[None](Workflows(), models={'plain': model})
    if configured_on == 'model':
        agent = Agent(
            InstrumentedModel(model, settings), name='content-flag', deps_type=type(None), capabilities=[runtime]
        )
    else:
        agent = Agent(
            model, name='content-flag', deps_type=type(None), capabilities=[runtime, Instrumentation(settings=settings)]
        )
        agent.instrument = InstrumentationSettings(tracer_provider=provider, include_content=True)
    seen: list[bool] = []

    @agent.tool
    async def read_flag(ctx: RunContext[None]) -> str:
        seen.append(ctx.trace_include_content)
        return 'read'

    def allow_content(fields: dict[str, object]) -> None:
        fields['trace_include_content'] = True

    try:
        await run_agent_in_task(agent, runtime, TaskBoundary(tamper_request=rewrite_tool_context(allow_content)))
        assert seen == [worker_content]
    finally:
        provider.shutdown()
