"""Memory and public tool tracing across the Render JSON boundary."""

from __future__ import annotations

import inspect

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from render.workflows import TaskContext, Workflows

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.memory import Memory, MemoryToolset
from pydantic_ai_harness.render import RenderWorkflows

from .conftest import RecordingTaskContext


def memory_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    returned = {
        part.tool_name: part.content
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    if 'read_memory' in returned:
        return ModelResponse(parts=[TextPart(str(returned['read_memory']))])
    if 'write_memory' in returned:
        return ModelResponse(parts=[ToolCallPart('read_memory', {'file': 'MEMORY.md'}, tool_call_id='read')])
    return ModelResponse(parts=[ToolCallPart('write_memory', {'content': 'remembered'}, tool_call_id='write')])


@pytest.mark.parametrize('inject_memory', [True, False])
async def test_constructor_memory_runs_with_remote_tools_and_snapshots(inject_memory: bool) -> None:
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent(
        FunctionModel(memory_model),
        name='memory-agent',
        deps_type=type(None),
        capabilities=[Memory(inject_memory=inject_memory), runtime],
    )

    @runtime.task
    async def run_agent(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('remember')).output

    context = RecordingTaskContext()
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)
    assert 'remembered' in await pending
    assert context.task_names.count('memory-agent__function_toolset__memory.call_tool') == 2
    assert sum(name.endswith('.load_snapshot') for name in context.task_names) == (3 if inject_memory else 0)


async def test_matching_memory_id_does_not_admit_a_runtime_toolset() -> None:
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent(
        TestModel(call_tools=[]),
        name='static-memory',
        deps_type=type(None),
        capabilities=[Memory(), runtime],
    )

    @runtime.task
    async def run_agent(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('read', toolsets=[MemoryToolset(Memory())])).output

    context = RecordingTaskContext()
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)
    with pytest.raises(UserError, match='cannot be added at runtime'):
        await pending
    assert context.task_names == []


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
