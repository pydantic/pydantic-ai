"""Tracing through the shared Monty executor, exercised via its public toolsets."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import anyio
import pytest
from anyio.lowlevel import checkpoint
from opentelemetry import baggage, context
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage
from pydantic_monty import FunctionSnapshot, instrument_telemetry

from pydantic_ai_harness.code_mode import CodeModeToolset
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflowToolset, WorkflowAgent


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(scope='session')
def monty_telemetry() -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrument_telemetry(tracer=provider.get_tracer('monty'))
    yield provider, exporter
    provider.shutdown()


@pytest.fixture
def telemetry(
    monty_telemetry: tuple[TracerProvider, InMemorySpanExporter],
) -> tuple[TracerProvider, InMemorySpanExporter]:
    provider, exporter = monty_telemetry
    exporter.clear()
    return provider, exporter


def _ctx(provider: TracerProvider) -> RunContext[object]:
    return RunContext[object](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        tracer=provider.get_tracer('harness-test'),
    )


def _assert_call_parent(span: ReadableSpan, spans: tuple[ReadableSpan, ...], name: str) -> None:
    assert span.parent is not None
    parent = next(s for s in spans if s.context == span.parent)
    supports_context = hasattr(FunctionSnapshot, 'trace_context')
    assert parent.name == ('call {function_name}' if supports_context else 'host')
    assert parent.attributes is not None
    assert parent.attributes.get('function_name') == (name if supports_context else None)


@pytest.mark.parametrize('mode', ['parallel', 'inline', 'global', 'barrier'])
@pytest.mark.parametrize('outcome', ['success', 'error', 'budget'])
async def test_code_mode_call_contexts(
    telemetry: tuple[TracerProvider, InMemorySpanExporter],
    mode: Literal['parallel', 'inline', 'global', 'barrier'],
    outcome: Literal['success', 'error', 'budget'],
) -> None:
    provider, exporter = telemetry
    ctx = _ctx(provider)

    async def first() -> str:
        before = context.get_current()
        assert baggage.get_baggage('request') == 'test-request'
        await checkpoint()
        assert context.get_current() is before
        if outcome == 'error':
            raise RuntimeError('tool failed')
        return 'first'

    async def second() -> str:
        assert baggage.get_baggage('request') == 'test-request'
        return 'second'

    wrapped = FunctionToolset[object](
        tools=[Tool(first, sequential=mode == 'inline'), Tool(second, sequential=mode in ('inline', 'barrier'))]
    )
    code = {
        'parallel': 'import asyncio\nawait asyncio.gather(first(), second())',
        'inline': '[first(), second()]',
        'global': 'import asyncio\nawait asyncio.gather(first(), second())',
        'barrier': 'pending = first()\nsecond()\nawait pending',
    }[mode]
    async with CodeModeToolset(wrapped=wrapped, max_tool_calls=1 if outcome == 'budget' else 10) as toolset:
        ctx.tool_manager = await ToolManager(
            toolset=toolset,
            root_capability=Instrumentation(settings=InstrumentationSettings(tracer_provider=provider)),
        ).for_run_step(ctx)
        tools = await toolset.get_tools(ctx)
        with (
            ctx.tracer.start_as_current_span('host'),
            ToolManager.parallel_execution_mode('sequential' if mode == 'global' else 'parallel'),
        ):
            token = context.attach(baggage.set_baggage('request', 'test-request'))
            try:
                before = context.get_current()
                if outcome == 'success':
                    await toolset.call_tool('run_code', {'code': code}, ctx, tools['run_code'])
                else:
                    match = 'tool failed' if outcome == 'error' else 'allows 1 nested tool calls'
                    with pytest.raises(ModelRetry, match=match):
                        await toolset.call_tool('run_code', {'code': code}, ctx, tools['run_code'])
                assert context.get_current() is before
            finally:
                context.detach(token)

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.attributes and s.attributes.get('gen_ai.tool.name')]
    if outcome == 'success':
        assert {s.attributes['gen_ai.tool.name'] for s in tool_spans if s.attributes} == {'first', 'second'}
    for span in tool_spans:
        assert span.attributes is not None
        _assert_call_parent(span, spans, str(span.attributes['gen_ai.tool.name']))


@pytest.mark.parametrize('mode', ['parallel', 'inline', 'global'])
async def test_cancelled_call_restores_context(
    telemetry: tuple[TracerProvider, InMemorySpanExporter], mode: Literal['parallel', 'inline', 'global']
) -> None:
    provider, exporter = telemetry
    ctx = _ctx(provider)
    started = anyio.Event()
    cleaned_up = anyio.Event()
    scope = anyio.CancelScope()

    async def wait() -> None:
        before = context.get_current()
        started.set()
        try:
            await anyio.sleep_forever()
        finally:
            assert context.get_current() is before
            cleaned_up.set()

    wrapped = FunctionToolset[object](tools=[Tool(wait, sequential=mode == 'inline')])
    async with CodeModeToolset(wrapped=wrapped) as toolset:
        ctx.tool_manager = await ToolManager(
            toolset=toolset,
            root_capability=Instrumentation(settings=InstrumentationSettings(tracer_provider=provider)),
        ).for_run_step(ctx)
        tools = await toolset.get_tools(ctx)

        async def runner() -> None:
            with (
                ctx.tracer.start_as_current_span('host'),
                ToolManager.parallel_execution_mode('sequential' if mode == 'global' else 'parallel'),
            ):
                before = context.get_current()
                with scope:
                    code = 'wait()' if mode == 'inline' else 'await wait()'
                    await toolset.call_tool('run_code', {'code': code}, ctx, tools['run_code'])
                assert scope.cancelled_caught
                assert context.get_current() is before
                assert cleaned_up.is_set()

        with anyio.fail_after(10):
            async with anyio.create_task_group() as group:
                group.start_soon(runner)
                await started.wait()
                scope.cancel()

    spans = exporter.get_finished_spans()
    span = next(s for s in spans if s.attributes and s.attributes.get('gen_ai.tool.name') == 'wait')
    _assert_call_parent(span, spans, 'wait')


async def test_workflow_agent_runs_are_children_of_monty_calls(
    telemetry: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, exporter = telemetry
    settings = InstrumentationSettings(tracer_provider=provider)
    agents = [
        WorkflowAgent(
            Agent(TestModel(custom_output_text=name), name=name, capabilities=[Instrumentation(settings=settings)])
        )
        for name in ('first', 'second')
    ]
    ctx = _ctx(provider)
    async with DynamicWorkflowToolset[object](agents=agents) as toolset:
        tools = await toolset.get_tools(ctx)
        with ctx.tracer.start_as_current_span('host'):
            before = context.get_current()
            await toolset.call_tool(
                'run_workflow',
                {'code': 'import asyncio\nawait asyncio.gather(first(task="one"), second(task="two"))'},
                ctx,
                tools['run_workflow'],
            )
            assert context.get_current() is before

    spans = exporter.get_finished_spans()
    agent_spans = [s for s in spans if s.attributes and s.attributes.get('agent_name')]
    assert len(agent_spans) == 2
    for span in agent_spans:
        assert span.attributes is not None
        _assert_call_parent(span, spans, str(span.attributes['agent_name']))
