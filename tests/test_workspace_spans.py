"""Spans for workspace operations, and for the shell commands behind file operations on command-only backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Instrumentation, LocalWorkspace
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace, WorkspaceRef, WorkspaceTimeoutError
from pydantic_ai.workspaces._telemetry import lifecycle_span

from .conftest import try_import
from .workspace_fakes import RunOnlyWorkspaceBackend

with try_import() as otel_sdk_installed:
    # `opentelemetry-sdk` arrives with the `logfire` extra, so it is not importable in the slim install.
    from logfire.testing import CaptureLogfire
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
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


def workspace_spans(exporter: InMemorySpanExporter, root: Path) -> list[dict[str, Any]]:
    """The workspace spans in start order, indented under the operation that ran them, with `root` written as `<root>`.

    The workspace's identity is left out: `test_operations_are_traced` checks it once.
    """
    spans = sorted(exporter.get_finished_spans(), key=lambda span: span.start_time or 0)
    by_id = {context.span_id: span for span in spans if (context := span.get_span_context()) is not None}

    def depth(span: ReadableSpan) -> int:
        # A workspace span always opens inside the run's own spans.
        assert span.parent is not None
        parent = by_id.get(span.parent.span_id)
        return depth(parent) + 1 if parent is not None and parent.name.startswith('workspace ') else 0

    return [
        {
            'name': '  ' * depth(span) + span.name,
            'status': span.status.status_code.name,
            **{
                key: str(value).replace(str(root.resolve()), '<root>').replace(str(root), '<root>')
                if isinstance(value, str)
                else value
                for key, value in (span.attributes or {}).items()
                if key not in ('pydantic_ai.workspace.provider', 'pydantic_ai.workspace.id')
            },
        }
        for span in spans
        if span.name.startswith('workspace ')
    ]


def file_agent(**kwargs: Any) -> Agent[Any, str]:
    agent = Agent(TestModel(call_tools=['use_files']), **kwargs)

    @agent.tool
    async def use_files(ctx: RunContext[Any]) -> str:
        await ctx.workspace.write_text('notes.txt', 'hello')
        text = await ctx.workspace.read_text('notes.txt')
        assert not await ctx.workspace.exists('missing.txt')
        with pytest.raises(FileNotFoundError):
            await ctx.workspace.stat('missing.txt')
        result = await ctx.workspace.run(['sh', '-c', 'exit 3'])
        return f'{text} {result.exit_code}'

    return agent


@pytest.mark.parametrize('include_content', [False, True])
async def test_operations_are_traced(tmp_path: Path, include_content: bool) -> None:
    instrumentation, exporter = instrumented(include_content)
    await file_agent(capabilities=[LocalWorkspace(tmp_path), instrumentation]).run('go')

    expected = {True: snapshot([
        {'name': 'workspace write_bytes', 'status': 'UNSET', 'file.size': 5, 'file.path': '<root>/notes.txt'},
        {'name': 'workspace read_bytes', 'status': 'UNSET', 'file.path': '<root>/notes.txt', 'file.size': 5},
        {'name': 'workspace exists', 'status': 'UNSET', 'file.path': '<root>/missing.txt'},
        {'name': 'workspace stat', 'status': 'UNSET', 'file.path': '<root>/missing.txt'},
        {'name': 'workspace run', 'status': 'UNSET', 'process.command_args': ('sh', '-c', 'exit 3'), 'process.exit.code': 3},
    ]), False: snapshot([
        {'name': 'workspace write_bytes', 'status': 'UNSET', 'file.size': 5},
        {'name': 'workspace read_bytes', 'status': 'UNSET', 'file.size': 5},
        {'name': 'workspace exists', 'status': 'UNSET'},
        {'name': 'workspace stat', 'status': 'UNSET'},
        {'name': 'workspace run', 'status': 'UNSET', 'process.exit.code': 3},
    ])}  # fmt: skip
    assert workspace_spans(exporter, tmp_path) == expected[include_content]
    # Every operation names the workspace, with or without content.
    assert {
        (span.attributes or {}).get('pydantic_ai.workspace.id')
        for span in exporter.get_finished_spans()
        if span.name.startswith('workspace ')
    } == {str(tmp_path)}


@pytest.mark.parametrize('include_content', [False, True])
async def test_shell_filesystem_commands_are_traced_without_their_text(tmp_path: Path, include_content: bool) -> None:
    """A command-only backend's file operation shows each command it ran, but never the command, which carries the data."""
    instrumentation, exporter = instrumented(include_content)
    agent = Agent(TestModel(call_tools=['read_large_file']), capabilities=[instrumentation])
    data = 'x' * 70_000

    @agent.tool
    async def read_large_file(ctx: RunContext[Any]) -> int:
        await ctx.workspace.write_text('large.txt', data)
        return len(await ctx.workspace.read_text('large.txt'))

    await agent.run('go', workspace=RunOnlyWorkspaceBackend(LocalWorkspaceBackend(tmp_path)))

    spans = workspace_spans(exporter, tmp_path)
    assert [{k: v for k, v in span.items() if k in ('name', 'pydantic_ai.workspace.command.step')} for span in spans] == snapshot([
        {'name': 'workspace write_bytes'},
        {'name': '  workspace run', 'pydantic_ai.workspace.command.step': 'realpath'},
        {'name': '  workspace run', 'pydantic_ai.workspace.command.step': 'write_chunk'},
        {'name': '  workspace run', 'pydantic_ai.workspace.command.step': 'write_chunk'},
        {'name': '  workspace run', 'pydantic_ai.workspace.command.step': 'commit'},
        {'name': 'workspace read_bytes'},
        {'name': '  workspace run', 'pydantic_ai.workspace.command.step': 'stat'},
        {'name': '  workspace run', 'pydantic_ai.workspace.command.step': 'read_chunk'},
        {'name': '  workspace run', 'pydantic_ai.workspace.command.step': 'read_chunk'},
    ])  # fmt: skip
    assert not any(key.startswith('process.command') for span in spans for key in span)
    assert all(data[:100] not in str(span.attributes) for span in exporter.get_finished_spans())


async def test_a_timeout_is_an_error(tmp_path: Path) -> None:
    instrumentation, exporter = instrumented(include_content=False)
    agent = Agent(TestModel(call_tools=['wait']), capabilities=[LocalWorkspace(tmp_path), instrumentation])

    @agent.tool
    async def wait(ctx: RunContext[Any]) -> str:
        with pytest.raises(WorkspaceTimeoutError):
            await ctx.workspace.run(['sleep', '5'], timeout=0.01)
        return 'timed out'

    await agent.run('go')

    assert [span['name'] for span in workspace_spans(exporter, tmp_path)] == snapshot(
        ['workspace run', '  workspace stop']
    )
    [span] = [span for span in exporter.get_finished_spans() if span.name == 'workspace run']
    assert span.status.status_code.name == 'ERROR'
    assert [event.attributes for event in span.events] == snapshot(
        [{'exception.type': 'pydantic_ai.workspaces.protocol.WorkspaceTimeoutError', 'exception.escaped': 'False'}]
    )


async def test_nothing_is_traced_without_instrumentation_or_outside_the_run(tmp_path: Path) -> None:
    """A sub-agent without instrumentation emits nothing, even inside an instrumented run, and neither does the workspace afterwards."""
    instrumentation, exporter = instrumented(include_content=False)
    sub_agent = file_agent()
    agent = Agent(TestModel(call_tools=['delegate']), capabilities=[instrumentation])

    @agent.tool
    async def delegate(ctx: RunContext[Any]) -> str:
        await sub_agent.run('go', workspace=ctx.workspace)
        assert await ctx.workspace.exists('notes.txt')
        return 'done'

    result = await agent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
    assert await result.workspace.exists('notes.txt')
    await Workspace(LocalWorkspaceBackend(tmp_path)).read_text('notes.txt')

    assert [span['name'] for span in workspace_spans(exporter, tmp_path)] == ['workspace exists']


async def test_an_instrumented_sub_agent_traces_with_its_own_settings(tmp_path: Path) -> None:
    """A sub-agent given the parent's workspace traces to its own provider with its own content setting."""
    parent_instrumentation, parent_exporter = instrumented(include_content=True)
    sub_instrumentation, sub_exporter = instrumented(include_content=False)
    sub_agent = file_agent(capabilities=[sub_instrumentation])
    agent = Agent(TestModel(call_tools=['delegate']), capabilities=[parent_instrumentation])

    @agent.tool
    async def delegate(ctx: RunContext[Any]) -> str:
        await sub_agent.run('go', workspace=ctx.workspace)
        assert await ctx.workspace.exists('notes.txt')
        return 'done'

    await agent.run('go', workspace=LocalWorkspaceBackend(tmp_path))

    assert workspace_spans(parent_exporter, tmp_path) == snapshot(
        [{'name': 'workspace exists', 'status': 'UNSET', 'file.path': '<root>/notes.txt'}]
    )
    assert [span['name'] for span in workspace_spans(sub_exporter, tmp_path)] == snapshot(
        ['workspace write_bytes', 'workspace read_bytes', 'workspace exists', 'workspace stat', 'workspace run']
    )
    assert not any('file.path' in span for span in workspace_spans(sub_exporter, tmp_path))


@pytest.mark.parametrize('shared', ['facade', 'backend'])
async def test_concurrent_runs_sharing_a_workspace_trace_separately(tmp_path: Path, shared: str) -> None:
    """Two runs with their own providers, interleaving calls on one workspace, each record only their own."""
    backend = LocalWorkspaceBackend(tmp_path)
    workspace = Workspace(backend) if shared == 'facade' else backend
    exporters: list[InMemorySpanExporter] = []

    def looping_agent() -> Agent[Any, str]:
        instrumentation, exporter = instrumented(include_content=False)
        exporters.append(exporter)
        agent = Agent(TestModel(call_tools=['look']), capabilities=[instrumentation])

        @agent.tool
        async def look(ctx: RunContext[Any]) -> str:
            for _ in range(3):
                await ctx.workspace.exists('notes.txt')
                # Yield, so the two runs' calls interleave.
                await anyio.sleep(0)
            return 'done'

        return agent

    first, second = looping_agent(), looping_agent()
    async with anyio.create_task_group() as tg:
        tg.start_soon(lambda: first.run('go', workspace=workspace))
        tg.start_soon(lambda: second.run('go', workspace=workspace))

    for exporter in exporters:
        assert [span['name'] for span in workspace_spans(exporter, tmp_path)] == ['workspace exists'] * 3
        assert len({span.context.trace_id for span in exporter.get_finished_spans() if span.context}) == 1


async def test_calls_from_a_task_the_tool_starts_are_traced(tmp_path: Path) -> None:
    instrumentation, exporter = instrumented(include_content=False)
    agent = Agent(TestModel(call_tools=['fan_out']), capabilities=[LocalWorkspace(tmp_path), instrumentation])

    @agent.tool
    async def fan_out(ctx: RunContext[Any]) -> str:
        async with anyio.create_task_group() as tg:
            tg.start_soon(ctx.workspace.exists, 'a.txt')
            tg.start_soon(ctx.workspace.exists, 'b.txt')
        return 'done'

    await agent.run('go')

    assert [span['name'] for span in workspace_spans(exporter, tmp_path)] == ['workspace exists'] * 2


class LazySandbox(RunOnlyWorkspaceBackend):
    """A command-only backend that creates its sandbox on first use, as the harness provider backends do."""

    sandbox: WorkspaceRef | None = None

    @property
    def ref(self) -> WorkspaceRef | None:
        return self.sandbox

    async def working_dir(self) -> str:
        if self.sandbox is None:
            with lifecycle_span('create', self):
                self.sandbox = WorkspaceRef(provider='lazy', id='box-1')
        return await super().working_dir()


async def test_a_sandbox_created_on_first_use_shows_inside_that_operation(tmp_path: Path) -> None:
    """The first call creates the sandbox under its own span; the next call reuses it."""
    instrumentation, exporter = instrumented(include_content=False)
    agent = Agent(TestModel(call_tools=['look']), capabilities=[instrumentation])

    @agent.tool
    async def look(ctx: RunContext[Any]) -> bool:
        return await ctx.workspace.exists('notes.txt') or await ctx.workspace.exists('other.txt')

    await agent.run('go', workspace=LazySandbox(LocalWorkspaceBackend(tmp_path)))

    assert [span['name'] for span in workspace_spans(exporter, tmp_path)] == snapshot(
        ['workspace exists', '  workspace create', '  workspace run', 'workspace exists', '  workspace run']
    )
    [create] = [span for span in exporter.get_finished_spans() if span.name == 'workspace create']
    assert create.attributes == snapshot(
        {'pydantic_ai.workspace.provider': 'lazy', 'pydantic_ai.workspace.id': 'box-1'}
    )


async def test_a_run_without_instrumentation_traces_nothing_even_under_instrument_all(
    tmp_path: Path, capfire: CaptureLogfire
) -> None:
    """Inside a run, the run's own setting decides, also for the sandbox it creates."""
    Agent.instrument_all()
    agent = Agent(TestModel(call_tools=['look']))
    agent.instrument = False

    @agent.tool
    async def look(ctx: RunContext[Any]) -> bool:
        return await ctx.workspace.exists('notes.txt')

    await agent.run('go', workspace=LazySandbox(LocalWorkspaceBackend(tmp_path)))

    assert capfire.exporter.exported_spans_as_dict() == []
