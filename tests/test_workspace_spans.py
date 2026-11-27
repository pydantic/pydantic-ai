"""Spans for workspace operations, and for the shell commands behind file operations on command-only backends.

These are unit tests, not VCR tests: what they check is span attributes and the `INVALID_SPAN` branches,
which no provider response can show, so the runs use `TestModel`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import anyio
import anyio.lowlevel
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace

from .conftest import Instrumented, try_import
from .workspace_fakes import RunOnlyWorkspaceBackend

with try_import() as otel_sdk_installed:
    # `opentelemetry-sdk` arrives with the `logfire` extra, so it is not importable in the slim install.
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytestmark = pytest.mark.skipif(not otel_sdk_installed(), reason='opentelemetry-sdk not installed')


def workspace_spans(exporter: InMemorySpanExporter, root: Path) -> list[dict[str, object]]:
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


def file_agent(capabilities: Sequence[AbstractCapability[object]] = ()) -> Agent[object, str]:
    agent = Agent(TestModel(call_tools=['use_files']), capabilities=capabilities)

    @agent.tool
    async def use_files(ctx: RunContext[object]) -> str:
        await ctx.workspace.write_text('notes.txt', 'hello')
        text = await ctx.workspace.read_text('notes.txt')
        assert not await ctx.workspace.exists('missing.txt')
        with pytest.raises(FileNotFoundError):
            await ctx.workspace.stat('missing.txt')
        result = await ctx.workspace.run(['sh', '-c', 'exit 3'])
        await ctx.workspace.run('exit 4', shell=True)
        return f'{text} {result.exit_code}'

    return agent


@pytest.mark.parametrize('include_content', [False, True])
async def test_operations_are_traced(tmp_path: Path, include_content: bool, instrumented: Instrumented) -> None:
    instrumentation, exporter = instrumented(include_content)
    await file_agent(capabilities=[LocalWorkspace(tmp_path), instrumentation]).run('go')

    expected = {True: snapshot([
        {'name': 'workspace write_bytes', 'status': 'UNSET', 'file.size': 5, 'file.path': '<root>/notes.txt'},
        {'name': 'workspace read_bytes', 'status': 'UNSET', 'file.path': '<root>/notes.txt', 'file.size': 5},
        {'name': 'workspace exists', 'status': 'UNSET', 'file.path': '<root>/missing.txt'},
        {'name': 'workspace stat', 'status': 'UNSET', 'file.path': '<root>/missing.txt'},
        {'name': 'workspace run', 'status': 'UNSET', 'process.command_args': ('sh', '-c', 'exit 3'), 'process.exit.code': 3},
        {'name': 'workspace run', 'status': 'UNSET', 'process.command_line': 'exit 4', 'process.exit.code': 4},
    ]), False: snapshot([
        {'name': 'workspace write_bytes', 'status': 'UNSET', 'file.size': 5},
        {'name': 'workspace read_bytes', 'status': 'UNSET', 'file.size': 5},
        {'name': 'workspace exists', 'status': 'UNSET'},
        {'name': 'workspace stat', 'status': 'UNSET'},
        {'name': 'workspace run', 'status': 'UNSET', 'process.exit.code': 3},
        {'name': 'workspace run', 'status': 'UNSET', 'process.exit.code': 4},
    ])}  # fmt: skip
    assert workspace_spans(exporter, tmp_path) == expected[include_content]
    # Every operation names the workspace's provider, and its id, a local path, only with content.
    assert {
        (
            (span.attributes or {}).get('pydantic_ai.workspace.provider'),
            (span.attributes or {}).get('pydantic_ai.workspace.id'),
        )
        for span in exporter.get_finished_spans()
        if span.name.startswith('workspace ')
    } == {('local', str(tmp_path) if include_content else None)}


async def test_shell_filesystem_commands_are_traced_without_their_text(
    tmp_path: Path, instrumented: Instrumented
) -> None:
    """A command-only backend's file operation shows each command it ran, but never the command, which carries the data.

    Even with content on: the command can hold the file's contents.
    """
    instrumentation, exporter = instrumented(True)
    agent = Agent(TestModel(call_tools=['read_large_file']), capabilities=[instrumentation])
    data = 'x' * 70_000

    @agent.tool
    async def read_large_file(ctx: RunContext[object]) -> int:
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


async def test_a_failed_run_is_an_error(tmp_path: Path, instrumented: Instrumented) -> None:
    """Only file lookups treat a missing file as an answer: a command that never started is a failure."""
    instrumentation, exporter = instrumented(False)
    agent = Agent(TestModel(call_tools=['fail']), capabilities=[LocalWorkspace(tmp_path), instrumentation])

    @agent.tool
    async def fail(ctx: RunContext[object]) -> str:
        with pytest.raises(FileNotFoundError):
            await ctx.workspace.run(['true'], cwd='missing')
        return 'failed'

    await agent.run('go')

    [span] = [span for span in exporter.get_finished_spans() if span.name == 'workspace run']
    assert span.status.status_code.name == 'ERROR'
    [event] = [event for event in span.events if event.name == 'exception']
    assert str((event.attributes or {})['exception.type']).endswith('FileNotFoundError')
    assert 'exception.message' not in (event.attributes or {})


async def test_nothing_is_traced_without_instrumentation_or_outside_the_run(
    tmp_path: Path, instrumented: Instrumented
) -> None:
    """A sub-agent without instrumentation emits nothing, even inside an instrumented run, and neither does the workspace afterwards."""
    instrumentation, exporter = instrumented(False)
    sub_agent = file_agent()
    agent = Agent(TestModel(call_tools=['delegate']), capabilities=[instrumentation])

    @agent.tool
    async def delegate(ctx: RunContext[object]) -> str:
        await sub_agent.run('go', workspace=ctx.workspace)
        assert await ctx.workspace.exists('notes.txt')
        return 'done'

    result = await agent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
    assert await result.workspace.exists('notes.txt')
    await Workspace(LocalWorkspaceBackend(tmp_path)).read_text('notes.txt')

    assert [span['name'] for span in workspace_spans(exporter, tmp_path)] == ['workspace exists']


async def test_concurrent_runs_sharing_a_workspace_trace_separately(tmp_path: Path, instrumented: Instrumented) -> None:
    """Two runs with their own providers, interleaving calls on one `Workspace`, each record only their own."""
    workspace = Workspace(LocalWorkspaceBackend(tmp_path))
    exporters: list[InMemorySpanExporter] = []

    def looping_agent() -> Agent[object, str]:
        instrumentation, exporter = instrumented(False)
        exporters.append(exporter)
        agent = Agent(TestModel(call_tools=['look']), capabilities=[instrumentation])

        @agent.tool
        async def look(ctx: RunContext[object]) -> str:
            for _ in range(3):
                await ctx.workspace.exists('notes.txt')
                # Yield, so the two runs' calls interleave.
                await anyio.lowlevel.checkpoint()
            return 'done'

        return agent

    first, second = looping_agent(), looping_agent()
    async with anyio.create_task_group() as tg:
        tg.start_soon(lambda: first.run('go', workspace=workspace))
        tg.start_soon(lambda: second.run('go', workspace=workspace))

    for exporter in exporters:
        assert [span['name'] for span in workspace_spans(exporter, tmp_path)] == ['workspace exists'] * 3
        assert len({span.context.trace_id for span in exporter.get_finished_spans() if span.context}) == 1
