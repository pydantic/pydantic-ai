"""Spans for workspace operations, and for the shell commands behind file operations on command-only backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Instrumentation, LocalWorkspace
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace, WorkspaceTimeoutError

from .conftest import try_import
from .workspace_fakes import RunOnlyWorkspaceBackend

with try_import() as otel_sdk_installed:
    # `opentelemetry-sdk` arrives with the `logfire` extra, so it is not importable in the slim install.
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
        if span.parent is None:
            return 0
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
