"""Workspace handles are rebuilt on the worker, including capability policy."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from render import TaskContext, Workflows

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import WorkspaceReadOnlyError
from pydantic_ai_harness import RenderWorkflows
from pydantic_ai_harness.memory import FileStore, Memory

from .conftest import RecordingTaskContext
from .test_memory_and_tracing import memory_model


@pytest.mark.parametrize('durable', [False, True])
async def test_workspace_calls_and_tool_calls_share_the_selected_workspace(tmp_path: Path, durable: bool) -> None:
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent[None, str](
        TestModel(call_tools=['write_note']),
        deps_type=type(None),
        name='workspace-agent',
        capabilities=[LocalWorkspace(tmp_path), runtime],
        validation_context=lambda ctx: {'workspace': ctx.workspace.ref},
    )

    @agent.tool
    async def write_note(ctx: RunContext[None]) -> str:
        assert ctx.validation_context == {'workspace': ctx.workspace.ref}
        await ctx.workspace.write_text('note.txt', 'saved')
        return await ctx.workspace.read_text('note.txt')

    @runtime.task
    async def entry(ctx: TaskContext) -> str:
        del ctx
        result = await agent.run('save a note')
        return await result.workspace.read_text('note.txt')

    context = RecordingTaskContext()
    if durable:
        pending = entry.func(context)
        assert inspect.isawaitable(pending)
        assert await pending == 'saved'
        assert 'workspace-agent__capability__workspace.call' in context.task_names
        assert 'workspace-agent__function_toolset__<agent>.call_tool' in context.task_names
    else:
        assert (await agent.run('save a note')).output
        assert context.task_names == []
    assert (tmp_path / 'note.txt').read_text() == 'saved'


async def test_worker_preserves_read_only_workspace_policy(tmp_path: Path) -> None:
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent[None, str](
        TestModel(call_tools=['write_note']),
        deps_type=type(None),
        name='read-only-agent',
        capabilities=[LocalWorkspace(tmp_path, read_only=True), runtime],
    )

    @agent.tool
    async def write_note(ctx: RunContext[None]) -> str:
        with pytest.raises(WorkspaceReadOnlyError):
            await ctx.workspace.write_text('blocked.txt', 'blocked')
        return 'refused'

    @runtime.task
    async def entry(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('try a write')).output

    pending = entry.func(RecordingTaskContext())
    assert inspect.isawaitable(pending)
    assert 'refused' in await pending
    assert not (tmp_path / 'blocked.txt').exists()


async def test_memory_file_store_uses_worker_workspace(tmp_path: Path) -> None:
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent[None, str](
        FunctionModel(memory_model),
        deps_type=type(None),
        name='workspace-memory',
        capabilities=[LocalWorkspace(tmp_path), Memory(store=FileStore('memory'), inject_memory=False), runtime],
    )

    @runtime.task
    async def entry(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('remember')).output

    pending = entry.func(RecordingTaskContext())
    assert inspect.isawaitable(pending)
    assert 'remembered' in await pending
    assert list(tmp_path.rglob('MEMORY.md'))
