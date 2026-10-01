"""Shared workspace contracts through Render's JSON task boundary.

The in-memory provider checks that workspace operations execute inside a child task. It supplies
storage shared by this test process, so these cases do not establish hosted storage sharing.
"""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from pathlib import Path
from typing import ParamSpec, TypeVar

import pytest
from render import TaskContext, Workflows
from render.workflows import TaskDefinition

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import RenderWorkflows

from ...durable_exec.workspace_scenarios import SCENARIOS, Check, ScenarioFailed, cases, scenario_agents
from ...workspace_fakes import InMemoryProvider
from .conftest import RecordingTaskContext

P = ParamSpec('P')
R = TypeVar('R')


class WorkspaceTaskContext(RecordingTaskContext):
    """Expose whether the shared provider is being reached from a child task."""

    def __init__(self) -> None:
        super().__init__()
        self.in_child_task = ContextVar('render_workspace_child_task', default=False)

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        token = self.in_child_task.set(True)
        try:
            return await super().run(task, *args, **kwargs)
        finally:
            self.in_child_task.reset(token)


@pytest.mark.parametrize('check', cases())
async def test_workspace_scenario(check: Check, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    app = Workflows()
    entry = RenderWorkflows[None](app)
    context = WorkspaceTaskContext()
    provider = InMemoryProvider(in_unit=lambda: context.in_child_task.get() or not entry.in_durable_context)
    agents = scenario_agents(lambda: RenderWorkflows[None](app), prefix='render_', provider=provider)

    @entry.task
    async def scenario(ctx: TaskContext, name: str, arg: str | None) -> object:
        del ctx
        return await SCENARIOS[name](agents, arg, 'render-test-entry')

    async def run(name: str, arg: str | None) -> object:
        try:
            pending = scenario.func(context, name, arg)
            assert inspect.isawaitable(pending)
            return await pending
        except Exception as error:
            raise ScenarioFailed(type(error).__name__, str(error)) from error

    await check(run, agents)


async def test_worker_restores_workspace_validation_context(tmp_path: Path) -> None:
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
    pending = entry.func(context)
    assert inspect.isawaitable(pending)
    assert await pending == 'saved'
    assert 'workspace-agent__capability__workspace.call' in context.task_names
    assert 'workspace-agent__function_toolset__<agent>.call_tool' in context.task_names
    assert (tmp_path / 'note.txt').read_text() == 'saved'
