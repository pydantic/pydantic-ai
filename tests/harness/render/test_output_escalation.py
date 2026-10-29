from __future__ import annotations

import inspect

from render.workflows import TaskContext, Workflows

from pydantic_ai_harness import RenderWorkflows
from tests.harness.escalate_on_output_retry._scenario import make_agent, make_models

from .conftest import RecordingTaskContext


async def test_task_dispatch_keeps_selection() -> None:
    executions: list[str] = []
    models = make_models(executions)
    runtime = RenderWorkflows[object](Workflows(), deps_type=object, models=models)
    agent = make_agent(models, runtime, executions)

    @runtime.task
    async def root(ctx: TaskContext) -> str:
        return (await agent.run('Answer.')).output

    context = RecordingTaskContext()
    pending = root.func(context)
    assert inspect.isawaitable(pending)
    assert await pending == 'ready'
    assert executions == ['primary', 'lookup', 'primary', 'backup']
    assert len(context.task_names) >= 4
