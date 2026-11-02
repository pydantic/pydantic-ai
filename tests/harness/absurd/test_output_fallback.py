from __future__ import annotations

import pytest

from tests.conftest import try_import
from tests.durable_exec._output_fallback import make_agent, make_models

with try_import() as imports_successful:
    from absurd_sdk import AsyncAbsurd

    from pydantic_ai_harness.absurd import AbsurdDurability

    from ._task import reenter_running_task, running_task_context

pytestmark = pytest.mark.skipif(not imports_successful(), reason='absurd-sdk not installed')


async def test_replay_reconstructs_selection(absurd: AsyncAbsurd) -> None:
    executions: list[str] = []
    models = make_models(executions)
    async with running_task_context(absurd) as ctx:
        task_id = ctx.task_id
        agent = make_agent(models, AbsurdDurability(models=models), executions)
        assert (await agent.run('Answer.')).output == 'ready'
    async with reenter_running_task(absurd, task_id):
        recovered = make_agent(models, AbsurdDurability(models=models), executions)
        assert (await recovered.run('Answer.')).output == 'ready'
    assert executions == ['primary', 'lookup', 'primary', 'backup']
