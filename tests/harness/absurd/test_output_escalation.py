import pytest

pytest.importorskip('absurd_sdk')

from absurd_sdk import AsyncAbsurd

from pydantic_ai_harness.absurd import AbsurdDurability
from tests.harness.escalate_on_output_retry._scenario import make_agent, make_models

from ._task import reenter_running_task, running_task_context


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
