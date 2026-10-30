import pytest

pytest.importorskip('pgtask')

from pydantic_ai_harness.pgtask import PGTaskDurability
from tests.durable_exec._output_fallback import make_agent, make_models

from ._task import PGTaskDatabase, run_in_task


async def test_replay_reconstructs_selection(pgtask_db: PGTaskDatabase) -> None:
    executions: list[str] = []
    models = make_models(executions)

    async def run(attempt: int) -> None:
        agent = make_agent(models, PGTaskDurability(models=models), executions)
        assert (await agent.run('Answer.')).output == 'ready'

    await run_in_task(pgtask_db, run, attempts=2)
    assert executions == ['primary', 'lookup', 'primary', 'backup']
