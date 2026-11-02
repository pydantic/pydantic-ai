from __future__ import annotations

import pytest

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.models import ModelRequestContext
from tests.conftest import try_import
from tests.durable_exec._output_fallback import make_agent, make_models

with try_import() as imports_successful:
    from pydantic_ai_harness.pgtask import PGTaskDurability

    from ._task import PGTaskDatabase, run_in_task

pytestmark = pytest.mark.skipif(not imports_successful(), reason='pgtask not installed')


async def test_replay_reconstructs_selection(pgtask_db: PGTaskDatabase) -> None:
    executions: list[str] = []
    models = make_models(executions)

    async def run(attempt: int) -> None:
        agent = make_agent(models, PGTaskDurability(models=models), executions)
        assert (await agent.run('Answer.')).output == 'ready'

    await run_in_task(pgtask_db, run, attempts=2)
    assert executions == ['primary', 'lookup', 'primary', 'backup']


class TestPGTaskDurability:
    async def test_mid_run_retry_rebuilds_output_selection(self, pgtask_db: PGTaskDatabase) -> None:
        executions: list[str] = []
        models = make_models(executions)
        interrupted = False

        def interrupt(ctx: RunContext[object], request_context: ModelRequestContext) -> ModelRequestContext:
            nonlocal interrupted
            if request_context.model.model_name == 'backup' and not interrupted:
                interrupted = True
                executions.append('interrupted')
                raise RuntimeError('Worker stopped before the fallback request.')
            return request_context

        async def run(attempt: int) -> None:
            agent = make_agent(
                models,
                PGTaskDurability(models=models),
                executions,
                capabilities=[Hooks(before_model_request=interrupt)],
            )
            assert (await agent.run('Answer.')).output == 'ready'

        await run_in_task(pgtask_db, run, attempts=2)
        assert executions == ['primary', 'lookup', 'primary', 'interrupted', 'backup']
