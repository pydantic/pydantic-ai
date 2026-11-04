"""`TrajectoryJudge` under `DBOSDurability`: the evaluation is recorded, not re-asked."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.trajectory_judge import TrajectoryJudge
from tests.conftest import detach_dbos_logging, try_import

from ._durable_agent import judge_calls, judge_respond, respond, steering, trajectory_judge, work

with try_import() as imports_successful:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability

pytestmark = [
    pytest.mark.xdist_group(name='harness-dbos'),
    pytest.mark.skipif(not imports_successful(), reason='dbos not installed'),
]


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_trajectory_judge',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
        'run_admin_server': False,
    }
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()
        detach_dbos_logging()


def test_a_judge_without_an_id_cannot_bind_to_a_durable_agent() -> None:
    with pytest.raises(UserError, match='needs an explicit `id`'):
        Agent(
            FunctionModel(respond),
            name='unnamed_trajectory_judge',
            deps_type=type(None),
            capabilities=[TrajectoryJudge(model=FunctionModel(judge_respond)), DBOSDurability[None]()],
        )


async def test_replay_reuses_the_recorded_evaluation(dbos: DBOS) -> None:
    agent = Agent(
        FunctionModel(respond),
        name='trajectory_judge_dbos',
        deps_type=type(None),
        tools=[work],
        capabilities=[trajectory_judge(FunctionModel(judge_respond)), DBOSDurability[None]()],
    )

    @DBOS.workflow(name='trajectory_judge_dbos')
    async def workflow() -> list[str]:
        return steering((await agent.run('Summarize notes.txt')).all_messages())

    judge_calls.clear()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        first = await workflow()
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert 'trajectory_judge_dbos__capability__judge.evaluate' in {step['function_name'] for step in steps}

    # Re-execute the workflow past its last recorded step, the way recovery does.
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps) + 1)
    assert await handle.get_result(polling_interval_sec=0.01) == first
    assert len(first) == 1
    assert len(judge_calls) == 2, 'the replay asked the judge again'
