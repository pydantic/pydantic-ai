"""`SubAgents` under `DBOSDurability`: recovery rebuilds created agents and background tasks, and reuses delegations."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.subagents import DynamicAgents, SubAgent, SubAgents, TaskControl
from tests.conftest import detach_dbos_logging, try_import
from tests.harness.subagents._background import WORKER_CALLS, EventKinds, background_parent, background_worker
from tests.harness.subagents._dynamic import POET_CALLS, POET_RESULT, creator

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
        'name': 'subagents_dynamic',
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


async def test_created_agent_survives_recovery(dbos: DBOS) -> None:
    agent = Agent(
        FunctionModel(creator),
        name='subagents_dynamic',
        deps_type=type(None),
        capabilities=[SubAgents[None](dynamic_agents=DynamicAgents()), DBOSDurability[None]()],
    )

    @DBOS.workflow(name='subagents_dynamic')
    async def workflow() -> str:
        return (await agent.run('Write a poem')).output

    POET_CALLS.clear()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        assert await workflow() == POET_RESULT
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert 'subagents_dynamic__capability__sub_agents.delegate' in {step['function_name'] for step in steps}

    # Re-execute the workflow function from its last step, the way recovery does.
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    assert await handle.get_result(polling_interval_sec=0.01) == POET_RESULT
    assert len(POET_CALLS) == 1, 'recovery ran the created agent again'


async def test_background_task_survives_recovery(dbos: DBOS) -> None:
    events = EventKinds()
    worker = Agent(FunctionModel(background_worker), name='worker')
    agent = Agent(
        background_parent(),
        name='subagents_background',
        deps_type=type(None),
        capabilities=[
            SubAgents[None](agents=[SubAgent(worker)], task_control=TaskControl()),
            events,
            DBOSDurability[None](),
        ],
    )

    @DBOS.workflow(name='subagents_background')
    async def workflow() -> list[str]:
        events.kinds.clear()
        return [(await agent.run('Delegate it')).output, *events.kinds]

    WORKER_CALLS.clear()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        output, *kinds = await workflow()
    assert 'from the worker' in output
    assert kinds == [
        'sub_agents.task_assigned',
        'sub_agents.task_started',
        'sub_agents.task_backgrounded',
        'sub_agents.task_completed',
    ]
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert 'subagents_background__capability__sub_agents.delegate' in {step['function_name'] for step in steps}

    # Re-execute the workflow function from its last step, the way recovery does.
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    recovered, *recovered_kinds = await handle.get_result(polling_interval_sec=0.01)
    assert 'from the worker' in recovered
    assert recovered_kinds == kinds
    assert WORKER_CALLS == ['worker'], 'recovery ran the background task again'
