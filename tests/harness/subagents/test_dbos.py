"""`SubAgents(dynamic_agents=...)` under `DBOSDurability`: recovery rebuilds the roster and reuses the delegation."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.subagents import DynamicAgents, SubAgents
from tests.conftest import detach_dbos_logging
from tests.harness.subagents._dynamic import POET_CALLS, POET_RESULT, creator

pytestmark = [pytest.mark.xdist_group(name='harness-dbos')]


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


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


_agent: Agent[None, str] = Agent(
    FunctionModel(creator),
    name='subagents_dynamic',
    deps_type=type(None),
    capabilities=[SubAgents[None](dynamic_agents=DynamicAgents()), DBOSDurability[None]()],
)


@DBOS.workflow(name='subagents_dynamic')
async def _workflow() -> str:
    return (await _agent.run('Write a poem')).output


@pytest.mark.anyio
async def test_created_agent_survives_recovery(dbos: DBOS) -> None:
    POET_CALLS.clear()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        assert await _workflow() == POET_RESULT
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert 'subagents_dynamic__capability__sub_agents.delegate' in {step['function_name'] for step in steps}

    # Re-execute the workflow function from its last step, the way recovery does.
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    assert await handle.get_result(polling_interval_sec=0.01) == POET_RESULT
    assert len(POET_CALLS) == 1, 'recovery ran the created agent again'
