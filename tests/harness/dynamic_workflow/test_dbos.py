"""`DynamicWorkflow` under `DBOSDurability`: concurrent sub-agents replay their own recorded steps on recovery."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow
from tests.conftest import detach_dbos_logging, try_import, without_admin_server
from tests.harness.dynamic_workflow._concurrent_sub_agents import EXPECTED, noop, orchestrator_model, summarize_model

with try_import() as imports_successful:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability

pytestmark = [
    pytest.mark.xdist_group(name='harness-dbos'),
    pytest.mark.skipif(not imports_successful(), reason='dbos not installed'),
]

summarize_agent = Agent(
    FunctionModel(summarize_model),
    name='summarize',
    description='Summarize a text.',
    tools=[noop],
    capabilities=[DBOSDurability()],
)
orchestrator_agent = Agent(
    FunctionModel(orchestrator_model),
    name='dynamic_workflow_dbos_agent',
    capabilities=[DynamicWorkflow(agents=[summarize_agent]), DBOSDurability()],
)


@DBOS.workflow(name='dynamic_workflow_dbos')
async def run_orchestrator(prompt: str) -> str:
    return (await orchestrator_agent.run(prompt)).output


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'dynamic_workflow_dbos',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
    }
    instance = DBOS(config=without_admin_server(config))
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()
        detach_dbos_logging()


async def test_concurrent_sub_agents_recover_their_own_results(dbos: DBOS) -> None:
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        assert await run_orchestrator('Summarize alpha and beta') == EXPECTED
    steps = await dbos.list_workflow_steps_async(workflow_id)

    # Re-execute the workflow function from its last step, the way recovery does.
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    assert await handle.get_result(polling_interval_sec=0.01) == EXPECTED
