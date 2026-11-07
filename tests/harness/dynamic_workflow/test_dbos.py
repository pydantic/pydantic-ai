"""`DynamicWorkflow` under `DBOSDurability`: gathered sub-agents run one at a time, so recovery replays each call its own steps."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Generator
from pathlib import Path

import pytest

from tests.conftest import detach_dbos_logging, try_import, without_admin_server
from tests.harness.dynamic_workflow._concurrent_sub_agents import EXPECTED, build_orchestrator

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


@pytest.fixture
def workflow(dbos: DBOS) -> Callable[[], Awaitable[str]]:
    # Built outside the event loop: registering a DBOS workflow reads its source.
    orchestrator = build_orchestrator(DBOSDurability)

    @DBOS.workflow(name='dynamic_workflow_dbos')
    async def run() -> str:
        return (await orchestrator.run('Summarize alpha and beta')).output

    return run


async def test_gathered_sub_agents_recover_their_own_results(
    dbos: DBOS, workflow: Callable[[], Awaitable[str]]
) -> None:
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        assert await workflow() == EXPECTED
    steps = await dbos.list_workflow_steps_async(workflow_id)

    # Re-execute the workflow function from its last step, the way recovery does.
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    assert await handle.get_result(polling_interval_sec=0.01) == EXPECTED
