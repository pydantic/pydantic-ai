"""`DynamicWorkflow` under `DBOSDurability`: gathered sub-agents run one at a time, so recovery replays each its own steps."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Generator
from pathlib import Path

import pytest

from tests.conftest import try_import
from tests.harness.dynamic_workflow._concurrent_sub_agents import EXPECTED, build_orchestrator, summarize_model

with try_import() as imports_successful:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability

pytestmark = [
    pytest.mark.xdist_group(name='harness-dbos'),
    pytest.mark.skipif(not imports_successful(), reason='dbos not installed'),
]


class _ProcessCrash(BaseException):
    """Stands in for the process dying: DBOS leaves the workflow `PENDING` for recovery."""


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'dynamic_workflow_dbos',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
    }
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()


@pytest.fixture
def model_calls() -> list[str]:
    return []


@pytest.fixture
def tool_calls() -> list[str]:
    return []


@pytest.fixture
def workflow(dbos: DBOS, model_calls: list[str], tool_calls: list[str]) -> Callable[[], Awaitable[str]]:
    @DBOS.step(name='noop_step')
    def noop_step() -> str:
        tool_calls.append('noop')
        return 'ok'

    # Built outside the event loop: registering a DBOS workflow reads its source.
    orchestrator = build_orchestrator(
        DBOSDurability, summarize_model(model_calls, alpha_waits_for_beta=False), tool=noop_step
    )
    attempts: list[str] = []

    @DBOS.workflow(name='dynamic_workflow_dbos')
    async def run() -> str:
        output = (await orchestrator.run('Summarize alpha and beta')).output
        attempts.append(output)
        if len(attempts) == 1:
            raise _ProcessCrash
        return output

    return run


async def test_sub_agents_run_one_at_a_time_and_recover_their_own_steps(
    dbos: DBOS, workflow: Callable[[], Awaitable[str]], model_calls: list[str], tool_calls: list[str]
) -> None:
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id), pytest.raises(_ProcessCrash):
        await workflow()
    assert model_calls == ['alpha 1', 'alpha 3', 'beta 1', 'beta 3']
    # Each sub-agent keeps its own model-request and `@DBOS.step` tool checkpoints.
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert [step['function_name'] for step in steps] == [
        'orchestrator__model.request',
        *['summarize__model.request', 'noop_step', 'summarize__model.request'] * 2,
        'orchestrator__model.request',
    ]

    # Resuming the pending workflow re-executes it the way recovery does: each sub-agent replays its own steps.
    handle = await DBOS.resume_workflow_async(workflow_id)
    assert await handle.get_result(polling_interval_sec=0.01) == EXPECTED
    assert len(model_calls) == 4
    assert tool_calls == ['noop', 'noop']


async def test_sub_agents_outside_a_workflow_run_concurrently(dbos: DBOS) -> None:
    calls: list[str] = []
    orchestrator = build_orchestrator(DBOSDurability, summarize_model(calls))
    assert (await orchestrator.run('Summarize alpha and beta')).output == EXPECTED
    # Alpha, dispatched first, finished last.
    assert sorted(calls) == ['alpha 1', 'alpha 3', 'beta 1', 'beta 3']
    assert calls[-1] == 'alpha 3'
