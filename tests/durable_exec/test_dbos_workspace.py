"""Workspaces under `DBOSDurability`, against a real DBOS runtime (sqlite).

Not VCR tests: the behavior under test is where each workspace call runs (a DBOS step, or directly
inside one) and what a forked re-execution replays, which only the DBOS system database can show.
The shared scenarios live in `workspace_scenarios.py`; this module runs them in a DBOS workflow and
adds what only DBOS has.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from inline_snapshot import snapshot

from ..workspace_fakes import InMemoryProvider
from .workspace_scenarios import SCENARIOS, Check, ScenarioFailed, cases, scenario_agents

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('DBOS is not installed', allow_module_level=True)


pytestmark = pytest.mark.xdist_group(name='dbos')


@pytest.fixture(scope='module')
async def dbos(tmp_path_factory: pytest.TempPathFactory) -> AsyncGenerator[DBOS]:
    # An async module-scoped fixture keeps one event loop for the module: DBOS binds its async
    # machinery to the loop it is launched under, and a per-test loop would leave it on a closed one.
    dbos_sqlite_file = tmp_path_factory.mktemp('dbos') / 'dbostest.sqlite'
    dbos_config: DBOSConfig = {
        'name': 'pydantic_dbos_workspace_tests',
        'system_database_url': f'sqlite:///{dbos_sqlite_file}',
        'run_admin_server': False,
        'enable_otlp': False,
    }
    dbos = DBOS(config=dbos_config)
    DBOS.launch()
    try:
        yield dbos
    finally:
        DBOS.destroy()
        # DBOS leaves its log filter on every logger, and the filter's emit path imports
        # `dbos._context`, which is fatal inside the Temporal workflow sandbox when an xdist
        # worker later runs the Temporal suite. See the `dbos` fixture in `test_dbos.py`.
        from dbos import _logger as dbos_logger_module

        for logger in [logging.root, *(logging.getLogger(name) for name in logging.root.manager.loggerDict)]:
            for log_filter in [f for f in logger.filters if isinstance(f, dbos_logger_module.DBOSLogTransformer)]:
                logger.removeFilter(log_filter)


# Function tools run inline in the workflow, so only a step may reach the environment.
provider = InMemoryProvider(in_unit=lambda: DBOS.step_id is not None)
agents = scenario_agents(DBOSDurability, prefix='dbos_', provider=provider)


@DBOS.workflow()
async def scenario_workflow(name: str, arg: str | None) -> Any:
    workflow_id = DBOS.workflow_id
    assert workflow_id is not None
    return await SCENARIOS[name](agents, arg, workflow_id)


async def run_scenario(name: str, arg: str | None) -> Any:
    try:
        return await scenario_workflow(name, arg)
    except Exception as error:
        raise ScenarioFailed(type(error).__name__, str(error)) from error


@pytest.mark.parametrize('check', cases())
async def test_workspace_scenario(dbos: DBOS, check: Check) -> None:
    provider.reset()
    await check(run_scenario, agents)


async def test_dbos_workspace_calls_are_steps_and_a_fork_replays_them(dbos: DBOS) -> None:
    provider.reset()
    workflow_id = f'workspace-{uuid.uuid4()}'
    with SetWorkflowID(workflow_id):
        output = await scenario_workflow('fresh', None)
    assert provider.log == ['create:env-1']
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert [step['function_name'] for step in steps] == snapshot(
        [
            'dbos_fresh__capability__workspace.call',
            'dbos_fresh__capability__workspace.call',
            'dbos_fresh__model.request',
            'dbos_fresh__capability__workspace.call',
            'dbos_fresh__capability__workspace.call',
            'dbos_fresh__capability__workspace.call',
            'dbos_fresh__capability__workspace.call',
            'dbos_fresh__model.request',
            'dbos_fresh__capability__workspace.call',
            'dbos_fresh__capability__workspace.call',
        ]
    )

    # Re-execute the workflow function from its last step, the way recovery does: every earlier
    # step replays its recorded output, `ensure` included, so the rebuilt workspace attaches to the
    # environment the original run created instead of creating another.
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    assert await handle.get_result() == output
    assert provider.log == ['create:env-1', 'attach:env-1']
    assert list(provider.environments) == ['env-1']
