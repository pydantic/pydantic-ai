from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from uuid import uuid4

import pytest

from pydantic_ai.messages import ModelResponse
from tests.conftest import detach_dbos_logging, try_import
from tests.durable_exec._output_fallback import make_agent, make_models

with try_import() as imports_successful:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability

pytestmark: list[pytest.MarkDecorator] = [
    pytest.mark.skipif(not imports_successful(), reason='dbos not installed'),
    pytest.mark.xdist_group(name='dbos'),
]


class Crash(BaseException):
    pass


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS]:
    config: DBOSConfig = {
        'name': 'output_fallback',
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


def test_recovery_reconstructs_selection(dbos: DBOS) -> None:
    executions: list[str] = []
    models = make_models(executions)
    attempts: list[list[str | None]] = []

    @DBOS.step(name='fallback_lookup')
    def lookup() -> str:
        executions.append('lookup')
        return 'evidence'

    agent = make_agent(models, DBOSDurability(models=models), executions, lookup_tool=lookup)

    @DBOS.workflow(name='output_fallback')
    def workflow() -> None:
        result = agent.run_sync('Answer.')
        attempts.append([m.model_name for m in result.all_messages() if isinstance(m, ModelResponse)])
        if len(attempts) == 1:
            raise Crash

    workflow_id = str(uuid4())
    with SetWorkflowID(workflow_id), pytest.raises(Crash):
        workflow()
    DBOS.resume_workflow(workflow_id).get_result()
    assert attempts == [['primary', 'primary', 'backup']] * 2
    assert executions == ['primary', 'lookup', 'primary', 'backup']
