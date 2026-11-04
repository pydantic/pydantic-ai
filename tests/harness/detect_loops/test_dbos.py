"""`DetectLoops` under `DBOSDurability`: recovering a workflow reaches the same decision over the journal."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.detect_loops import DetectLoops
from tests.conftest import detach_dbos_logging, try_import

from ._stuck_agent import NUDGE_METADATA, model_calls, nudges, read_file, stuck_model

with try_import() as imports_successful:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability

pytestmark = [
    pytest.mark.xdist_group(name='harness-dbos'),
    pytest.mark.skipif(not imports_successful(), reason='dbos not installed'),
]


class _Crash(BaseException):
    """Leave the first attempt pending, so resuming it replays the journaled model requests."""


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_detect_loops',
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


def test_recovery_reaches_the_same_decision(dbos: DBOS) -> None:
    agent = Agent(
        FunctionModel(stuck_model),
        name='detect_loops_dbos',
        tools=[read_file],
        capabilities=[DetectLoops(), DBOSDurability()],
    )
    attempts: list[list[ModelMessage]] = []

    @DBOS.workflow(name='detect_loops_dbos')
    def workflow() -> None:
        attempts.append(agent.run_sync('Summarize missing.txt').all_messages())
        if len(attempts) == 1:
            raise _Crash

    model_calls.clear()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id), pytest.raises(_Crash):
        workflow()

    DBOS.resume_workflow(workflow_id).get_result()

    first, recovered = attempts
    assert nudges(first) == [NUDGE_METADATA]
    assert nudges(recovered) == [NUDGE_METADATA]
    # Recovery read every model response from the journal; the hooks ran again over them.
    assert len(model_calls) == 4
