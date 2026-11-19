"""`RequireVerification` under `DBOSDurability`: recovering a workflow rebuilds the ledger over the journal."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.require_verification import RequireVerification
from tests.conftest import try_import

from ._editing_agent import attempts, editing_model, model_calls, tools

with try_import() as imports_successful:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability

pytestmark = [
    pytest.mark.xdist_group(name='harness-dbos'),
    pytest.mark.skipif(not imports_successful(), reason='dbos not installed'),
]


class _Crash(BaseException):
    """Leave the first attempt pending, so resuming it replays the journaled steps."""


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_require_verification',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
    }
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()


def test_recovery_redirects_the_same_way(dbos: DBOS) -> None:
    agent = Agent(
        FunctionModel(editing_model),
        name='require_verification_dbos',
        tools=tools,
        capabilities=[RequireVerification(), DBOSDurability()],
    )
    runs: list[list[ModelMessage]] = []

    @DBOS.workflow(name='require_verification_dbos')
    def workflow() -> None:
        runs.append(agent.run_sync('Fix the bug').all_messages())
        if len(runs) == 1:
            raise _Crash
        # Recovery runs on a DBOS executor thread, where `run_sync` sets a new event loop that nothing else closes.
        asyncio.get_event_loop().close()

    model_calls.clear()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id), pytest.raises(_Crash):
        workflow()

    DBOS.resume_workflow(workflow_id).get_result()

    first, recovered = runs
    assert attempts(first) == [1]
    assert attempts(recovered) == [1]
    # Recovery read every model response from the journal; the hooks ran again over them.
    assert len(model_calls) == 4
