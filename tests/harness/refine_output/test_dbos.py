"""`RefineOutput` under `DBOSDurability`: recovering a workflow counts the same rounds over the journal."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.refine_output import DEFAULT_REFINE_OUTPUT_INSTRUCTIONS, RefineOutput
from tests.conftest import detach_dbos_logging, try_import, without_admin_server

from ._drafting_agent import drafting_model, model_calls, reviews

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
        'name': 'durable_refine_output',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
    }
    instance = DBOS(config=without_admin_server(config))
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()
        detach_dbos_logging()


def test_recovery_counts_the_same_rounds(dbos: DBOS) -> None:
    agent = Agent(
        FunctionModel(drafting_model), name='refine_output_dbos', capabilities=[RefineOutput(), DBOSDurability()]
    )
    attempts: list[list[ModelMessage]] = []

    @DBOS.workflow(name='refine_output_dbos')
    def workflow() -> None:
        attempts.append(agent.run_sync('Explain the change').all_messages())
        if len(attempts) == 1:
            raise _Crash
        # Recovery runs on a DBOS executor thread, where `run_sync` sets a new event loop that nothing else closes.
        asyncio.get_event_loop().close()

    model_calls.clear()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id), pytest.raises(_Crash):
        workflow()

    DBOS.resume_workflow(workflow_id).get_result()

    first, recovered = attempts
    assert reviews(first) == [(DEFAULT_REFINE_OUTPUT_INSTRUCTIONS, 1)]
    assert reviews(recovered) == [(DEFAULT_REFINE_OUTPUT_INSTRUCTIONS, 1)]
    # Recovery read both model responses from the journal; the hook ran again over them.
    assert len(model_calls) == 2
