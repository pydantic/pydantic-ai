"""`ToolBudgets` under `DBOSDurability`: recovering a workflow counts the journaled calls once."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness import ToolBudgets
from tests.conftest import try_import

from ._search_agent import refusal, refusals, search, searches, searching_model

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
        'name': 'durable_tool_budgets',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
    }
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()


def test_recovery_does_not_double_count(dbos: DBOS) -> None:
    agent = Agent(
        FunctionModel(searching_model),
        name='tool_budgets_dbos',
        tools=[search],
        capabilities=[ToolBudgets(per_tool={'search': 2}), DBOSDurability()],
    )
    attempts: list[list[ModelMessage]] = []

    @DBOS.workflow(name='tool_budgets_dbos')
    def workflow() -> None:
        attempts.append(agent.run_sync('Research it').all_messages())
        if len(attempts) == 1:
            raise _Crash
        # Recovery runs on a DBOS executor thread, where `run_sync` sets a new event loop that nothing else closes.
        asyncio.get_event_loop().close()

    searches.clear()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id), pytest.raises(_Crash):
        workflow()

    DBOS.resume_workflow(workflow_id).get_result()

    first, recovered = attempts
    assert refusals(first) == [refusal('per_tool')]
    assert refusals(recovered) == [refusal('per_tool')]
    # Recovery replays the journaled model responses and runs the plain function tool again, which DBOS doesn't
    # journal. Each attempt counts from zero, so the replay lets the same two searches through and refuses the third.
    assert searches == ['q0', 'q1', 'q0', 'q1']
