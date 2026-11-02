"""`DetectLoops` under `DBOSDurability`: recovering a workflow reaches the same decision over the journal."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.detect_loops import DetectLoops, LoopJudge
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


def test_a_loop_within_a_response_reaches_the_same_decision_without_the_judge(dbos: DBOS) -> None:
    calls: list[None] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(None)
        if 'Change your approach' in str(messages[-1]):
            return ModelResponse(parts=[TextPart('The config is fine.')])
        return ModelResponse(parts=[TextPart('I should check the config file again. ' * 40)])

    def judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('the judge must not be asked in workflow code')  # pragma: no cover

    agent = Agent(
        FunctionModel(model),
        name='detect_loops_dbos_in_response',
        capabilities=[DetectLoops(stream_detector=LoopJudge(model=FunctionModel(judge))), DBOSDurability()],
    )
    attempts: list[list[ModelMessage]] = []

    @DBOS.workflow(name='detect_loops_dbos_in_response')
    def workflow() -> None:
        attempts.append(agent.run_sync('Is the config fine?').all_messages())
        if len(attempts) == 1:
            raise _Crash

    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id), pytest.raises(_Crash):
        workflow()

    DBOS.resume_workflow(workflow_id).get_result()

    expected = [{**NUDGE_METADATA, 'pattern': 'repeated_stream_text', 'tool_names': []}]
    first, recovered = attempts
    assert nudges(first) == expected
    assert nudges(recovered) == expected
    # Recovery read both model responses from the journal.
    assert len(calls) == 2
