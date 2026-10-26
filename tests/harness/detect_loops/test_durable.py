"""`DetectLoops` under `DBOSDurability` and `PrefectDurability`: detection runs in workflow and flow code."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, SystemPromptPart, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.detect_loops import DetectLoops
from tests.conftest import detach_dbos_logging, try_import

with try_import() as imports_successful:
    from dbos import DBOS, DBOSConfig, SetWorkflowID
    from prefect import flow
    from prefect.settings import PREFECT_LOCAL_STORAGE_PATH, temporary_settings

    from pydantic_ai.durable_exec.dbos import DBOSDurability
    from pydantic_ai.durable_exec.prefect import PrefectDurability

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='dbos or prefect not installed')]

_NUDGE = {
    'origin': 'pydantic-ai-harness.detect_loops.nudge.v1',
    'pattern': 'repeated_result',
    'tool_names': ['read_file'],
}

_model_calls = 0


def _model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    global _model_calls
    _model_calls += 1
    if 'Change your approach' in str(messages[-1]):
        return ModelResponse(parts=[TextPart('missing.txt does not exist.')])
    return ModelResponse(parts=[ToolCallPart('read_file', {'path': 'missing.txt'})])


def read_file(path: str) -> str:
    return 'Error: file not found'


def _nudges(agent_messages: list[ModelMessage]) -> list[object]:
    return [part.metadata for message in agent_messages for part in message.parts if isinstance(part, SystemPromptPart)]


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


@pytest.mark.xdist_group(name='harness-dbos')
async def test_dbos_workflow_reaches_the_same_decision_on_replay(dbos: DBOS) -> None:
    global _model_calls
    agent = Agent(
        FunctionModel(_model),
        name='detect_loops_dbos',
        tools=[read_file],
        capabilities=[DetectLoops(), DBOSDurability()],
    )

    @DBOS.workflow(name='detect_loops_dbos')
    async def workflow() -> list[ModelMessage]:
        return (await agent.run('Summarize missing.txt')).all_messages()

    _model_calls = 0
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        first = await workflow()
    with SetWorkflowID(workflow_id):
        replayed = await workflow()

    assert _nudges(first) == [_NUDGE]
    assert _nudges(replayed) == [_NUDGE]
    # The replay reads every model response from the journal.
    assert _model_calls == 4


@pytest.mark.xdist_group(name='prefect')
async def test_prefect_flow_delivers_the_nudge(prefect_test_server: None, tmp_path: Path) -> None:
    agent = Agent(
        FunctionModel(_model),
        name='detect_loops_prefect',
        tools=[read_file],
        capabilities=[DetectLoops(), PrefectDurability()],
    )

    @flow
    async def run() -> list[ModelMessage]:
        return (await agent.run('Summarize missing.txt')).all_messages()

    with temporary_settings({PREFECT_LOCAL_STORAGE_PATH: tmp_path}):
        messages = await run()

    assert _nudges(messages) == [_NUDGE]
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].text == 'missing.txt does not exist.'
