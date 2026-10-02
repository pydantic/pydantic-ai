"""`TrajectoryJudge` under a real durable engine: the evaluation is a recorded step."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.trajectory_judge import TrajectoryJudge
from tests.conftest import detach_dbos_logging


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_trajectory_judge',
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


_judge_calls = 0


def _judge_respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    global _judge_calls
    _judge_calls += 1
    return ModelResponse(parts=[ToolCallPart('final_result_Steer', {'message': 'cite your sources'})])


def _steered(messages: list[ModelMessage]) -> bool:
    return any(
        isinstance(part, UserPromptPart) and isinstance(part.content, str) and part.content.startswith('Steering')
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    if _steered(messages):
        return ModelResponse(parts=[TextPart('steered')])
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('search', {})])
    return ModelResponse(parts=[TextPart('drifting')])


_agent: Agent[None, str] = Agent(
    FunctionModel(_respond),
    name='durable_trajectory_judge',
    deps_type=type(None),
    capabilities=[
        TrajectoryJudge[None](model=FunctionModel(_judge_respond), id='drift', every=2),
        DBOSDurability[None](),
    ],
)


@_agent.tool_plain
def search() -> str:
    return 'a rumour'


@DBOS.workflow(name='durable_trajectory_judge')
async def _workflow() -> str:
    return (await _agent.run('research the topic')).output


@pytest.mark.anyio
async def test_dbos_records_the_evaluation_as_a_step(dbos: DBOS) -> None:
    """The verdict crosses the real engine's boundary, and a replay does not ask the judge again."""
    global _judge_calls
    _judge_calls = 0
    workflow_id = str(uuid.uuid4())

    with SetWorkflowID(workflow_id):
        assert await _workflow() == 'steered'
    with SetWorkflowID(workflow_id):
        assert await _workflow() == 'steered'

    assert _judge_calls == 1, 'the replay asked the judging model again'
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert [step['function_name'] for step in steps] == [
        'durable_trajectory_judge__model.request',
        'durable_trajectory_judge__model.request',
        'durable_trajectory_judge__capability__drift.evaluate',
        'durable_trajectory_judge__model.request',
    ]
