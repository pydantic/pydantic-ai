"""`ToolCallJudge` under a real durable engine: the verdict is recorded, not re-asked."""

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
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.tool_call_judge import ToolCallJudge, ToolCallVerdict
from tests.conftest import detach_dbos_logging


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_tool_call_judge',
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
_verdicts: list[ToolCallVerdict] = []


def _judge_respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    global _judge_calls
    _judge_calls += 1
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'response': 'no'})])


def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    returned = any(
        isinstance(part, ToolReturnPart)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
    )
    if returned:
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('issue_refund', {'amount': 10}, tool_call_id='call-1')])


_judge: ToolCallJudge[None] = ToolCallJudge(
    FunctionModel(_judge_respond),
    id='refund-judge',
    tools=['issue_refund'],
    question='Would this refund more than the original charge?',
    on_verdict=_verdicts.append,
)
_agent: Agent[None, str] = Agent(
    FunctionModel(_respond),
    name='durable_judge',
    deps_type=type(None),
    capabilities=[_judge, DBOSDurability[None]()],
)


@_agent.tool_plain
def issue_refund(amount: int) -> str:
    return f'refunded {amount}'


@DBOS.workflow(name='durable_judge')
async def _workflow() -> str:
    return (await _agent.run('refund the customer')).output


def test_the_judge_has_no_default_id() -> None:
    """A default would merge two judges on one agent, so durability needs an explicit one."""
    assert ToolCallJudge(FunctionModel(_judge_respond), question='Would this cause harm?').id is None
    assert _judge.id == 'refund-judge'


def test_a_judge_without_an_id_cannot_bind_to_a_durable_agent() -> None:
    with pytest.raises(UserError, match='needs an explicit `id`'):
        Agent(
            FunctionModel(_respond),
            name='unnamed_judge',
            deps_type=type(None),
            capabilities=[
                ToolCallJudge[None](FunctionModel(_judge_respond), question='Would this cause harm?'),
                DBOSDurability[None](),
            ],
        )


@pytest.mark.anyio
async def test_dbos_replays_the_recorded_verdict(dbos: DBOS) -> None:
    """A replay must reuse the recorded verdict: no second judge request, same decision."""
    global _judge_calls
    _judge_calls = 0
    _verdicts.clear()
    workflow_id = str(uuid.uuid4())

    with SetWorkflowID(workflow_id):
        assert await _workflow() == 'done'
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert 'durable_judge__capability__refund-judge.judge' in {step['function_name'] for step in steps}

    # Re-execute the workflow function from its last step, the way recovery does. Calling it again
    # under the same ID is no replay: from `dbos` 2.28 it returns the stored result without running it.
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    assert await handle.get_result() == 'done'

    assert _judge_calls == 1, 'the replay asked the judging model again'
    assert [v.verdict for v in _verdicts] == ['allow', 'allow'], '`on_verdict` fires on every replay'
