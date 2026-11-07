"""`SubAgents` under `DBOSDurability`: dynamic agents survive recovery, and a delegate tool runs as a child workflow."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.subagents import DynamicAgents, SubAgent, SubAgents
from tests.conftest import detach_dbos_logging, try_import, without_admin_server
from tests.harness.subagents._dynamic import POET_CALLS, POET_RESULT, creator

with try_import() as imports_successful:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability

pytestmark = [
    pytest.mark.xdist_group(name='harness-dbos'),
    pytest.mark.skipif(not imports_successful(), reason='dbos not installed'),
]


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'subagents_dynamic',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
    }
    instance = DBOS(config=without_admin_server(config))
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()
        detach_dbos_logging()


async def test_created_agent_survives_recovery(dbos: DBOS) -> None:
    agent = Agent(
        FunctionModel(creator),
        name='subagents_dynamic',
        deps_type=type(None),
        capabilities=[SubAgents[None](dynamic_agents=DynamicAgents()), DBOSDurability[None]()],
    )

    @DBOS.workflow(name='subagents_dynamic')
    async def workflow() -> str:
        return (await agent.run('Write a poem')).output

    POET_CALLS.clear()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        assert await workflow() == POET_RESULT
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert 'subagents_dynamic__capability__sub_agents.delegate' in {step['function_name'] for step in steps}

    # Re-execute the workflow function from its last step, the way recovery does.
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    assert await handle.get_result(polling_interval_sec=0.01) == POET_RESULT
    assert len(POET_CALLS) == 1, 'recovery ran the created agent again'


def _worker(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('from the worker')])


def _delegator(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    if returns:
        return ModelResponse(parts=[TextPart(returns[-1].model_response_str())])
    return ModelResponse(parts=[ToolCallPart('delegate_task', {'agent_name': 'dbos_worker', 'task': 'Work on it'})])


if imports_successful():
    dbos_worker = Agent(FunctionModel(_worker), name='dbos_worker', capabilities=[DBOSDurability()])
    dbos_delegator = Agent(
        FunctionModel(_delegator),
        name='dbos_delegator',
        deps_type=type(None),
        capabilities=[
            SubAgents[None](agents=[SubAgent(dbos_worker)]),
            SetToolMetadata(tools=['delegate_task'], dbos={'child_workflow': {}}),
            DBOSDurability[None](),
        ],
    )

    @DBOS.workflow(name='subagents_child_workflow')
    async def child_workflow_workflow() -> str:
        return (await dbos_delegator.run('Delegate it')).output


async def test_durable_delegate_runs_as_a_child_workflow(dbos: DBOS) -> None:
    """With `child_workflow` metadata on the delegate tool, a durable delegate's steps land in the child's history."""
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        assert await child_workflow_workflow() == 'from the worker'
    steps = await dbos.list_workflow_steps_async(workflow_id)
    child_name = 'dbos_delegator__function_toolset__sub_agents.call_tool__child_workflow'
    assert [step['function_name'] for step in steps] == [
        'dbos_delegator__model.request',
        child_name,
        'DBOS.getResult',
        'dbos_delegator__model.request',
    ]
    child_id = steps[1]['child_workflow_id']
    assert child_id is not None
    child_steps = await dbos.list_workflow_steps_async(child_id)
    assert [step['function_name'] for step in child_steps] == ['dbos_worker__model.request']


async def test_child_workflow_rejects_options_that_need_the_parent_run(dbos: DBOS) -> None:
    agent = Agent(
        FunctionModel(_delegator),
        name='dbos_budgeted_delegator',
        deps_type=type(None),
        capabilities=[
            SubAgents[None](agents=[SubAgent(dbos_worker, max_calls=1)]),
            SetToolMetadata(tools=['delegate_task'], dbos={'child_workflow': {}}),
            DBOSDurability[None](),
        ],
    )

    @DBOS.workflow(name='subagents_budgeted_child_workflow')
    async def workflow() -> str:
        return (await agent.run('Delegate it')).output

    with SetWorkflowID(str(uuid.uuid4())), pytest.raises(UserError, match='does not support'):
        await workflow()
