"""`SubAgents(dynamic_agents=...)` under `DBOSDurability`: recovery rebuilds the roster and reuses the delegation."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterable, Generator
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Agent, AgentStreamEvent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.subagents import DynamicAgents, SubAgent, SubAgents, SubAgentToolset
from tests.conftest import detach_dbos_logging, try_import
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
        'run_admin_server': False,
    }
    instance = DBOS(config=config)
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
        capabilities=[SubAgents[None](agents=[SubAgent(dbos_worker)]), DBOSDurability[None]()],
    )

    @DBOS.workflow(name='subagents_child_workflow')
    async def child_workflow_workflow() -> str:
        return (await dbos_delegator.run('Delegate it')).output


async def test_durable_delegate_runs_as_a_child_workflow(dbos: DBOS) -> None:
    """A delegate with its own `DBOSDurability` runs in a child workflow, whose history holds its steps."""
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


async def _handle_events(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
    pass  # pragma: no cover


def _delegate_metadata(capability: SubAgents[None]) -> object:
    toolset = capability.get_toolset()
    assert isinstance(toolset, SubAgentToolset)
    return toolset.tools['delegate_task'].metadata


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({'include_self': True}, id='include_self'),
        pytest.param({'models': {'fast': FunctionModel(_worker)}}, id='models'),
        pytest.param({'shared_capabilities': [Hooks()]}, id='shared_capabilities'),
        pytest.param({'event_stream_handler': _handle_events}, id='event_stream_handler'),
    ],
)
def test_delegation_relying_on_the_parent_run_stays_in_it(kwargs: dict[str, Any]) -> None:
    assert _delegate_metadata(SubAgents[None](agents=[SubAgent(dbos_worker)], **kwargs)) is None


def test_delegation_inheriting_tools_stays_in_the_parent_run() -> None:
    with pytest.warns(HarnessDeprecationWarning, match='inherit_tools'):
        capability = SubAgents[None](agents=[SubAgent(dbos_worker)], inherit_tools=True)
    assert _delegate_metadata(capability) is None


def test_child_workflow_needs_every_delegate_durable_without_a_call_budget() -> None:
    plain = Agent(FunctionModel(_worker), name='plain_worker')
    assert _delegate_metadata(SubAgents[None](agents=[SubAgent(dbos_worker)])) == {'dbos': {'child_workflow': {}}}
    assert _delegate_metadata(SubAgents[None](agents=[SubAgent(dbos_worker), SubAgent(plain)])) is None
    assert _delegate_metadata(SubAgents[None](agents=[SubAgent(dbos_worker, max_calls=1)])) is None
