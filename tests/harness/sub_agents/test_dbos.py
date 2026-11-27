"""`SubAgents` under `DBOSDurability`: recovery reuses created agents and background tasks; child-workflow delegates."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Generator
from pathlib import Path

import pytest

from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import PrefixTools, SetToolMetadata
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai_harness import DenyUnanswered
from pydantic_ai_harness.ask_user import AskUser
from pydantic_ai_harness.deny_unanswered import DEFAULT_QUESTION_MESSAGE
from pydantic_ai_harness.permissions import PermissionMode, PermissionRule, Permissions
from pydantic_ai_harness.sub_agents import DynamicAgents, SubAgent, SubAgents, SubAgentToolset, TaskControl
from tests.conftest import try_import
from tests.harness._deferred_tools import QUESTION, call_then_echo
from tests.harness.sub_agents._background import (
    WORKER_CALLS,
    EventKinds,
    background_parent,
    background_worker,
    reset_worker,
)
from tests.harness.sub_agents._dynamic import POET_CALLS, POET_RESULT, creator

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
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()


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


@pytest.mark.parametrize(
    ('name', 'sub_agents'),
    [
        pytest.param('budgeted', lambda: SubAgents[None](agents=[SubAgent(dbos_worker, max_calls=1)]), id='max_calls'),
        pytest.param(
            'task_control',
            lambda: SubAgents[None](agents=[SubAgent(dbos_worker)], task_control=TaskControl()),
            id='task_control',
        ),
        pytest.param(
            'managed',
            lambda: SubAgents[None](agents=[SubAgent(dbos_worker)]),
            id='managed_permissions',
        ),
    ],
)
async def test_child_workflow_rejects_options_that_need_the_parent_run(
    dbos: DBOS, name: str, sub_agents: Callable[[], SubAgents[None]]
) -> None:
    agent = Agent(
        FunctionModel(_delegator),
        name=f'dbos_{name}_delegator',
        deps_type=type(None),
        capabilities=[
            sub_agents(),
            *([Permissions[None](managed_rules=[PermissionRule('deny', command='rm')])] if name == 'managed' else []),
            SetToolMetadata(tools=['delegate_task'], dbos={'child_workflow': {}}),
            DBOSDurability[None](),
        ],
    )

    @DBOS.workflow(name=f'subagents_{name}_child_workflow')
    async def workflow() -> str:
        return (await agent.run('Delegate it')).output

    with SetWorkflowID(str(uuid.uuid4())), pytest.raises(UserError, match='does not support'):
        await workflow()


async def test_background_task_survives_recovery(dbos: DBOS) -> None:
    events = EventKinds()
    worker = Agent(FunctionModel(background_worker), name='worker')
    agent = Agent(
        background_parent(),
        name='subagents_background',
        deps_type=type(None),
        capabilities=[
            SubAgents[None](agents=[SubAgent(worker)], task_control=TaskControl()),
            events,
            DBOSDurability[None](),
        ],
    )

    @DBOS.workflow(name='subagents_background')
    async def workflow() -> list[str]:
        events.kinds.clear()
        return [(await agent.run('Delegate it')).output, *events.kinds]

    reset_worker()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        output, *kinds = await workflow()
    assert 'from the worker' in output
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert 'subagents_background__capability__sub_agents.delegate' in {step['function_name'] for step in steps}

    # Re-execute the workflow function from its last step, the way recovery does.
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    recovered, *recovered_kinds = await handle.get_result(polling_interval_sec=0.01)
    assert 'from the worker' in recovered
    # DBOS doesn't journal how the task interleaves with the delegating tool call, which announces the backgrounding.
    for order in (kinds, recovered_kinds):
        assert [kind for kind in order if kind != 'sub_agents.task_backgrounded'] == [
            'sub_agents.task_assigned',
            'sub_agents.task_started',
            'sub_agents.task_completed',
        ]
        assert order.count('sub_agents.task_backgrounded') == 1
    assert WORKER_CALLS == ['worker'], 'recovery ran the background task again'


async def test_parent_deny_unanswered_reaches_a_durable_child_in_the_workflow(dbos: DBOS) -> None:
    """A delegate bound to `DBOSDurability` accepts the parent's forwarded `DenyUnanswered` inside the workflow."""
    observed: list[str] = []
    child_agent = Agent(
        call_then_echo('ask_user_question', {'questions': QUESTION}, 'q1', 'child done', observed),
        name='deny_child',
        capabilities=[AskUser[object](answerer=None), DBOSDurability[object]()],
    )
    parent = Agent(
        call_then_echo('delegate_task', {'agent_name': 'deny_child', 'task': 'ask'}, 'd1', 'parent done'),
        name='deny_parent',
        capabilities=[
            DenyUnanswered[object](),
            SubAgents[object](agents=[SubAgent(child_agent)]),
            DBOSDurability[object](),
        ],
    )

    @DBOS.workflow(name='deny_parent_workflow')
    async def workflow() -> str:
        return (await parent.run('delegate')).output

    assert await workflow() == 'parent done'
    assert observed == [DEFAULT_QUESTION_MESSAGE]


async def test_the_parents_managed_permissions_reach_a_durable_child_in_the_workflow(dbos: DBOS) -> None:
    """A delegate bound to `DBOSDurability` is held to the parent's managed rules without `shared_capabilities`."""
    observed: list[str] = []

    def run_command(command: str) -> str:
        raise AssertionError('the denied command must not run')  # pragma: no cover

    child_agent = Agent(
        call_then_echo('run_command', {'command': 'rm -rf doomed'}, 'c1', 'child done', observed),
        name='managed_child',
        tools=[Tool(run_command, metadata={'code_arg_language': 'shell', 'code_arg_name': 'command'})],
        capabilities=[DBOSDurability[object]()],
    )
    parent = Agent(
        call_then_echo('delegate_task', {'agent_name': 'managed_child', 'task': 'clean up'}, 'd1', 'parent done'),
        name='managed_parent',
        capabilities=[
            Permissions[object](managed_rules=[PermissionRule('deny', command='rm')]),
            SubAgents[object](agents=[SubAgent(child_agent)]),
            DBOSDurability[object](),
        ],
    )

    @DBOS.workflow(name='managed_parent_workflow')
    async def workflow() -> str:
        return (await parent.run('delegate')).output

    assert await workflow() == 'parent done'
    assert observed == ['A managed permission rule denied this command, so it did not run. Do not retry it.']


@pytest.mark.parametrize(
    ('managed', 'shape'),
    [(True, 'capability'), (True, 'wrapped'), (True, 'toolset'), (False, 'capability')],
    ids=['managed', 'managed_wrapped', 'managed_toolset', 'unmanaged'],
)
async def test_a_durable_delegate_with_its_own_sub_agents_is_refused_under_a_managed_layer(
    dbos: DBOS, managed: bool, shape: str
) -> None:
    """Its child-workflow delegations can't see the layer it inherited; without a layer it runs as before."""
    grandchild = Agent(FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('grandchild done')])))
    sub_agents = SubAgents[object](agents=[SubAgent(grandchild, name='grandchild')])
    toolset = SubAgentToolset[object](
        agents={'grandchild': SubAgent(grandchild)},
        forward_usage=True,
        inherit_tools=False,
        shared_capabilities=[],
        event_stream_handler=None,
        tool_name='delegate_task',
        tool_retries=1,
        contain_errors=False,
        call_counts={},
    )
    child_agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('child done')])),
        name=f'delegating_child_{managed}_{shape}',
        toolsets=[toolset] if shape == 'toolset' else [],
        capabilities=[
            *([PrefixTools[object](wrapped=sub_agents, prefix='team')] if shape == 'wrapped' else []),
            *([sub_agents] if shape == 'capability' else []),
            DBOSDurability[object](),
        ],
    )
    parent = Agent(
        call_then_echo('delegate_task', {'agent_name': 'child', 'task': 'go'}, 'd1', 'parent done'),
        name=f'delegating_parent_{managed}_{shape}',
        capabilities=[
            *([Permissions[object](managed_rules=[PermissionRule('deny', command='rm')])] if managed else []),
            SubAgents[object](agents=[SubAgent(child_agent, name='child')]),
            DBOSDurability[object](),
        ],
    )

    @DBOS.workflow(name=f'delegating_parent_{managed}_{shape}_workflow')
    async def workflow() -> str:
        return (await parent.run('delegate')).output

    if managed:
        with pytest.raises(UserError, match=r"Sub-agent 'child' is bound to a durable execution engine .*14944"):
            await workflow()
    else:
        assert await workflow() == 'parent done'


def _delegates_to_self_once(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    if returns:
        return ModelResponse(parts=[TextPart(returns[-1].model_response_str())])
    first = messages[0].parts[-1]
    if isinstance(first, UserPromptPart) and first.content == 'Delegate it':
        return ModelResponse(parts=[ToolCallPart('delegate_task', {'agent_name': 'self', 'task': 'Work on it'})])
    return ModelResponse(parts=[TextPart('from the delegate')])


def _full_mode(ctx: RunContext[object]) -> PermissionMode:
    return 'full'


@pytest.mark.parametrize('max_mode', ['full', _full_mode], ids=['string', 'function'])
async def test_a_durable_delegate_to_self_runs_under_its_own_managed_layer(
    dbos: DBOS, max_mode: PermissionMode | Callable[[RunContext[object]], PermissionMode]
) -> None:
    """The delegate holds the managed `Permissions` itself, so its child workflows read the layer."""
    agent = Agent(
        FunctionModel(_delegates_to_self_once),
        name=f'dbos_self_managed_{callable(max_mode)}',
        capabilities=[
            Permissions[object](max_mode=max_mode, managed_rules=[PermissionRule('deny', command='rm')]),
            SubAgents[object](include_self=True),
            DBOSDurability[object](),
        ],
    )

    @DBOS.workflow(name=f'dbos_self_managed_{callable(max_mode)}_workflow')
    async def workflow() -> str:
        return (await agent.run('Delegate it')).output

    assert await workflow() == 'from the delegate'


@pytest.mark.parametrize('max_mode', ['full', _full_mode], ids=['string', 'function'])
async def test_a_durable_delegate_with_the_same_managed_layer_runs(
    dbos: DBOS, max_mode: PermissionMode | Callable[[RunContext[object]], PermissionMode]
) -> None:
    """A delegate whose own `Permissions` has the run's limits reads them in its child workflows.

    A `max_mode` function is refused: the delegate's child workflow would call it with the delegate's context.
    """
    rules = [PermissionRule[object]('deny', command='rm')]
    grandchild = Agent(FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('grandchild done')])))
    child_agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('child done')])),
        name=f'holding_child_{callable(max_mode)}',
        capabilities=[
            Permissions[object](max_mode=max_mode, managed_rules=rules),
            SubAgents[object](agents=[SubAgent(grandchild, name='grandchild')]),
            DBOSDurability[object](),
        ],
    )
    parent = Agent(
        call_then_echo('delegate_task', {'agent_name': 'child', 'task': 'go'}, 'd1', 'parent done'),
        name=f'holding_parent_{callable(max_mode)}',
        capabilities=[
            Permissions[object](max_mode=max_mode, managed_rules=rules),
            SubAgents[object](agents=[SubAgent(child_agent, name='child')]),
            DBOSDurability[object](),
        ],
    )

    @DBOS.workflow(name=f'holding_parent_{callable(max_mode)}_workflow')
    async def workflow() -> str:
        return (await parent.run('delegate')).output

    if callable(max_mode):
        with pytest.raises(UserError, match=r"Sub-agent 'child' is bound to a durable execution engine .*14944"):
            await workflow()
    else:
        assert await workflow() == 'parent done'


async def test_a_managed_permissions_added_per_run_is_refused_in_the_workflow(dbos: DBOS) -> None:
    """So the agent's own capabilities, which a child-workflow delegation reads, hold every managed `Permissions`."""
    agent = Agent(
        FunctionModel(_delegator),
        name='dbos_per_run_managed',
        deps_type=type(None),
        capabilities=[DBOSDurability[None]()],
    )

    @DBOS.workflow(name='subagents_per_run_managed')
    async def workflow() -> str:
        managed = Permissions[None](managed_rules=[PermissionRule('deny', command='rm')])
        return (await agent.run('Delegate it', capabilities=[managed])).output

    with SetWorkflowID(str(uuid.uuid4())), pytest.raises(UserError, match='added per-run inside a DBOS workflow'):
        await workflow()
