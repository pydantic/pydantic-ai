"""`SubAgents` under `TemporalDurability`: delegation to named agents, to `self` with a model menu, and from folders.

These tests start a local Temporal dev server via `WorkflowEnvironment.start_local()`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

try:
    from temporalio import workflow
    from temporalio.client import Client, WorkflowHistory
    from temporalio.common import RetryPolicy
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Replayer, Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.subagents import DynamicAgents, SubAgent, SubAgents
from tests.harness.subagents._dynamic import POET_CALLS, POET_RESULT, creator
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal')]

TEMPORAL_PORT = 7273  # avoid conflict with the other Temporal suites
TASK_QUEUE = 'pydantic-ai-harness-subagents-queue'
ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)
# See tests/spend/test_temporal.py for why these modules pass through.
_RESTRICTIONS = SandboxRestrictions.default.with_passthrough_modules('coverage', 'annotated_types')


def _delegator(args: dict[str, object]) -> FunctionModel:
    """A parent model that delegates with `args` once, then answers with the delegate's result."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if returns:
            return ModelResponse(parts=[TextPart(returns[-1].model_response_str())])
        first = messages[0].parts[-1]
        if isinstance(first, UserPromptPart) and first.content == 'Delegate it':
            return ModelResponse(parts=[ToolCallPart('delegate_task', args)])
        return ModelResponse(parts=[TextPart('from the delegate')])

    return FunctionModel(respond)


def _worker(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('from the worker')])


worker = Agent(FunctionModel(_worker), name='worker')

named_agent = Agent(
    _delegator({'agent_name': 'worker', 'task': 'Work on it'}),
    name='named_delegating_agent',
    deps_type=type(None),
    capabilities=[
        SubAgents[None](agents=[SubAgent(worker)]),
        TemporalDurability[None](activity_config=ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class NamedWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await named_agent.run(prompt)).output


async def test_delegation_to_a_named_agent_with_its_own_model(client: Client) -> None:
    """The delegate runs in an activity, where the parent's `ctx.model` is unavailable and not needed."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[NamedWorkflow],
        plugins=[AgentPlugin(named_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_RESTRICTIONS),
    ):
        output = await client.execute_workflow(
            NamedWorkflow.run,
            'Delegate it',
            id='test_subagents_temporal_named',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=60),
        )
    assert output == 'from the worker'


fast = FunctionModel(_worker)
menu_agent = Agent(
    _delegator({'agent_name': 'self', 'task': 'Work on it'}),
    name='menu_delegating_agent',
    deps_type=type(None),
    capabilities=[
        SubAgents[None](include_self=True, models={'fast': fast}),
        # The menu's model instance is registered so the delegate can run on it in the workflow.
        TemporalDurability[None](activity_config=ACTIVITY_CONFIG, models={'fast': fast}),
    ],
)
menu_agent_picking = Agent(
    _delegator({'agent_name': 'self', 'task': 'Work on it', 'model': 'fast'}),
    name='menu_picking_delegating_agent',
    deps_type=type(None),
    capabilities=[
        SubAgents[None](include_self=True, models={'fast': fast}),
        TemporalDurability[None](activity_config=ACTIVITY_CONFIG, models={'fast': fast}),
    ],
)


@workflow.defn
class MenuWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await menu_agent.run(prompt)).output


@workflow.defn
class MenuPickingWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await menu_agent_picking.run(prompt)).output


@pytest.mark.parametrize(
    ('workflow_type', 'agent', 'expected'),
    [
        pytest.param(MenuWorkflow, menu_agent, 'from the delegate', id='own-model'),
        pytest.param(MenuPickingWorkflow, menu_agent_picking, 'from the worker', id='menu-model'),
    ],
)
async def test_delegation_to_self_with_a_model_menu_runs_in_the_workflow(
    client: Client, workflow_type: type[MenuWorkflow | MenuPickingWorkflow], agent: Agent[None, str], expected: str
) -> None:
    """With or without a menu key, the delegate to `self` runs in the workflow, where the parent's model is known."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[workflow_type],
        plugins=[AgentPlugin(agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_RESTRICTIONS),
    ):
        output = await client.execute_workflow(
            workflow_type.run,
            'Delegate it',
            id=f'test_subagents_temporal_{agent.name}',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=60),
        )
    assert output == expected


dynamic_agent = Agent(
    FunctionModel(creator),
    name='dynamic_delegating_agent',
    deps_type=type(None),
    capabilities=[
        SubAgents[None](dynamic_agents=DynamicAgents()),
        TemporalDurability[None](activity_config=ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class DynamicWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await dynamic_agent.run(prompt)).output


async def test_created_agent_survives_replay(client: Client) -> None:
    """Every workflow task replays from history, so the roster is rebuilt and the delegation's result reused."""
    POET_CALLS.clear()
    workflow_id = 'test_subagents_temporal_dynamic'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DynamicWorkflow],
        plugins=[AgentPlugin(dynamic_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_RESTRICTIONS),
        max_cached_workflows=0,
        debug_mode=True,
    ):
        output = await client.execute_workflow(
            DynamicWorkflow.run,
            'Write a poem',
            id=workflow_id,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=60),
        )
    assert output == POET_RESULT
    assert len(POET_CALLS) == 1

    history = await client.get_workflow_handle(workflow_id).fetch_history()
    activities = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.HasField('activity_task_scheduled_event_attributes')
    ]
    assert 'agent__dynamic_delegating_agent__capability__sub_agents__delegate' in activities
    replay = await Replayer(
        workflows=[DynamicWorkflow],
        plugins=[PydanticAIPlugin()],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_RESTRICTIONS),
        debug_mode=True,
    ).replay_workflow(history)
    assert replay.replay_failure is None


durable_worker = Agent(
    FunctionModel(_worker),
    name='durable_worker',
    capabilities=[TemporalDurability(activity_config=ACTIVITY_CONFIG)],
)
child_workflow_agent = Agent(
    _delegator({'agent_name': 'durable_worker', 'task': 'Work on it'}),
    name='child_workflow_delegating_agent',
    deps_type=type(None),
    capabilities=[
        SubAgents[None](agents=[SubAgent(durable_worker)]),
        TemporalDurability[None](activity_config=ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class ChildWorkflowWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await child_workflow_agent.run(prompt)).output


def _activities(history: WorkflowHistory) -> list[str]:
    return [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.HasField('activity_task_scheduled_event_attributes')
    ]


async def test_durable_delegate_runs_as_a_child_workflow(client: Client) -> None:
    """A delegate with its own `TemporalDurability` runs in a child workflow, whose history holds its activities."""
    workflow_id = 'test_subagents_temporal_child_workflow'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ChildWorkflowWorkflow],
        plugins=[AgentPlugin(child_workflow_agent), AgentPlugin(durable_worker)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_RESTRICTIONS),
    ):
        output = await client.execute_workflow(
            ChildWorkflowWorkflow.run,
            'Delegate it',
            id=workflow_id,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=60),
        )
    assert output == 'from the worker'

    history = await client.get_workflow_handle(workflow_id).fetch_history()
    [child_id] = [
        event.start_child_workflow_execution_initiated_event_attributes.workflow_id
        for event in history.events
        if event.HasField('start_child_workflow_execution_initiated_event_attributes')
    ]
    assert not any('sub_agents' in name for name in _activities(history))
    child_history = await client.get_workflow_handle(child_id).fetch_history()
    assert 'agent__durable_worker__model_request' in _activities(child_history)
    replay = await Replayer(
        workflows=[ChildWorkflowWorkflow],
        plugins=[PydanticAIPlugin()],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_RESTRICTIONS),
    ).replay_workflow(history)
    assert replay.replay_failure is None


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    """Temporal's Python SDK runs on asyncio."""
    return 'asyncio'


@pytest.fixture(scope='module')
async def client() -> AsyncIterator[Client]:
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        port=TEMPORAL_PORT,
        dev_server_extra_args=['--dynamic-config-value', 'frontend.enableServerVersionCheck=false'],
        download_dest_dir=temporal_dev_server_cache_dir(),
    ):
        yield await Client.connect(f'localhost:{TEMPORAL_PORT}', plugins=[PydanticAIPlugin()])
