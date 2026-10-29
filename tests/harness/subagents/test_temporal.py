"""`SubAgents` under `TemporalDurability`: delegation to named agents, to `self` with a model menu, and from folders.

These tests start a local Temporal dev server via `WorkflowEnvironment.start_local()`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.subagents import SubAgent, SubAgents
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

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        for part in respond(messages, info).parts:
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    return FunctionModel(respond, stream_function=stream)


def _worker(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('from the worker')])


async def _stream_worker(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'from the worker'


worker = Agent(FunctionModel(_worker, stream_function=_stream_worker), name='worker')

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


fast = FunctionModel(_worker, stream_function=_stream_worker)
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
