"""SubAgents 与 Temporal workflow/activity 边界的集成回归测试。"""

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
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from tests.harness.conftest import ignore_source_reads_left_open, skip_temporal_sandbox_on_314

pytestmark = [
    pytest.mark.temporal,
    pytest.mark.xdist_group(name='harness-temporal'),
    ignore_source_reads_left_open,
    skip_temporal_sandbox_on_314,
]

TASK_QUEUE = 'pydantic-ai-harness-subagents-temporal-queue'
ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=30),
    retry_policy=RetryPolicy(maximum_attempts=1),
)
_SANDBOXED = SandboxRestrictions.default.with_passthrough_modules('coverage', 'annotated_types', 'pydantic_graph')


def _delegate_named_agent(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    if returns:
        return ModelResponse(parts=[TextPart(returns[-1].model_response_str())])
    return ModelResponse(
        parts=[ToolCallPart('delegate_task', {'agent_name': 'named_worker', 'task': 'Run the named task'})]
    )


_named_worker = Agent(TestModel(custom_output_text='named-worker-result'), name='named_worker')
_named_delegate_agent = Agent(
    FunctionModel(_delegate_named_agent),
    name='named_delegate_agent',
    deps_type=type(None),
    capabilities=[
        SubAgents[None](agents=[SubAgent(_named_worker)], agent_folders=None),
        TemporalDurability[None](activity_config=ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class NamedDelegateWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _named_delegate_agent.run(prompt)).output


def _delegate_self_with_menu(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    if returns:
        return ModelResponse(parts=[TextPart(returns[-1].model_response_str())])
    return ModelResponse(
        parts=[ToolCallPart('delegate_task', {'agent_name': 'self', 'task': 'Run the self task', 'model': 'test'})]
    )


_self_menu_agent = Agent(
    FunctionModel(_delegate_self_with_menu),
    name='self_menu_agent',
    deps_type=type(None),
    capabilities=[
        SubAgents[None](
            include_self=True,
            models={'test': TestModel(custom_output_text='self-menu-result', call_tools=[])},
            agent_folders=None,
        ),
        TemporalDurability[None](activity_config=ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class SelfMenuWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _self_menu_agent.run(prompt)).output


@pytest.fixture(scope='module')
async def client() -> AsyncIterator[Client]:
    async with await WorkflowEnvironment.start_local() as env:  # pyright: ignore[reportUnknownMemberType]
        yield await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])


async def test_named_agent_with_own_model_runs_as_temporal_activity(client: Client) -> None:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[NamedDelegateWorkflow],
        plugins=[AgentPlugin(_named_delegate_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_SANDBOXED),
    ):
        output = await client.execute_workflow(
            NamedDelegateWorkflow.run,
            'Delegate to the named worker',
            id='test_temporal_named_subagent_own_model',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=60),
        )

    assert output == 'named-worker-result'


async def test_self_delegate_with_model_menu_runs_under_temporal(client: Client) -> None:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SelfMenuWorkflow],
        plugins=[AgentPlugin(_self_menu_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_SANDBOXED),
    ):
        output = await client.execute_workflow(
            SelfMenuWorkflow.run,
            'Delegate to self with a selected model',
            id='test_temporal_self_subagent_model_menu',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=60),
        )

    assert output == 'self-menu-result'
