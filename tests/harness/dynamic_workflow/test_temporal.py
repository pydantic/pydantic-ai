"""Temporal integration test for DynamicWorkflow.

The orchestration script runs in workflow code through the same Monty loop as CodeMode (see
`pydantic_ai_harness._monty_exec`), so it replays. A sub-agent that carries its own
`TemporalDurability` makes its model requests as activities.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

try:
    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Replayer, Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
    from temporalio.workflow import ActivityConfig
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow

pytestmark = pytest.mark.anyio

TEMPORAL_PORT = 7247  # avoid conflict with the code_mode and spend suites
TASK_QUEUE = 'pydantic-ai-harness-dynamic-workflow-queue'
ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)
SCRIPT = "import asyncio\nawait asyncio.gather(summarize(task='alpha'), summarize(task='beta'))"


def _workflow_runner() -> SandboxedWorkflowRunner:
    # Same passthroughs as `tests/code_mode/test_temporal.py`, for the same reasons.
    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules('coverage', 'pydantic_graph')
    )


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    """Temporal's Python SDK runs on asyncio."""
    return 'asyncio'


@pytest.fixture(scope='module')
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        port=TEMPORAL_PORT,
        dev_server_extra_args=['--dynamic-config-value', 'frontend.enableServerVersionCheck=false'],
    ) as env:
        yield env


@pytest.fixture
async def client(temporal_env: WorkflowEnvironment) -> Client:
    return await Client.connect(f'localhost:{TEMPORAL_PORT}', plugins=[PydanticAIPlugin()])


def _summarize_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    (task,) = [part.content for part in messages[0].parts if isinstance(part, UserPromptPart)]
    return ModelResponse(parts=[TextPart(f'summary of {task}')])


def _orchestrator_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returns = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    if not returns:
        return ModelResponse(parts=[ToolCallPart('run_workflow', {'code': SCRIPT})])
    return ModelResponse(parts=[TextPart(f'done: {returns[0].content}')])


summarize_agent = Agent(
    FunctionModel(_summarize_model),
    name='summarize',
    description='Summarize a text.',
    capabilities=[TemporalDurability(activity_config=ACTIVITY_CONFIG)],
)
orchestrator_agent = Agent(
    FunctionModel(_orchestrator_model),
    name='dynamic_workflow_temporal_agent',
    capabilities=[DynamicWorkflow(agents=[summarize_agent]), TemporalDurability(activity_config=ACTIVITY_CONFIG)],
)


@workflow.defn
class DynamicWorkflowWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await orchestrator_agent.run(prompt)
        return str(result.output)


async def test_dynamic_workflow_runs_in_temporal_workflow(client: Client) -> None:
    """The script runs workflow-side, sub-agents run through activities, and the history replays."""
    workflow_id = 'test_dynamic_workflow_temporal_1'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DynamicWorkflowWorkflow],
        plugins=[AgentPlugin(orchestrator_agent), AgentPlugin(summarize_agent)],
        workflow_runner=_workflow_runner(),
    ):
        output = await client.execute_workflow(
            DynamicWorkflowWorkflow.run,
            args=['Summarize alpha and beta'],
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
    assert output == "done: ['summary of alpha', 'summary of beta']"

    history = await client.get_workflow_handle(workflow_id).fetch_history()
    activities = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.HasField('activity_task_scheduled_event_attributes')
    ]
    assert activities.count('agent__summarize__model_request') == 2

    replay_result = await Replayer(
        workflows=[DynamicWorkflowWorkflow],
        plugins=[PydanticAIPlugin()],
        workflow_runner=_workflow_runner(),
    ).replay_workflow(history)
    assert replay_result.replay_failure is None
