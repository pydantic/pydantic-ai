"""Temporal integration tests for `BackgroundTools`."""

from __future__ import annotations

import asyncio
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
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset

from pydantic_ai_harness import BackgroundTools

pytestmark = pytest.mark.anyio

TEMPORAL_PORT = 7253
TASK_QUEUE = 'pydantic-ai-harness-background-tools'
_tool_calls = 0


def _model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if (
                    isinstance(part, UserPromptPart)
                    and isinstance(part.content, str)
                    and 'Background tool' in part.content
                ):
                    return ModelResponse(parts=[TextPart(content='done')])

    if any(
        isinstance(part, ToolReturnPart) and 'running in background' in str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
    ):
        return ModelResponse(parts=[TextPart(content='waiting')])
    return ModelResponse(parts=[ToolCallPart(tool_name='research', args={}, tool_call_id='research-call')])


async def research() -> str:
    global _tool_calls
    _tool_calls += 1
    await asyncio.sleep(0.05)
    return 'durable result'


_agent = Agent(
    FunctionModel(_model),
    name='background_tools_temporal_agent',
    toolsets=[FunctionToolset(tools=[research], id='background-tools', metadata={'background': True})],
    capabilities=[
        BackgroundTools(),
        TemporalDurability(
            activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=30), retry_policy=RetryPolicy(maximum_attempts=1)
            )
        ),
    ],
)


@workflow.defn
class BackgroundToolsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _agent.run(prompt)
        return str(result.output)


def _workflow_runner() -> SandboxedWorkflowRunner:
    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            'annotated_types', 'coverage', 'pydantic_graph'
        )
    )


@pytest.fixture(scope='module')
def anyio_backend() -> str:
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


async def test_background_result_survives_temporal_history_replay(client: Client) -> None:
    global _tool_calls
    _tool_calls = 0
    workflow_id = 'test_background_tools_temporal_replay'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[BackgroundToolsWorkflow],
        plugins=[AgentPlugin(_agent)],
        workflow_runner=_workflow_runner(),
    ):
        output = await client.execute_workflow(
            BackgroundToolsWorkflow.run,
            args=['go'],
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )

    assert output == 'done'
    assert _tool_calls == 1

    history = await client.get_workflow_handle(workflow_id).fetch_history()
    replay = await Replayer(
        workflows=[BackgroundToolsWorkflow],
        plugins=[PydanticAIPlugin()],
        workflow_runner=_workflow_runner(),
    ).replay_workflow(history)

    assert replay.replay_failure is None
    assert _tool_calls == 1
