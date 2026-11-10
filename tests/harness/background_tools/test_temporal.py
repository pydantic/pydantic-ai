"""Temporal integration tests for `BackgroundTools`."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Replayer, Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from pydantic_ai import Agent, DeferredToolRequests, Tool
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai_harness import BackgroundTools
from tests.harness._temporal import ignore_source_reads_left_open
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal'), ignore_source_reads_left_open]

TEMPORAL_PORT = 7253
TASK_QUEUE = 'pydantic-ai-harness-background-tools'
_tool_calls = 0


def _model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if (
                    isinstance(part, UserPromptPart)
                    and not isinstance(part.content, str)
                    and any(
                        isinstance(item, TextContent)
                        and item.metadata
                        == {'origin': 'pydantic-ai-harness.background_tools.result.v1', 'tool_call_id': 'research-call'}
                        for item in part.content
                    )
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


def _pausing_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(tool_name='research', args={}, tool_call_id='research-call'),
            ToolCallPart(tool_name='gated', args={}, tool_call_id='gated-call'),
        ]
    )


def gated() -> str:
    # Never approved in this test.
    return 'approved'  # pragma: no cover


_pausing_agent = Agent(
    FunctionModel(_pausing_model),
    name='background_tools_temporal_pausing_agent',
    output_type=[str, DeferredToolRequests],
    toolsets=[
        FunctionToolset(tools=[research], id='background-tools', metadata={'background': True}),
        FunctionToolset(tools=[Tool(gated, requires_approval=True)], id='gated-tools'),
    ],
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
class BackgroundToolsPauseWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[str]:
        result = await _pausing_agent.run(prompt)
        assert isinstance(result.output, DeferredToolRequests)
        recorded = result.all_messages()[-1]
        assert isinstance(recorded, ModelRequest)
        return [
            item.content if isinstance(item, TextContent) else str(item)
            for part in recorded.parts
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
            for item in part.content
        ]


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
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        port=TEMPORAL_PORT,
        dev_server_extra_args=['--dynamic-config-value', 'frontend.enableServerVersionCheck=false'],
        download_dest_dir=temporal_dev_server_cache_dir(),
    ) as env:
        yield env


@pytest.fixture
async def client(temporal_env: WorkflowEnvironment) -> Client:
    return await Client.connect(f'localhost:{TEMPORAL_PORT}', plugins=[PydanticAIPlugin()])


async def test_background_result_survives_temporal_history_replay(client: Client) -> None:
    global _tool_calls
    _tool_calls = 0
    workflow_id = 'test_background_tools_temporal_replay'
    # `debug_mode=True` disables Temporal's 2-second workflow-task deadlock detector. Since
    # `temporalio` 1.29 it also runs each activation inline on the test's event loop instead of in
    # an executor thread, and lifts the sandbox's `breakpoint()` restriction on the runner passed
    # here. Under `coverage`'s `sys.monitoring` core on Python 3.14, the first activation pays for
    # lazily parsing every traced source file, which can exceed 2 seconds on a busy CI runner and
    # fail the run with a spurious `_DeadlockError` (TMPRL1101). The test asserts replay
    # determinism, not timing.
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[BackgroundToolsWorkflow],
        plugins=[AgentPlugin(_agent)],
        workflow_runner=_workflow_runner(),
        debug_mode=True,
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
        debug_mode=True,
    ).replay_workflow(history)

    assert replay.replay_failure is None
    assert _tool_calls == 1


async def test_deferred_pause_waits_for_a_background_result_under_temporal(client: Client) -> None:
    """A durable run pausing for approval records a live background task's result, and replays it (#7537)."""
    global _tool_calls
    _tool_calls = 0
    workflow_id = 'test_background_tools_temporal_pause'
    # `debug_mode=True` for the reason given in `test_background_result_survives_temporal_history_replay`.
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[BackgroundToolsPauseWorkflow],
        plugins=[AgentPlugin(_pausing_agent)],
        workflow_runner=_workflow_runner(),
        debug_mode=True,
    ):
        recorded = await client.execute_workflow(
            BackgroundToolsPauseWorkflow.run,
            args=['go'],
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )

    # The result is recorded after the paused history instead of being dropped with the task.
    assert any('durable result' in text for text in recorded)
    assert _tool_calls == 1

    history = await client.get_workflow_handle(workflow_id).fetch_history()
    replay = await Replayer(
        workflows=[BackgroundToolsPauseWorkflow],
        plugins=[PydanticAIPlugin()],
        workflow_runner=_workflow_runner(),
        debug_mode=True,
    ).replay_workflow(history)

    assert replay.replay_failure is None
    assert _tool_calls == 1
