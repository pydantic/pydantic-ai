"""Temporal integration tests for `ToolBridge`.

The bridge runs workflow-side: `tool_call` dispatches through the run's tool manager, and the bridged
tool itself runs in an activity, like a direct call. The history replays.
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
    from temporalio.worker import Replayer, Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai_harness import ToolBridge
from tests.harness._temporal import ignore_source_reads_left_open
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal'), ignore_source_reads_left_open]

TEMPORAL_PORT = 7264  # avoid conflict with the other Temporal suites
TASK_QUEUE = 'pydantic-ai-harness-tool-bridge'

_CALLS = [
    ToolCallPart('tool_call', {'name': 'add', 'args_json': '{"a": 3, "b": 4}'}, tool_call_id='bridge-add'),
    ToolCallPart('tool_call', {'name': 'delete_file', 'args_json': '{"path": "a.txt"}'}, tool_call_id='bridge-delete'),
]


def _model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    step = sum(isinstance(message, ModelResponse) for message in messages)
    if step < len(_CALLS):
        return ModelResponse(parts=[_CALLS[step]])
    returns = [
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    return ModelResponse(parts=[TextPart(' | '.join(returns))])


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def delete_file(ctx: RunContext[object], path: str) -> str:
    """Delete a file."""
    if not ctx.tool_call_approved:
        raise ApprovalRequired
    return f'deleted {path} as {ctx.tool_call_id} for {ctx.tool_call_metadata}'


def _approve(ctx: RunContext[object], requests: DeferredToolRequests) -> DeferredToolResults:
    return requests.build_results(
        approve_all=True, metadata={call.tool_call_id: {'by': 'alice'} for call in requests.approvals}
    )


_agent = Agent(
    FunctionModel(_model),
    name='tool_bridge_temporal_agent',
    toolsets=[FunctionToolset(tools=[add, delete_file], id='files')],
    capabilities=[
        HandleDeferredToolCalls(_approve),
        ToolBridge(),
        TemporalDurability(
            activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=30), retry_policy=RetryPolicy(maximum_attempts=1)
            )
        ),
    ],
)


@workflow.defn
class ToolBridgeWorkflow:
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


async def test_bridged_calls_run_and_replay(client: Client) -> None:
    workflow_id = 'test_tool_bridge_temporal_replay'
    # `debug_mode=True` disables the 2-second deadlock detector, which a cold coverage run can trip;
    # see `tests/harness/background_tools/test_temporal.py`.
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ToolBridgeWorkflow],
        plugins=[AgentPlugin(_agent)],
        workflow_runner=_workflow_runner(),
        debug_mode=True,
    ):
        output = await client.execute_workflow(
            ToolBridgeWorkflow.run, args=['go'], id=workflow_id, task_queue=TASK_QUEUE
        )

    assert output == "7 | deleted a.txt as bridge-delete for {'by': 'alice'}"

    history = await client.get_workflow_handle(workflow_id).fetch_history()
    replay = await Replayer(
        workflows=[ToolBridgeWorkflow],
        plugins=[PydanticAIPlugin()],
        workflow_runner=_workflow_runner(),
        debug_mode=True,
    ).replay_workflow(history)

    assert replay.replay_failure is None
