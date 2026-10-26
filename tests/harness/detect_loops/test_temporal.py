"""`DetectLoops` under `TemporalDurability`: detection runs in workflow code and reaches the same decision on replay."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.detect_loops import DetectLoops
from tests.harness._temporal import ignore_source_reads_left_open, skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [
    pytest.mark.temporal,
    pytest.mark.xdist_group(name='harness-temporal'),
    skip_temporal_sandbox_on_314,
    ignore_source_reads_left_open,
]


def _model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if 'Change your approach' in str(messages[-1]):
        return ModelResponse(parts=[TextPart('missing.txt does not exist.')])
    return ModelResponse(parts=[ToolCallPart('read_file', {'path': 'missing.txt'})])


agent = Agent(FunctionModel(_model), name='detect_loops', capabilities=[DetectLoops(), TemporalDurability()])


@agent.tool_plain
def read_file(path: str) -> str:
    return 'Error: file not found'


@workflow.defn
class LoopingWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self) -> bytes:
        result = await agent.run('Summarize missing.txt')
        return ModelMessagesTypeAdapter.dump_json(result.all_messages())


async def test_nudge_is_delivered_inside_a_workflow_and_survives_replay() -> None:
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types', 'coverage')
    )
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        workflow_id = uuid4().hex
        async with Worker(client, task_queue='detect-loops', workflows=[LoopingWorkflow], workflow_runner=runner):
            dumped = await client.execute_workflow(
                LoopingWorkflow.run, id=workflow_id, task_queue='detect-loops', execution_timeout=timedelta(seconds=60)
            )
        history = await client.get_workflow_handle(workflow_id).fetch_history()

    messages = ModelMessagesTypeAdapter.validate_json(dumped)
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].text == 'missing.txt does not exist.'
    nudges = [part.metadata for message in messages for part in message.parts if isinstance(part, SystemPromptPart)]
    assert nudges == [
        {
            'origin': 'pydantic-ai-harness.detect_loops.nudge.v1',
            'pattern': 'repeated_result',
            'tool_names': ['read_file'],
        }
    ]
    # Replaying the recorded history re-runs detection in workflow code; a different decision would be
    # a nondeterminism error.
    replayer = Replayer(workflows=[LoopingWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=runner)
    replay = await replayer.replay_workflow(history, raise_on_replay_failure=False)
    assert replay.replay_failure is None
