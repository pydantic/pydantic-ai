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
from temporalio.client import Client, WorkflowFailureError
from temporalio.exceptions import ApplicationError
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
    return ModelResponse(parts=[ToolCallPart('_read_file', {'path': 'missing.txt'})])


def _read_file(path: str) -> str:
    return 'Error: file not found'


agent = Agent(
    FunctionModel(_model), name='detect_loops', tools=[_read_file], capabilities=[DetectLoops(), TemporalDurability()]
)
stopping_agent = Agent(
    FunctionModel(_model),
    name='detect_loops_error',
    tools=[_read_file],
    capabilities=[DetectLoops(action='error'), TemporalDurability()],
)


@workflow.defn
class LoopingWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self) -> bytes:
        result = await agent.run('Summarize missing.txt')
        return ModelMessagesTypeAdapter.dump_json(result.all_messages())


@workflow.defn
class StoppingWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [stopping_agent]

    @workflow.run
    async def run(self) -> str:
        return (await stopping_agent.run('Summarize missing.txt')).output


async def test_detection_runs_inside_a_workflow_and_survives_replay() -> None:
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types', 'coverage')
    )
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        workflow_id = uuid4().hex
        async with Worker(
            client, task_queue='detect-loops', workflows=[LoopingWorkflow, StoppingWorkflow], workflow_runner=runner
        ):
            dumped = await client.execute_workflow(
                LoopingWorkflow.run, id=workflow_id, task_queue='detect-loops', execution_timeout=timedelta(seconds=60)
            )
            # `LoopDetectedError` is an `AgentRunError`, so it fails the workflow instead of retrying its task.
            with pytest.raises(WorkflowFailureError) as exc_info:
                await client.execute_workflow(
                    StoppingWorkflow.run,
                    id=uuid4().hex,
                    task_queue='detect-loops',
                    execution_timeout=timedelta(seconds=60),
                )
        history = await client.get_workflow_handle(workflow_id).fetch_history()

    assert isinstance(exc_info.value.cause, ApplicationError)
    assert exc_info.value.cause.type == 'LoopDetectedError'
    messages = ModelMessagesTypeAdapter.validate_json(dumped)
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].text == 'missing.txt does not exist.'
    nudges = [part.metadata for message in messages for part in message.parts if isinstance(part, SystemPromptPart)]
    assert nudges == [
        {
            'origin': 'pydantic-ai-harness.detect_loops.nudge.v1',
            'pattern': 'repeated_result',
            'tool_names': ['_read_file'],
        }
    ]
    # Replaying the recorded history re-runs detection in workflow code; a different decision would be
    # a nondeterminism error.
    replayer = Replayer(workflows=[LoopingWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=runner)
    replay = await replayer.replay_workflow(history, raise_on_replay_failure=False)
    assert replay.replay_failure is None
