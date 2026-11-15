"""`RefineOutput` under `TemporalDurability`: the round count lives in workflow code and survives replay."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse, SystemPromptPart, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.refine_output import RefineOutput
from tests.harness._temporal import ignore_source_reads_left_open, skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [
    pytest.mark.temporal,
    pytest.mark.xdist_group(name='harness-temporal'),
    skip_temporal_sandbox_on_314,
    ignore_source_reads_left_open,
]


def _model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    answered = any(isinstance(message, ModelResponse) for message in messages)
    return ModelResponse(parts=[TextPart('revised' if answered else 'draft')])


agent = Agent(FunctionModel(_model), name='refine_output', capabilities=[RefineOutput(), TemporalDurability()])


@workflow.defn
class RefiningWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self) -> bytes:
        result = await agent.run('Explain the change')
        return ModelMessagesTypeAdapter.dump_json(result.all_messages())


async def test_review_runs_inside_a_workflow_and_survives_replay() -> None:
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types', 'coverage')
    )
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        workflow_id = uuid4().hex
        async with Worker(client, task_queue='refine-output', workflows=[RefiningWorkflow], workflow_runner=runner):
            dumped = await client.execute_workflow(
                RefiningWorkflow.run,
                id=workflow_id,
                task_queue='refine-output',
                execution_timeout=timedelta(seconds=60),
            )
        history = await client.get_workflow_handle(workflow_id).fetch_history()

    messages = ModelMessagesTypeAdapter.validate_json(dumped)
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].text == 'revised'
    rounds = [part.metadata for message in messages for part in message.parts if isinstance(part, SystemPromptPart)]
    assert rounds == [{'origin': 'pydantic-ai-harness.refine_output.v1', 'round': 1}]
    # Replaying the recorded history counts the rounds again in workflow code; a different count would be
    # a nondeterminism error.
    replayer = Replayer(workflows=[RefiningWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=runner)
    replay = await replayer.replay_workflow(history, raise_on_replay_failure=False)
    assert replay.replay_failure is None
