"""`WrapUpAtLimits` under `TemporalDurability`: the time trigger reads the workflow clock and decides the same on replay."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.wrap_up_at_limits import DEFAULT_WRAP_UP_PROMPT, WrapUpAtLimits
from tests.harness._temporal import ignore_source_reads_left_open, skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

from ._messages import wrap_ups

pytestmark = [
    pytest.mark.temporal,
    pytest.mark.xdist_group(name='harness-temporal'),
    skip_temporal_sandbox_on_314,
    ignore_source_reads_left_open,
]


def _model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if info.model_settings and info.model_settings.get('tool_choice') == 'none':
        return ModelResponse(parts=[TextPart('what I found so far')])
    return ModelResponse(parts=[ToolCallPart('look_up', {'topic': 'a'})])  # pragma: no cover


def look_up(topic: str) -> str:  # pragma: no cover
    return f'notes on {topic}'


# A 60s reserve on a 30s timeout makes the first request the last, whatever the steps take.
agent = Agent(
    FunctionModel(_model),
    name='wrap_up_temporal',
    tools=[look_up],
    capabilities=[WrapUpAtLimits(timeout_margin=60), TemporalDurability()],
)


@workflow.defn
class WrapUpWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self) -> bytes:
        result = await agent.run('Research', timeout=30)
        return ModelMessagesTypeAdapter.dump_json(result.all_messages())


async def test_timeout_wraps_up_in_workflow_code_and_survives_replay() -> None:
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types', 'coverage')
    )
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        workflow_id = uuid4().hex
        async with Worker(client, task_queue='wrap-up-at-limits', workflows=[WrapUpWorkflow], workflow_runner=runner):
            dumped = await client.execute_workflow(
                WrapUpWorkflow.run,
                id=workflow_id,
                task_queue='wrap-up-at-limits',
                execution_timeout=timedelta(seconds=60),
            )
        history = await client.get_workflow_handle(workflow_id).fetch_history()

    messages = ModelMessagesTypeAdapter.validate_json(dumped)
    assert wrap_ups(messages) == [(DEFAULT_WRAP_UP_PROMPT, 'timeout')]
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].text == 'what I found so far'
    # Replaying re-runs the hooks in workflow code against the workflow clock; a different decision would be
    # a nondeterminism error.
    replayer = Replayer(workflows=[WrapUpWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=runner)
    replay = await replayer.replay_workflow(history, raise_on_replay_failure=False)
    assert replay.replay_failure is None
