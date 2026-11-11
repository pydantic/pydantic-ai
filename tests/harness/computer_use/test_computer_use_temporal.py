"""`ComputerUse` in a Temporal workflow: each `computer` call is one activity, which replay doesn't run again."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from pydantic_ai import Agent
from pydantic_ai.computer_use import Click
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

pytest.importorskip('temporalio')

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
from temporalio.workflow import ActivityConfig

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.computer_use import ComputerUse
from tests.harness._temporal import skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

from ..display_fakes import ScreenProvider

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal'), skip_temporal_sandbox_on_314]

_provider = ScreenProvider(in_unit=activity.in_activity)


def _model(messages: list[ModelMessage], info: object) -> ModelResponse:
    calls = sum(isinstance(part, ToolCallPart) for message in messages for part in message.parts)
    if calls == 0:
        # The model request runs in an activity in this process, so it can plan a fault for the second call.
        return ModelResponse(parts=[ToolCallPart('computer', {'actions': [{'type': 'click', 'x': 0, 'y': 1}]})])
    if calls == 1:
        _provider.lose_next_action = True
        clicks = [{'type': 'click', 'x': x, 'y': 2} for x in range(3)]
        return ModelResponse(parts=[ToolCallPart('computer', {'actions': clicks})])
    return ModelResponse(parts=[TextPart('done')])


_agent = Agent(
    FunctionModel(_model),
    name='computer_use_temporal',
    capabilities=[
        _provider.capability(),
        ComputerUse(settle_seconds=0),
        # The setting the docs give for a run whose calls must not repeat.
        TemporalDurability(
            activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(minutes=5), retry_policy=RetryPolicy(maximum_attempts=1)
            )
        ),
    ],
)


@workflow.defn
class ComputerUseWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [_agent]

    @workflow.run
    async def run(self, prompt: str) -> list[str]:
        result = await _agent.run(prompt)
        return [
            str(part.content_items()[0])
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]


async def test_each_call_is_one_activity_and_replays_without_acting() -> None:
    _provider.reset()
    workflow_id = uuid4().hex
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'tests', 'annotated_types')
    )
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        async with Worker(client, task_queue='computer-use', workflows=[ComputerUseWorkflow], workflow_runner=runner):
            first, lost = await client.execute_workflow(
                ComputerUseWorkflow.run,
                'Click.',
                id=workflow_id,
                task_queue='computer-use',
                execution_timeout=timedelta(minutes=10),
            )
        history = await client.get_workflow_handle(workflow_id).fetch_history()

    assert first.startswith('Ran 1 action.')
    # The second call's worker was lost after its first click. With retries off, the model is told, and nothing ran
    # twice.
    assert lost == (
        'The call was interrupted before it finished, so some of its actions may have run. Take a screenshot before '
        'acting again.'
    )
    assert _provider.actions == [Click(x=0, y=1), Click(x=0, y=2)]

    await Replayer(
        workflows=[ComputerUseWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=runner
    ).replay_workflow(history)
    assert len(_provider.actions) == 2
