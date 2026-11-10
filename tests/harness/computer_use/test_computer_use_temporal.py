"""`ComputerUse` in a Temporal workflow: each `computer` call is one activity that runs at most once."""

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
from temporalio.client import Client, WorkflowHistory
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

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
    capabilities=[_provider.capability(), ComputerUse(settle_seconds=0), TemporalDurability()],
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


def _activities(history: WorkflowHistory, operation: str) -> list[tuple[int, int]]:
    """The retry limit and start-to-close timeout in seconds of each activity `operation` ran as."""
    return [
        (attributes.retry_policy.maximum_attempts, attributes.start_to_close_timeout.ToSeconds())
        for event in history.events
        if event.HasField('activity_task_scheduled_event_attributes')
        and (attributes := event.activity_task_scheduled_event_attributes).activity_type.name.endswith(operation)
    ]


async def test_each_call_is_one_activity_that_runs_once_and_replays_without_acting() -> None:
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
                # Longer than the activities' own timeouts, so the workflow's deadline doesn't cut them short.
                execution_timeout=timedelta(minutes=20),
            )
        history = await client.get_workflow_handle(workflow_id).fetch_history()

    assert first.startswith('Ran 1 action.')
    # The second call's worker was lost after its first click: the model is told, and nothing ran twice.
    assert lost == (
        'The call was interrupted before it finished, so some of its actions may have run. Take a screenshot before '
        'acting again.'
    )
    assert _provider.actions == [Click(x=0, y=1), Click(x=0, y=2)]
    # A call acts, so it runs at most once, with time for 20 long actions.
    assert _activities(history, 'computer__call') == [(1, 900), (1, 900)]
    # Reading the size for the native tool may start the screen, which can install it first.
    sizes = _activities(history, 'computer__display_size')
    assert sizes and {timeout for _, timeout in sizes} == {600}

    await Replayer(
        workflows=[ComputerUseWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=runner
    ).replay_workflow(history)
    assert len(_provider.actions) == 2
