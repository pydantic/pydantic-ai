"""`ComputerUse` in a Temporal workflow: each display operation is its own activity, and replay acts on nothing."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.workspaces import Click

pytest.importorskip('temporalio')

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.computer_use import ComputerUse
from tests.harness._temporal import skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

from ...workspace_fakes import InMemoryProvider

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal'), skip_temporal_sandbox_on_314]

_provider = InMemoryProvider(in_unit=activity.in_activity)


def _model(messages: list[ModelMessage], info: object) -> ModelResponse:
    calls = sum(isinstance(part, ToolCallPart) for message in messages for part in message.parts)
    if calls < 2:
        return ModelResponse(parts=[ToolCallPart('computer', {'actions': [{'type': 'click', 'x': calls, 'y': 1}]})])
    return ModelResponse(parts=[TextPart('done')])


_agent = Agent(
    FunctionModel(_model),
    name='computer_use_temporal',
    capabilities=[_provider.capability(), ComputerUse(settle_seconds=0), TemporalDurability()],
)


def _faulty_model(messages: list[ModelMessage], info: object) -> ModelResponse:
    # The model request runs in an activity in this process, so it can plan a fault for the next call's units.
    calls = sum(isinstance(part, ToolCallPart) for message in messages for part in message.parts)
    if calls == 0:
        _provider.lose_next_action = True
        clicks = [{'type': 'click', 'x': x, 'y': 1} for x in range(3)]
        return ModelResponse(parts=[ToolCallPart('computer', {'actions': clicks})])
    if calls == 1:
        _provider.restart_display = True
        return ModelResponse(parts=[ToolCallPart('computer', {'actions': [{'type': 'click', 'x': 9, 'y': 9}]})])
    return ModelResponse(parts=[TextPart('done')])


_faulty_agent = Agent(
    FunctionModel(_faulty_model),
    name='computer_use_temporal_faults',
    capabilities=[_provider.capability(), ComputerUse(settle_seconds=0), TemporalDurability()],
)


@workflow.defn
class ComputerUseWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [_agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _agent.run(prompt)).output


@workflow.defn
class ComputerUseFaultsWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [_faulty_agent]

    @workflow.run
    async def run(self, prompt: str) -> list[str]:
        result = await _faulty_agent.run(prompt)
        return [
            str(part.content_items()[0])
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]


async def test_computer_use_runs_each_action_in_an_activity_and_replays_without_acting() -> None:
    _provider.reset()
    workflow_id = uuid4().hex
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'tests', 'annotated_types')
    )
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        async with Worker(client, task_queue='computer-use', workflows=[ComputerUseWorkflow], workflow_runner=runner):
            output = await client.execute_workflow(
                ComputerUseWorkflow.run,
                'Click twice.',
                id=workflow_id,
                task_queue='computer-use',
                execution_timeout=timedelta(seconds=60),
            )
        history = await client.get_workflow_handle(workflow_id).fetch_history()

    assert output == 'done'
    assert _provider.actions == [Click(x=0, y=1), Click(x=1, y=1)]
    await Replayer(
        workflows=[ComputerUseWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=runner
    ).replay_workflow(history)
    assert len(_provider.actions) == 2


async def test_a_lost_action_is_reported_once_and_a_restart_holds_the_next_action() -> None:
    """The action whose activity failed after it took effect isn't run again, and the model is told it may have run.

    The display then restarts in the activity of the next call, on a backend rebuilt for it: workflow code learns of it
    from the journal, so that call's action doesn't land on the new screen.
    """
    _provider.reset()
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'tests', 'annotated_types')
    )
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        async with Worker(
            client, task_queue='computer-use', workflows=[ComputerUseFaultsWorkflow], workflow_runner=runner
        ):
            lost, held = await client.execute_workflow(
                ComputerUseFaultsWorkflow.run,
                'Click.',
                id=uuid4().hex,
                task_queue='computer-use',
                execution_timeout=timedelta(seconds=60),
            )

    assert _provider.actions == [Click(x=0, y=1)]
    assert lost.startswith(
        'Action 1 of 3 (click) failed: The act call did not finish, so it may or may not have taken effect'
    )
    assert 'The 2 actions after it did not run.' in lost
    assert held.startswith(
        'The display restarted since your last screenshot, so the windows that were open are gone. No action ran'
    )
