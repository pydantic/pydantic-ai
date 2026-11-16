"""`ToolBudgets` under `TemporalDurability`: budgets are counted in workflow code and replay to the same refusals."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models.function import FunctionModel

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness import ToolBudgets
from tests.harness._temporal import ignore_source_reads_left_open, skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

from ._search_agent import refusal, refusals, search, searches, searching_model

pytestmark = [
    pytest.mark.temporal,
    pytest.mark.xdist_group(name='harness-temporal'),
    skip_temporal_sandbox_on_314,
    ignore_source_reads_left_open,
]

agent = Agent(
    FunctionModel(searching_model),
    name='tool_budgets',
    tools=[search],
    capabilities=[ToolBudgets(per_tool={'search': 2}), TemporalDurability()],
)


@workflow.defn
class SearchingWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self) -> bytes:
        result = await agent.run('Research it')
        return ModelMessagesTypeAdapter.dump_json(result.all_messages())


async def test_budget_is_counted_in_workflow_code_and_survives_replay() -> None:
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            __name__, 'tests.harness.tool_budgets._search_agent', 'annotated_types', 'coverage'
        )
    )
    searches.clear()
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        workflow_id = uuid4().hex
        async with Worker(client, task_queue='tool-budgets', workflows=[SearchingWorkflow], workflow_runner=runner):
            dumped = await client.execute_workflow(
                SearchingWorkflow.run,
                id=workflow_id,
                task_queue='tool-budgets',
                execution_timeout=timedelta(seconds=60),
            )
        history = await client.get_workflow_handle(workflow_id).fetch_history()

    assert refusals(ModelMessagesTypeAdapter.validate_json(dumped)) == [refusal('per_tool')]
    assert searches == ['q0', 'q1']
    # Replaying the recorded history re-runs the budget checks in workflow code; a different decision would be a
    # nondeterminism error, and the tool activities are read from history rather than run again.
    replayer = Replayer(workflows=[SearchingWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=runner)
    replay = await replayer.replay_workflow(history, raise_on_replay_failure=False)
    assert replay.replay_failure is None
    assert searches == ['q0', 'q1']
