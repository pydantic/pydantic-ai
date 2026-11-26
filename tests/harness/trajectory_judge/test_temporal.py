"""`TrajectoryJudge` under `TemporalDurability`: the evaluation is an activity, and replay is deterministic."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from tests.harness._temporal import ignore_source_reads_left_open, skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

from ._durable_agent import judge_calls, judge_respond, respond, steering, trajectory_judge, work

pytestmark = [
    pytest.mark.temporal,
    pytest.mark.xdist_group(name='harness-temporal'),
    skip_temporal_sandbox_on_314,
    ignore_source_reads_left_open,
]

agent = Agent(
    FunctionModel(respond),
    name='trajectory_judge_temporal',
    deps_type=type(None),
    tools=[work],
    capabilities=[trajectory_judge(FunctionModel(judge_respond)), TemporalDurability[None]()],
)


@workflow.defn
class JudgedWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self) -> list[str]:
        return steering((await agent.run('Summarize notes.txt')).all_messages())


async def test_evaluation_runs_as_an_activity_and_survives_replay() -> None:
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types', 'coverage')
    )
    judge_calls.clear()
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        workflow_id = uuid4().hex
        async with Worker(client, task_queue='trajectory-judge', workflows=[JudgedWorkflow], workflow_runner=runner):
            notes = await client.execute_workflow(
                JudgedWorkflow.run,
                id=workflow_id,
                task_queue='trajectory-judge',
                execution_timeout=timedelta(seconds=60),
            )
        history = await client.get_workflow_handle(workflow_id).fetch_history()

    assert len(notes) == 1
    assert len(judge_calls) == 2
    replayer = Replayer(workflows=[JudgedWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=runner)
    replay = await replayer.replay_workflow(history, raise_on_replay_failure=False)
    assert replay.replay_failure is None
    assert len(judge_calls) == 2, 'the replay asked the judge again'
