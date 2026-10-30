from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai.messages import ModelResponse
from tests.durable_exec._output_fallback import make_agent, make_models
from tests.harness._temporal import ignore_source_reads_left_open, skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [
    pytest.mark.temporal,
    pytest.mark.xdist_group(name='harness-temporal'),
    skip_temporal_sandbox_on_314,
    ignore_source_reads_left_open,
]

executions: list[str] = []
models = make_models(executions)
agent = make_agent(models, TemporalDurability(models=models), executions)


@workflow.defn
class EscalatingWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self) -> list[str | None]:
        result = await agent.run('Answer.')
        return [m.model_name for m in result.all_messages() if isinstance(m, ModelResponse)]


async def test_replay_reconstructs_selection() -> None:
    executions.clear()
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types', 'coverage')
    )
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        workflow_id = uuid4().hex
        async with Worker(
            client, task_queue='output-escalation', workflows=[EscalatingWorkflow], workflow_runner=runner
        ):
            result = await client.execute_workflow(
                EscalatingWorkflow.run,
                id=workflow_id,
                task_queue='output-escalation',
                execution_timeout=timedelta(seconds=60),
            )
        history = await client.get_workflow_handle(workflow_id).fetch_history()
    assert result == ['primary', 'primary', 'backup']
    replayer = Replayer(workflows=[EscalatingWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=runner)
    replay = await replayer.replay_workflow(history, raise_on_replay_failure=False)
    assert replay.replay_failure is None
    assert executions == ['primary', 'lookup', 'primary', 'backup']
