from __future__ import annotations

import sys
from datetime import timedelta
from uuid import uuid4

import pytest

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.worker import Replayer, Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai.messages import ModelResponse
from tests.durable_exec._output_fallback import make_agent, make_models

if sys.version_info >= (3, 14):  # pragma: lax no cover
    pytest.skip('temporalio sandbox is incompatible with Python 3.14', allow_module_level=True)

pytestmark: list[pytest.MarkDecorator] = [
    pytest.mark.temporal,
    pytest.mark.xdist_group(name='temporal-durability'),
]

executions: list[str] = []
models = make_models(executions)
agent = make_agent(models, TemporalDurability(models=models), executions)


@workflow.defn
class OutputFallbackWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self) -> list[str | None]:
        result = await agent.run('Answer.')
        return [m.model_name for m in result.all_messages() if isinstance(m, ModelResponse)]


async def test_replay_reconstructs_selection(client: Client) -> None:
    executions.clear()
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types', 'coverage')
    )
    workflow_id = uuid4().hex
    async with Worker(client, task_queue='output-fallback', workflows=[OutputFallbackWorkflow], workflow_runner=runner):
        result = await client.execute_workflow(
            OutputFallbackWorkflow.run,
            id=workflow_id,
            task_queue='output-fallback',
            execution_timeout=timedelta(seconds=60),
        )
    history = await client.get_workflow_handle(workflow_id).fetch_history()
    assert result == ['primary', 'primary', 'backup']
    replayer = Replayer(workflows=[OutputFallbackWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=runner)
    replay = await replayer.replay_workflow(history, raise_on_replay_failure=False)
    assert replay.replay_failure is None
    assert executions == ['primary', 'lookup', 'primary', 'backup']
