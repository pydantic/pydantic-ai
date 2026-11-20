"""`ReportToolErrors` under `TemporalDurability`: a failed tool activity stays with the workflow, as without it."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
from temporalio.workflow import ActivityConfig

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness import ReportToolErrors
from tests.harness._temporal import ignore_source_reads_left_open, skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [
    pytest.mark.temporal,
    pytest.mark.xdist_group(name='harness-temporal'),
    skip_temporal_sandbox_on_314,
    ignore_source_reads_left_open,
]


def dividing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('divide', {'a': 1, 'b': 0}, tool_call_id='c1')])
    return ModelResponse(parts=[TextPart('done')])  # pragma: no cover


def divide(a: int, b: int) -> float:
    return a / b


agent = Agent(
    FunctionModel(dividing_model),
    name='report_tool_errors',
    tools=[divide],
    capabilities=[
        ReportToolErrors(),
        TemporalDurability(
            activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=10), retry_policy=RetryPolicy(maximum_attempts=1)
            )
        ),
    ],
)


@workflow.defn
class DividingWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self) -> str:
        return (await agent.run('Divide')).output  # pragma: lax no cover


async def test_failed_tool_activity_still_fails_the_workflow() -> None:
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types', 'coverage')
    )
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        async with Worker(
            client, task_queue='report-tool-errors', workflows=[DividingWorkflow], workflow_runner=runner
        ):
            with pytest.raises(WorkflowFailureError) as exc_info:
                await client.execute_workflow(
                    DividingWorkflow.run,
                    id=uuid4().hex,
                    task_queue='report-tool-errors',
                    execution_timeout=timedelta(seconds=60),
                )
    activity_error = exc_info.value.cause
    assert isinstance(activity_error, ActivityError)
    assert isinstance(activity_error.cause, ApplicationError)
    assert activity_error.cause.type == 'ZeroDivisionError'
