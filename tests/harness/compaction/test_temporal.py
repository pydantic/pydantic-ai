"""Under Temporal, failed summary operations still arrive as `ActivityError` (#12291), so automatic recovery cannot catch them yet."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.compaction import SummarizingCompaction
from tests.harness._temporal import ignore_source_reads_left_open, skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [
    pytest.mark.temporal,
    pytest.mark.xdist_group(name='harness-temporal'),
    skip_temporal_sandbox_on_314,
    ignore_source_reads_left_open,
]


def _summary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if 'api_failure' in str(messages):
        raise ModelAPIError('summary', 'unavailable')
    raise UnexpectedModelBehavior('invalid summary')


def _answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
    return ModelResponse(parts=[TextPart('done')])


agent = Agent(
    FunctionModel(_answer),
    name='failed_summary',
    capabilities=[
        SummarizingCompaction(
            model=FunctionModel(_summary), max_messages=1, keep_messages=1, preserve_first_user_message=False
        ),
        TemporalDurability(
            activity_config={
                'start_to_close_timeout': timedelta(seconds=10),
                'retry_policy': RetryPolicy(maximum_attempts=1),
            }
        ),
    ],
)


@workflow.defn
class FailedSummaryWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, failure: str) -> str:
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(failure)]),
            ModelResponse(parts=[TextPart('earlier answer')]),
        ]
        return (await agent.run('Next?', message_history=history)).output


async def test_temporal_summary_failures_still_propagate_as_activity_errors() -> None:
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types', 'coverage')
    )
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        async with Worker(
            client, task_queue='failed-summary', workflows=[FailedSummaryWorkflow], workflow_runner=runner
        ):
            for failure, error_type in [
                ('invalid_summary', 'UnexpectedModelBehavior'),
                ('api_failure', 'ModelAPIError'),
            ]:
                with pytest.raises(WorkflowFailureError) as caught:
                    await client.execute_workflow(
                        FailedSummaryWorkflow.run,
                        failure,
                        id=uuid4().hex,
                        task_queue='failed-summary',
                        execution_timeout=timedelta(seconds=30),
                    )
                assert isinstance(caught.value.cause, ActivityError)
                assert isinstance(caught.value.cause.cause, ApplicationError)
                assert caught.value.cause.cause.type == error_type
