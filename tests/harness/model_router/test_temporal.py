"""`ModelRouter` under `TemporalDurability`: the router request runs in an activity, and its `UserError` fails the run."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.model_router import ModelChoice, ModelRouter

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from tests.harness._temporal import ignore_source_reads_left_open, skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [
    pytest.mark.temporal,
    pytest.mark.xdist_group(name='harness-temporal'),
    skip_temporal_sandbox_on_314,
    ignore_source_reads_left_open,
]


def fast_answer(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('fast')])


def route(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if 'files' in str(messages[0]):
        raise UserError('Files are not supported by this model.')
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'choice': 'fast'})])


capable = TestModel()
fast = FunctionModel(fast_answer)
agent = Agent(
    capable,
    name='model_router_temporal',
    deps_type=type(None),
    capabilities=[
        ModelRouter[None](
            choices={
                'fast': ModelChoice(fast, 'Use for lookups.'),
                'capable': ModelChoice(capable, 'Use for difficult work.'),
            },
            router_model=FunctionModel(route),
            default='capable',
        ),
        TemporalDurability[None](models={'fast': fast}),
    ],
)


@workflow.defn
class RoutedWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await agent.run(prompt)).output


async def test_routes_in_an_activity_and_a_router_user_error_fails_the_workflow() -> None:
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types', 'coverage')
    )
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        async with Worker(client, task_queue='model-router', workflows=[RoutedWorkflow], workflow_runner=runner):

            async def run(prompt: str) -> str:
                return await client.execute_workflow(
                    RoutedWorkflow.run,
                    prompt,
                    id=uuid4().hex,
                    task_queue='model-router',
                    execution_timeout=timedelta(seconds=60),
                )

            output = await run('Find the capital of France')
            with pytest.raises(WorkflowFailureError) as exc_info:
                await run('Read these files')

    assert output == 'fast'
    assert 'Files are not supported by this model.' in str(exc_info.value.cause)
