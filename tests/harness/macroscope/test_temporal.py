"""`Macroscope` under `TemporalDurability`: the review runs as an activity on a real Temporal dev server."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.macroscope import Macroscope, MacroscopeReview
from tests.harness._temporal import skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal'), skip_temporal_sandbox_on_314]

_ISSUE_LINE = (
    'issue_event={"issue_id":"i1","sequence":1,"path":"a.py","line":4,'
    '"severity":"medium","category":"REVIEW_TYPE_CORRECTNESS","body":"only checks completion"}'
)


@workflow.defn
class ReviewWorkflow(PydanticAIWorkflow):
    agent: Agent[None, str]

    @workflow.run
    async def run(self) -> list[object]:
        result = await self.agent.run('review please')
        return [p.content for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]


def _model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if not any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
        return ModelResponse(parts=[ToolCallPart('run_macroscope_review', {})])
    return ModelResponse(parts=[TextPart('done')])


async def test_review_runs_as_a_temporal_activity(tmp_path: Path) -> None:
    command = tmp_path / 'macroscope'
    command.write_text(f"#!/bin/sh\nprintf '%s\\n' review_id=rev-1 '{_ISSUE_LINE}' issue_status=completed >&2\n")
    command.chmod(0o755)
    ReviewWorkflow.agent = Agent(
        FunctionModel(_model),
        name='durable_macroscope',
        capabilities=[LocalWorkspace(tmp_path), Macroscope(command=str(command), base='main'), TemporalDurability()],
    )
    ReviewWorkflow.__pydantic_ai_agents__ = [ReviewWorkflow.agent]
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        # The test's dynamically configured agent lives in this test module.
        runner = SandboxedWorkflowRunner(
            restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types')
        )
        async with Worker(client, task_queue='macroscope', workflows=[ReviewWorkflow], workflow_runner=runner):
            returns = await client.execute_workflow(
                ReviewWorkflow.run, id=uuid4().hex, task_queue='macroscope', execution_timeout=timedelta(seconds=120)
            )
    assert len(returns) == 1
    review = MacroscopeReview.model_validate(returns[0])
    assert (review.review_id, review.status, [i.issue_id for i in review.issues]) == ('rev-1', 'completed', ['i1'])
