"""Temporal coverage for what `AgentControl` does from inside a workflow.

Registration used to be the one thing an agent could not do from a workflow: it wrote a variable from
a background thread, which is not replay-safe, so the capability skipped it and the docs told you to
run the agent outside the workflow once. A hint is a span, and emitting one is exactly what a run
inside a workflow already does for everything else -- so a workflow-only agent now registers like any
other.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from logfire.testing import CaptureLogfire

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import AgentControl

TEMPORAL_PORT = 7245
TASK_QUEUE = 'pydantic-ai-harness-agent-control-queue'


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(scope='module')
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        port=TEMPORAL_PORT,
        dev_server_extra_args=['--dynamic-config-value', 'frontend.enableServerVersionCheck=false'],
    ) as env:
        yield env


@pytest.fixture
async def client(temporal_env: WorkflowEnvironment) -> Client:
    return await Client.connect(f'localhost:{TEMPORAL_PORT}', plugins=[PydanticAIPlugin()])


def _model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('done')])


agent_control_agent = Agent(
    FunctionModel(_model),
    name='agent_control_temporal_agent',
    capabilities=[AgentControl(), TemporalDurability()],
)


def _workflow_runner() -> SandboxedWorkflowRunner:
    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            'annotated_types',
            'coverage',
            'logfire',
            'opentelemetry',
            'pydantic_ai_harness.logfire',
            'pydantic_graph',
        )
    )


@workflow.defn
class AgentControlWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return str((await agent_control_agent.run(prompt)).output)


async def test_agent_control_reports_an_unconfigured_agent_from_a_temporal_workflow(
    client: Client, capfire: CaptureLogfire
) -> None:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AgentControlWorkflow],
        plugins=[AgentPlugin(agent_control_agent)],
        workflow_runner=_workflow_runner(),
    ):
        result = await client.execute_workflow(
            AgentControlWorkflow.run,
            args=['hello'],
            id='test_agent_control_temporal_config_hint',
            task_queue=TASK_QUEUE,
        )
    assert result == 'done'

    hints = [
        span['attributes']
        for span in capfire.exporter.exported_spans_as_dict()
        if span['name'] == 'agent_control_config_hint'
    ]
    assert [attributes['agent_control.variable_name'] for attributes in hints] == [
        'agent__agent_control_temporal_agent'
    ]
