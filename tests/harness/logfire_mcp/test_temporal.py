"""Temporal composition test for `LogfireMCP`'s per-run authentication."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.logfire_mcp import LogfireMCP
from tests.harness.conftest import ignore_source_reads_left_open

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal'), ignore_source_reads_left_open]

TEMPORAL_PORT = 7246  # avoid conflict with the code_mode and spend suites
TASK_QUEUE = 'pydantic-ai-harness-logfire-mcp-queue'

# `coverage` and `annotated_types` are imported lazily while tracing and validating workflow code,
# which Temporal otherwise reports as imported after initial workflow load.
_SANDBOXED = SandboxRestrictions.default.with_passthrough_modules('coverage', 'annotated_types')


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


def _user_token(ctx: RunContext[str]) -> str:
    return ctx.deps


# The URL is set by the test, once the local server is up.
per_user = LogfireMCP[str](auth=_user_token, include_instructions=False)
per_user_agent = Agent(
    TestModel(),
    name='logfire_mcp_per_user_agent',
    deps_type=str,
    capabilities=[
        per_user,
        TemporalDurability[str](
            activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=60), retry_policy=RetryPolicy(maximum_attempts=1)
            )
        ),
    ],
)


@workflow.defn
class PerUserWorkflow:
    @workflow.run
    async def run(self, token: str) -> str:
        return (await per_user_agent.run('Who am I?', deps=token)).output


async def test_auth_function_runs_under_temporal(client: Client, whoami_url: str) -> None:
    per_user.url = whoami_url
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[PerUserWorkflow],
        plugins=[AgentPlugin(per_user_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_SANDBOXED),
    ):
        output = await client.execute_workflow(
            PerUserWorkflow.run,
            'alice-token',
            id='test_logfire_mcp_temporal_per_user',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=25),
        )

    assert output == '{"whoami":"Bearer alice-token"}'
