"""Temporal composition test for `LogfireMCP`'s current-time instruction.

The instruction reads the clock, which Temporal's workflow sandbox restricts, so it runs as a
durable operation. This test starts a local Temporal dev server via `WorkflowEnvironment.start_local()`.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

try:
    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
    from temporalio.workflow import ActivityConfig
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.logfire_mcp import LogfireMCP

pytestmark = pytest.mark.anyio

TEMPORAL_PORT = 7246  # avoid conflict with the code_mode and spend suites
TASK_QUEUE = 'pydantic-ai-harness-logfire-mcp-queue'

# `coverage` and `annotated_types` are imported lazily while tracing and validating workflow code,
# which Temporal otherwise reports as imported after initial workflow load.
_SANDBOXED = SandboxRestrictions.default.with_passthrough_modules('coverage', 'annotated_types')


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    """Temporal's Python SDK runs on asyncio."""
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


def _echo_instructions(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=info.instructions or '')])


# MCP's test server leaves its lifespan annotation unresolved with some pydantic-settings versions. The
# server is built at import, where the suite's `pytestmark` filters do not apply yet.
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', "Field 'lifespan' has an incomplete definition")
    server = FastMCP('provider')

# Module level, as Temporal requires.
agent = Agent(
    FunctionModel(_echo_instructions),
    name='logfire_mcp_agent',
    deps_type=type(None),
    capabilities=[
        LogfireMCP[None](client=server),
        TemporalDurability[None](
            activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=60), retry_policy=RetryPolicy(maximum_attempts=1)
            )
        ),
    ],
)


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


@workflow.defn
class LogfireWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await agent.run(prompt)).output


async def test_current_time_is_read_in_an_activity(client: Client) -> None:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[LogfireWorkflow],
        plugins=[AgentPlugin(agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_SANDBOXED),
    ):
        output = await client.execute_workflow(
            LogfireWorkflow.run,
            'Recent errors',
            id='test_logfire_mcp_temporal',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=25),
        )

    assert 'Current UTC time is `' in output


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
