"""The workspace check of `CapabilityCreation` under `TemporalDurability`, against a real sandboxed worker.

Not a VCR test: it's about where the check's I/O runs. Workflow code can neither read the process's directory
nor start a thread, so the check does both through units. These tests start a local Temporal dev server via
`WorkflowEnvironment.start_local()`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

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
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.capability_creation import CapabilityCreation
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal')]

TEMPORAL_PORT = 7262  # avoid conflict with the other Temporal suites
TASK_QUEUE = 'pydantic-ai-harness-capability-creation-queue'
BASE_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)
# See tests/spend/test_temporal.py for why these modules pass through.
_RESTRICTIONS = SandboxRestrictions.default.with_passthrough_modules('coverage', 'annotated_types')


def _answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('ran')])


# Relative, so the check has to read the process's directory. The model authors nothing, so it's never created.
_DIRECTORY = Path('.authored')

operator_agent = Agent(
    FunctionModel(_answer),
    name='operator_agent',
    deps_type=type(None),
    capabilities=[
        LocalWorkspace[None]('.'),
        CapabilityCreation[None](_DIRECTORY),
        TemporalDurability[None](activity_config=BASE_ACTIVITY_CONFIG),
    ],
)
lower_agent = Agent(
    FunctionModel(_answer),
    name='lower_agent',
    deps_type=type(None),
    capabilities=[
        LocalWorkspace[None]('.', path_authority={str(_DIRECTORY): 'none'}),
        CapabilityCreation[None](_DIRECTORY),
        TemporalDurability[None](activity_config=BASE_ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class CreationWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        agent = operator_agent if name == operator_agent.name else lower_agent
        try:
            return (await agent.run('go')).output
        except UserError as exc:
            return str(exc)


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    """Temporal's Python SDK runs on asyncio."""
    return 'asyncio'


@pytest.fixture(scope='module')
async def client() -> AsyncIterator[Client]:
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        port=TEMPORAL_PORT,
        dev_server_extra_args=['--dynamic-config-value', 'frontend.enableServerVersionCheck=false'],
        download_dest_dir=temporal_dev_server_cache_dir(),
    ):
        yield await Client.connect(f'localhost:{TEMPORAL_PORT}', plugins=[PydanticAIPlugin()])


async def _execute(client: Client, name: str) -> str:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CreationWorkflow],
        plugins=[AgentPlugin(operator_agent), AgentPlugin(lower_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_RESTRICTIONS),
    ):
        return await client.execute_workflow(
            CreationWorkflow.run,
            name,
            id=f'test_capability_creation_{name}',
            task_queue=TASK_QUEUE,
        )


async def test_runs_at_operator_authority(client: Client) -> None:
    assert await _execute(client, 'operator_agent') == 'ran'


async def test_refuses_below_operator_authority(client: Client) -> None:
    assert "has `'none'` authority" in await _execute(client, 'lower_agent')
