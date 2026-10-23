"""`ExaAgent` under `TemporalDurability`: a worker that replays a run claims the calls the original did.

These tests start a local Temporal dev server via `WorkflowEnvironment.start_local()`.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from datetime import timedelta

import pytest
from pydantic import BaseModel

try:
    from exa_py.agent.types import AgentEffort, AgentEvent, AgentOutput, AgentRun
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Replayer, Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability
    from pydantic_ai_harness.exa import ExaAgent
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio or exa-py not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal')]

TEMPORAL_PORT = 7261  # avoid conflict with the other Temporal suites
TASK_QUEUE = 'pydantic-ai-harness-exa-agent-queue'
BASE_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)
# See tests/spend/test_temporal.py for why these modules pass through.
_RESTRICTIONS = SandboxRestrictions.default.with_passthrough_modules('coverage', 'annotated_types', 'exa_py')

_polls: list[str] = []


class _Runs:
    async def create(
        self,
        *,
        query: str,
        system_prompt: str | None = None,
        output_schema: dict[str, object] | type[BaseModel] | None = None,
        effort: AgentEffort | None = None,
        previous_run_id: str | None = None,
    ) -> AgentRun | AsyncGenerator[AgentEvent, None]:
        return AgentRun(id='run_1', status='queued')

    async def poll_until_finished(
        self, run_id: str, *, poll_interval: int = 1000, timeout_ms: int = 3600000
    ) -> AgentRun:
        _polls.append(run_id)
        return AgentRun(id=run_id, status='completed', output=AgentOutput(text='Done.'))


def _delegate(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Delegate one task to `exa_agent`, then answer."""
    if len([message for message in messages if isinstance(message, ModelRequest)]) > 1:
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('exa_agent', {'query': 'research this'})])


def _exa_agent_agent() -> Agent[None, str]:
    return Agent(
        FunctionModel(_delegate),
        name='exa_agent_agent',
        deps_type=type(None),
        capabilities=[ExaAgent[None](runs=_Runs()), TemporalDurability[None](activity_config=BASE_ACTIVITY_CONFIG)],
    )


exa_agent_agent = _exa_agent_agent()
# What a worker in another process builds from the same code: a new `ExaAgent` instance.
recovering_agent = _exa_agent_agent()


@workflow.defn(name='ExaAgentWorkflow')
class ExaAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await exa_agent_agent.run(prompt)).output


@workflow.defn(name='ExaAgentWorkflow')
class RecoveringExaAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await recovering_agent.run(prompt)).output


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


async def test_a_new_worker_replays_the_run_and_claims_its_deferred_call(client: Client) -> None:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ExaAgentWorkflow],
        plugins=[AgentPlugin(exa_agent_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_RESTRICTIONS),
    ):
        handle = await client.start_workflow(
            ExaAgentWorkflow.run,
            'research this',
            id='test_exa_agent_temporal_replay',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=25),
        )
        assert await handle.result() == 'done'
    history = await handle.fetch_history()
    assert _polls == ['run_1']

    replayer = Replayer(
        workflows=[RecoveringExaAgentWorkflow],
        plugins=[PydanticAIPlugin(), AgentPlugin(recovering_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_RESTRICTIONS),
    )
    replay = await replayer.replay_workflow(history, raise_on_replay_failure=False)
    assert replay.replay_failure is None
    assert _polls == ['run_1'], 'the replay polled the Exa run again'
