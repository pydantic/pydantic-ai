"""Temporal composition tests for `SpendLimits`.

The clock, reads, and accrual execute as activities through `TemporalDurability`, so the
workflow sandbox does not need `pydantic_ai_harness` passed through and replay does not repeat
the store mutation.

These tests start a local Temporal dev server via `WorkflowEnvironment.start_local()` -- the
Temporal SDK downloads and runs `temporalite` automatically.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from decimal import Decimal

import pytest

try:
    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        PydanticAIPlugin,
        TemporalDurability,
    )
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
    from temporalio.workflow import ActivityConfig
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from pydantic_ai_harness.spend import Budget, SpendLimitExceeded, SpendLimits

pytestmark = pytest.mark.anyio

TEMPORAL_PORT = 7245  # avoid conflict with the code_mode suite
TASK_QUEUE = 'pydantic-ai-harness-spend-queue'
BASE_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)

# Both runners pass through `coverage`, which imports parser modules lazily while tracing
# workflow code, and `annotated_types`, which pydantic imports
# lazily while building the type adapter for an activity result -- either one trips Temporal's
# "imported after initial workflow load" warning, which `filterwarnings = ['error']` turns into
# a workflow task failure. Neither is specific to this capability; `pydantic_ai_harness` is the
# only variable between the runners.
_SANDBOXED = SandboxRestrictions.default.with_passthrough_modules('coverage', 'annotated_types')
_PASSTHROUGH = _SANDBOXED.with_passthrough_modules('pydantic_ai_harness')


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


def _priced_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """A response the registry can price, so the budget moves by a known amount."""
    return ModelResponse(
        parts=[TextPart(content='ok')],
        model_name='gpt-4o',
        usage=RequestUsage(input_tokens=1000, output_tokens=1000),
    )


def _fixed_price(response: ModelResponse) -> Decimal:
    """A price that does not depend on the registry, so the assertions are exact."""
    return Decimal('2')


# Module level, as Temporal requires. Two agents so the two workflows do not share a counter.
counting_limits = SpendLimits[None](budgets=[Budget(usd=Decimal('100'), window='day')], price=_fixed_price)
counting_agent = Agent(
    FunctionModel(_priced_model),
    name='spend_counting_agent',
    deps_type=type(None),
    capabilities=[counting_limits, TemporalDurability[None](activity_config=BASE_ACTIVITY_CONFIG)],
)

exhausted_agent = Agent(
    FunctionModel(_priced_model),
    name='spend_exhausted_agent',
    deps_type=type(None),
    capabilities=[
        SpendLimits[None](budgets=[Budget(usd=Decimal('1'), window='day')], price=_fixed_price),
        TemporalDurability[None](activity_config=BASE_ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class CountingWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await counting_agent.run(prompt)).output


@workflow.defn
class ExhaustedWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        await exhausted_agent.run(prompt)
        try:
            await exhausted_agent.run(prompt)
        except SpendLimitExceeded as error:
            return str(error)
        return 'the budget did not refuse the second request'  # pragma: no cover


async def test_the_default_sandbox_dispatches_the_clock_to_an_activity(client: Client) -> None:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CountingWorkflow],
        plugins=[AgentPlugin(counting_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_SANDBOXED),
    ):
        output = await client.execute_workflow(
            CountingWorkflow.run,
            'hello',
            id='test_spend_temporal_sandbox',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=25),
        )

    assert output == 'ok'
    assert (await counting_limits.status())[0].spent.usd == Decimal('2')


async def test_budgets_accumulate_inside_a_forced_workflow(client: Client) -> None:
    """Overriding the advice, the hooks do run and `status()` reads the counter workflow-side.

    Pins what the unsupported configuration does within a single execution. It says nothing
    about replay, where this same accrual runs again.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CountingWorkflow],
        plugins=[AgentPlugin(counting_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_PASSTHROUGH),
    ):
        output = await client.execute_workflow(
            CountingWorkflow.run,
            'hello',
            id='test_spend_temporal_counting',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=25),
        )

    assert output == 'ok'


async def test_the_gate_refuses_a_second_run_inside_a_forced_workflow(client: Client) -> None:
    """The counter outlives one run inside the workflow, which is what makes the gate a gate.

    Same caveat as above: one execution, nothing asserted about replay.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ExhaustedWorkflow],
        plugins=[AgentPlugin(exhausted_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_PASSTHROUGH),
    ):
        raised = await client.execute_workflow(
            ExhaustedWorkflow.run,
            'hello',
            id='test_spend_temporal_exhausted',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=25),
        )

    assert 'exhausted for this day' in raised
