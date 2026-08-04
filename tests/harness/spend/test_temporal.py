"""Temporal composition tests for `SpendLimits`.

`SpendLimits` is **not supported** inside a Temporal workflow: its hooks run in workflow code
while only the model request is the activity, so Temporal replays the accrual and a window
counts the same response more than once. See
<https://github.com/pydantic/pydantic-ai-harness/issues/531>.

What is pinned here is that behaviour being stated rather than worked around. The default
sandbox refuses the clock these hooks read, and the capability translates that into what it
means instead of into the setting that silences it. The remaining two tests drive the forced
configuration -- module passed through the sandbox -- to pin what a caller who overrides the
advice actually gets: the counters do accumulate and the gate does refuse, within one workflow
execution. Neither asserts anything about replay, which is the part that is unsafe.

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
from pydantic_ai.exceptions import UserError
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

# The two runners the tests contrast. Both pass through `coverage`, which imports parser
# modules lazily while tracing workflow code, and `annotated_types`, which pydantic imports
# lazily while building the type adapter for an activity result -- either one trips Temporal's
# "imported after initial workflow load" warning, which `filterwarnings = ['error']` turns into
# a workflow task failure. Neither is specific to this capability; `pydantic_ai_harness` is the
# only variable under test.
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
        await counting_agent.run(prompt)
        statuses = await counting_limits.status()
        return str(statuses[0].spent.usd)


@workflow.defn
class SandboxProbeWorkflow:
    """Reports what the capability raises when the sandbox is left at its defaults."""

    @workflow.run
    async def run(self, prompt: str) -> str:
        try:
            await exhausted_agent.run(prompt)
        except UserError as error:
            return str(error)
        return 'the sandbox allowed the clock'  # pragma: no cover


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


async def test_the_default_sandbox_refuses_the_clock_and_names_the_real_problem(client: Client) -> None:
    """The sandbox refuses a symptom; the message has to name the replay, not the passthrough.

    A caller told only how to silence the error would pass the module through and get a
    counter Temporal replays, which is worse than the error they started with.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SandboxProbeWorkflow],
        plugins=[AgentPlugin(exhausted_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_SANDBOXED),
    ):
        message = await client.execute_workflow(
            SandboxProbeWorkflow.run,
            'hello',
            id='test_spend_temporal_sandbox',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=25),
        )

    assert 'not safe to run inside a Temporal workflow' in message
    assert 'exhausted()' in message
    assert 'issues/531' in message
    assert 'with_passthrough_modules' not in message


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
        spent = await client.execute_workflow(
            CountingWorkflow.run,
            'hello',
            id='test_spend_temporal_counting',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=25),
        )

    assert Decimal(spent) == Decimal('2')


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
