"""Workspaces under `TemporalDurability`, against a real sandboxed worker.

Not VCR tests: the behavior under test is where each workspace call runs (an activity, or directly
inside one), which needs the Temporal server's history rather than a provider recording. The shared
scenarios live in `workspace_scenarios.py`; this module runs them in a Temporal workflow and adds what
only Temporal has. The fake provider's environments live in the worker process: the sandboxed workflow
re-imports the scenario module and can construct a backend, but only an activity can reach one.
"""

from __future__ import annotations

import sys
import uuid
from datetime import timedelta
from importlib.machinery import ModuleSpec
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import (
    ReadOnlyWorkspace,
    Workspace,
    WorkspaceOutputLimitError,
    WorkspaceReadOnlyError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from ...workspace_fakes import InMemoryProvider
from ..workspace_scenarios import SCENARIOS, Check, ScenarioFailed, cases, scenario_agents

try:
    from temporalio import activity, workflow
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]
    from temporalio.api.enums.v1 import EventType
    from temporalio.api.failure.v1 import Failure
    from temporalio.client import Client, WorkflowExecutionStatus, WorkflowFailureError, WorkflowHandle, WorkflowHistory
    from temporalio.common import RetryPolicy
    from temporalio.testing import ActivityEnvironment
    from temporalio.worker import Replayer, Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

    from pydantic_ai.durable_exec._workspace import WorkspaceCall
    from pydantic_ai.durable_exec.prefect import PrefectDurability
    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        PydanticAIPlugin,
        TemporalDurability,
        _workflow_runner,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.durable_exec.temporal._run_context import TemporalRunContext
    from pydantic_ai.durable_exec.temporal._toolset import with_non_retryable_errors
    from pydantic_ai.durable_exec.temporal._transports import _WorkspaceCallWire

except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)


# The 3.14 durable-exec CI leg takes this skip; every other leg falls through.
if sys.version_info >= (3, 14):  # pragma: lax no cover
    pytest.skip(
        'temporalio sandbox is incompatible with Python 3.14: '
        'sandbox module state accumulates across validation cycles causing import failures after ~22 workflows '
        '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
        allow_module_level=True,
    )

try:
    import logfire  # pyright: ignore[reportUnusedImport]  # noqa: F401
except ImportError:  # pragma: lax no cover
    pytest.skip('logfire not installed', allow_module_level=True)

try:
    import mcp  # pyright: ignore[reportUnusedImport]  # noqa: F401
except ImportError:  # pragma: lax no cover
    pytest.skip('mcp not installed', allow_module_level=True)

try:
    import openai  # pyright: ignore[reportUnusedImport]  # noqa: F401
except ImportError:  # pragma: lax no cover
    pytest.skip('openai not installed', allow_module_level=True)


with workflow.unsafe.imports_passed_through():
    from ..._inline_snapshot import snapshot

    # Loads `vcr`, which Temporal doesn't like without passing through the import
    from ._shared import (
        BASE_ACTIVITY_CONFIG,
        TASK_QUEUE,
        _workflow_failure_cause,  # pyright: ignore[reportPrivateUsage]
    )

pytestmark = [
    pytest.mark.filterwarnings('ignore::pydantic.PydanticDeprecatedSince20'),
    pytest.mark.xdist_group(name='temporal-workspace'),
]


@pytest.mark.parametrize('durability', [TemporalDurability, PrefectDurability])
def test_idless_capability_toolset_still_requires_an_id_without_a_workspace(
    durability: type[TemporalDurability] | type[PrefectDurability],
) -> None:
    with pytest.raises(UserError, match='unique `id`'):
        Agent(TestModel(), name='instructions_only', capabilities=[Capability(instructions='x'), durability()])


def test_temporal_runner_passes_installed_harness_through(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai.durable_exec import temporal

    runner = SandboxedWorkflowRunner()

    def installed(module: str) -> ModuleSpec | None:
        return ModuleSpec(module, loader=None) if module == 'pydantic_ai_harness' else None

    monkeypatch.setattr(temporal, 'find_spec', installed)
    configured = _workflow_runner(runner)
    assert isinstance(configured, SandboxedWorkflowRunner)
    assert 'pydantic_ai_harness' in configured.restrictions.passthrough_modules
    assert 'opentelemetry' in configured.restrictions.passthrough_modules

    def absent(module: str) -> ModuleSpec | None:
        return None

    monkeypatch.setattr(temporal, 'find_spec', absent)
    configured = _workflow_runner(runner)
    assert isinstance(configured, SandboxedWorkflowRunner)
    assert 'pydantic_ai_harness' not in configured.restrictions.passthrough_modules
    assert 'opentelemetry' not in configured.restrictions.passthrough_modules


def test_workspace_failures_do_not_retry_temporal_activities() -> None:
    policy = with_non_retryable_errors(RetryPolicy())
    assert {
        WorkspaceTimeoutError.__name__,
        WorkspaceOutputLimitError.__name__,
        WorkspaceReadOnlyError.__name__,
        WorkspaceUnavailableError.__name__,
    } <= set(policy.non_retryable_error_types or [])


# --- The shared scenarios, in a workflow ---------------------------------------------------------

provider = InMemoryProvider(in_unit=activity.in_activity)
agents = scenario_agents(lambda: TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG), prefix='', provider=provider)


@workflow.defn
class ScenarioWorkflow:
    @workflow.run
    async def run(self, name: str, arg: str) -> Any:
        info = workflow.info()
        return await SCENARIOS[name](agents, arg or None, f'{info.workflow_id}:{info.run_id}')


async def _execute(client: Client, name: str, arg: str | None = None) -> tuple[Any, WorkflowHistory]:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ScenarioWorkflow],
        plugins=[AgentPlugin(agent) for agent in agents.all()],
    ):
        handle = await client.start_workflow(
            ScenarioWorkflow.run,
            args=[name, arg or ''],
            id=f'{name}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )
        try:
            output = await handle.result()
        except WorkflowFailureError as error:
            cause = _workflow_failure_cause(error)
            raise ScenarioFailed(cause.type or '', cause.message) from error
        return output, await handle.fetch_history()


@pytest.mark.parametrize('check', [case for case in cases() if case.id != 'uncaught'])
async def test_workspace_scenario(client: Client, check: Check) -> None:
    provider.reset()

    async def run(name: str, arg: str | None) -> Any:
        return (await _execute(client, name, arg))[0]

    await check(run, agents)


async def test_an_uncaught_builtin_workspace_error_fails_the_workflow_task(client: Client) -> None:
    """Like any exception in workflow code, it fails the task, which Temporal retries until a fix is deployed."""
    provider.reset()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ScenarioWorkflow],
        plugins=[AgentPlugin(agent) for agent in agents.all()],
    ):
        handle = await client.start_workflow(
            ScenarioWorkflow.run, args=['uncaught', ''], id=f'uncaught-{uuid.uuid4()}', task_queue=TASK_QUEUE
        )
        try:
            with anyio.fail_after(30):  # hang guard
                while not (failures := await _workflow_task_failures(handle)):
                    await anyio.sleep(0.1)
            assert 'missing.txt' in failures[0].message
            assert failures[0].application_failure_info.type == 'FileNotFoundError'
            assert (await handle.describe()).status == WorkflowExecutionStatus.RUNNING
        finally:
            await handle.terminate()


async def _workflow_task_failures(handle: WorkflowHandle[Any, Any]) -> list[Failure]:
    return [
        event.workflow_task_failed_event_attributes.failure
        async for event in handle.fetch_history_events()
        if event.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED
    ]


def _activity_names(history: WorkflowHistory) -> list[str]:
    return [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.HasField('activity_task_scheduled_event_attributes')
    ]


async def test_workspace_calls_from_workflow_code_are_activities_and_replay_dispatches_nothing(client: Client) -> None:
    """Hooks and the result reach the workspace through activities; the tools' calls run inside theirs."""
    provider.reset()
    _, history = await _execute(client, 'fresh')
    assert _activity_names(history) == snapshot(
        [
            'agent__fresh__capability__workspace__call',
            'agent__fresh__capability__workspace__call',
            'agent__fresh__model_request',
            'agent__fresh__toolset__<agent>__call_tool',
            'agent__fresh__toolset__<agent>__call_tool',
            'agent__fresh__model_request',
            'agent__fresh__capability__workspace__call',
            'agent__fresh__capability__workspace__call',
        ]
    )
    log = list(provider.log)

    replay = await Replayer(workflows=[ScenarioWorkflow], plugins=[PydanticAIPlugin()]).replay_workflow(history)
    assert replay.replay_failure is None
    assert provider.log == log


# --- The activity side on its own ---------------------------------------------------------------


async def test_activity_refuses_a_ref_no_worker_capability_recognizes() -> None:
    """An `ensure` activity for a ref the worker's capabilities cannot rebuild fails with an explanation."""
    agent = Agent(TestModel(), name='ctx', capabilities=[provider.capability(), TemporalDurability()])
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    ensure = next(
        item
        for item in durability.temporal_activities
        if ActivityDefinition.must_from_callable(item).name == 'agent__ctx__capability__workspace__call'  # pyright: ignore[reportUnknownMemberType]
    )
    wire = _WorkspaceCallWire(
        call=WorkspaceCall(method='ensure'),
        ref=WorkspaceRef(provider='other', id='x'),
        serialized_run_context={'run_id': 'r', 'workspace_ref': {'provider': 'other', 'id': 'x'}},
    )

    with pytest.raises(UserError, match="No capability can supply the workspace 'x' from provider 'other'"):
        await ActivityEnvironment().run(ensure, wire, None)


def test_activity_run_context_rebuilds_the_workspace_from_the_serialized_ref() -> None:
    """The restore path on its own: a custom context that sets `workspace` wins, a missing capability explains."""
    agent = Agent(TestModel(), name='ctx', capabilities=[provider.capability(read_only=True), TemporalDurability()])
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    serialized = {'run_id': 'r', 'workspace_ref': {'provider': 'fake', 'id': 'seeded'}}

    restored = durability.deserialize_operation_run_context(serialized, None)
    assert isinstance(restored.workspace, ReadOnlyWorkspace)
    assert restored.workspace.ref == WorkspaceRef(provider='fake', id='seeded')

    class OwnWorkspace(TemporalRunContext[Any]):
        @classmethod
        def deserialize_run_context(cls, ctx: dict[str, Any], deps: Any) -> OwnWorkspace:
            return cls(**{**ctx, 'workspace': Workspace(provider.backend(None))}, deps=deps)

    own = Agent(
        TestModel(),
        name='ctx',
        capabilities=[provider.capability(), TemporalDurability(run_context_type=OwnWorkspace)],
    )
    own_durability = TemporalDurability.from_agent(own)
    assert own_durability is not None
    own_ctx = own_durability.deserialize_operation_run_context(serialized, None)
    assert type(own_ctx.workspace) is Workspace and own_ctx.workspace.ref is None

    no_ref = durability.deserialize_operation_run_context({'run_id': 'r'}, None)
    assert no_ref.workspace.ref is None

    foreign = durability.deserialize_operation_run_context(
        {'run_id': 'r', 'workspace_ref': {'provider': 'other', 'id': 'x'}}, None
    )
    assert foreign.workspace.ref is None
