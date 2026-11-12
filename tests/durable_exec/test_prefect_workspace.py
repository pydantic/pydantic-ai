"""Workspaces under `PrefectDurability`, against the Prefect test harness.

Not VCR tests: the behavior under test is where each workspace call runs (a Prefect task, or
directly inside one) and what a flow retry replays, which a provider recording could not show.
The shared scenarios live in `workspace_scenarios.py`; this module runs them in a Prefect flow and
adds what only Prefect has.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, WrapperCapability, durable_operation
from pydantic_ai.durable_exec._workspace import WorkspaceCall
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import WorkspaceTimeoutError

from ..workspace_fakes import InMemoryProvider
from .workspace_scenarios import SCENARIOS, Check, ScenarioFailed, cases, scenario_agents, tool_runs

try:
    from prefect import flow, task
    from prefect.context import FlowRunContext, TaskRunContext

    from pydantic_ai.durable_exec.prefect import PrefectDurability, TaskConfig
except ImportError:  # pragma: lax no cover
    pytest.skip('Prefect is not installed', allow_module_level=True)


pytestmark = [pytest.mark.xdist_group(name='prefect'), pytest.mark.usefixtures('prefect_test_server')]


@pytest.fixture(autouse=True)
def blockbuster_excluded_modules() -> tuple[str, ...]:
    """Prefect's `@flow` constructor synchronously inspects its decorated function's source."""
    return ('pydantic_ai.durable_exec.prefect',)


task_names: list[str] = []


def _workspace_tasks() -> list[str]:
    """The method of each workspace task, in the order the flow ran them."""
    return [name.split(':')[-1] for name in task_names if name.startswith('Capability: workspace.call')]


@pytest.fixture(autouse=True)
def record_task_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Record every task run's name at the point its body starts, in the order the flow ran them."""
    from pydantic_ai.durable_exec.prefect import _operation_backend

    task_names.clear()
    original = _operation_backend.PrefectOperationBackend.execute

    async def execute(self: Any, **kwargs: Any) -> object:
        call = kwargs['cache_key'][0]
        task_names.append(f'{kwargs["name"]}:{call.method}' if isinstance(call, WorkspaceCall) else kwargs['name'])
        return await original(self, **kwargs)

    monkeypatch.setattr(_operation_backend.PrefectOperationBackend, 'execute', execute)


# A tool runs inside its own task and calls the workspace directly; workflow code calls it through tasks.
provider = InMemoryProvider(in_unit=lambda: TaskRunContext.get() is not None)
agents = scenario_agents(PrefectDurability, prefix='prefect_', provider=provider)


@flow
async def scenario_flow(name: str, arg: str | None) -> Any:
    context = FlowRunContext.get()
    assert context is not None and context.flow_run is not None
    return await SCENARIOS[name](agents, arg, str(context.flow_run.id))


async def run_scenario(name: str, arg: str | None) -> Any:
    try:
        return await scenario_flow(name, arg)
    except Exception as error:
        raise ScenarioFailed(type(error).__name__, str(error)) from error


@pytest.mark.parametrize('check', cases())
async def test_workspace_scenario(check: Check) -> None:
    provider.reset()
    await check(run_scenario, agents)


async def test_prefect_flow_retry_replays_workspace_tasks_and_run_ids() -> None:
    provider.reset()
    tool_runs.clear()
    attempts: list[dict[str, Any]] = []

    @flow(retries=1)
    async def fail_after_the_run_once() -> dict[str, Any]:
        output = await SCENARIOS['fresh'](agents, None, '')
        second = await agents.plain.run('Again.')
        attempts.append({'output': output, 'run_id': second.run_id})
        if len(attempts) == 1:
            raise RuntimeError('boom')
        return output

    await fail_after_the_run_once()

    # The retry replayed every task the first attempt recorded, with the same run ID, and attached to
    # the environments the first attempt created (one per agent) instead of creating more.
    assert attempts[0] == attempts[1]
    assert tool_runs == ['write_left']
    assert provider.log == snapshot(['create:env-1', 'create:env-2'])
    assert _workspace_tasks() == snapshot(
        [
            'ensure',
            'write_bytes',
            'list_dir',
            'read_bytes',
            'ensure',
            'ensure',
            'write_bytes',
            'list_dir',
            'read_bytes',
            'ensure',
        ]
    )


async def test_prefect_default_ids_without_a_workspace_survive_a_flow_retry() -> None:
    agent = Agent(TestModel(), name='prefect_run_id', capabilities=[PrefectDurability()])
    attempts: list[list[tuple[str, str]]] = []

    @flow(retries=1)
    async def run() -> None:
        results = [await agent.run('First.'), await agent.run('Second.')]
        attempts.append([(result.run_id, result.conversation_id) for result in results])
        if len(attempts) == 1:
            raise RuntimeError('retry')

    await run()
    # Each agent run in the flow gets its own IDs, and the retry gives each the same ones again.
    first, second = attempts[0]
    assert attempts[1] == attempts[0]
    assert len({*first, *second}) == 4
    assert first[0].endswith(':0') and second[0].endswith(':1')
    # Outside a flow, the agent keeps generating fresh UUID7s.
    outside = await agent.run('Outside.')
    assert [uuid.UUID(outside.run_id).version, uuid.UUID(outside.conversation_id).version] == [7, 7]


@pytest.mark.parametrize(
    'operation_name',
    [
        pytest.param(None, id='without-custom-operation'),
        pytest.param('record', id='custom-operation'),
        pytest.param('call:run:8:left-run', id='custom-operation-counter-name'),
    ],
)
async def test_prefect_concurrent_workspace_runs_keep_their_environments_on_retry(operation_name: str | None) -> None:
    """A cached preparation task can reverse arrival order; workspace keys follow explicit run IDs."""
    readiness_wait_timeout = 30
    provider = InMemoryProvider(in_unit=lambda: TaskRunContext.get() is not None)
    right_finished = anyio.Event()
    left_replayed = anyio.Event()
    arrivals: list[str] = []
    attempts: list[list[tuple[str, str, str]]] = []
    tool_runs: list[str] = []
    operation_runs: list[str] = []

    class SideEffectCapability(AbstractCapability[None]):
        id = 'workspace'

        async def before_run(self, ctx: RunContext[None]) -> None:
            if ctx.prompt == 'right':
                await self.record(ctx)

        @durable_operation(operation_name or 'record')
        async def record(self, ctx: RunContext[None]) -> None:
            assert ctx.run_id is not None
            operation_runs.append(ctx.run_id)

    async def mark(ctx: RunContext[None]) -> str:
        assert isinstance(ctx.prompt, str)
        tool_runs.append(ctx.prompt)
        await ctx.workspace.write_text('owner.txt', ctx.prompt)
        return 'ok'

    capabilities: list[AbstractCapability[None]] = [provider.capability()]
    if operation_name is not None:
        capabilities.append(SideEffectCapability())
    capabilities.append(PrefectDurability())
    agent = Agent(
        TestModel(call_tools=['mark']),
        deps_type=type(None),
        name='concurrent_workspaces',
        tools=[mark],
        capabilities=capabilities,
    )

    @task(persist_result=True)
    async def prepare(label: str) -> str:
        if label == 'left':
            with anyio.fail_after(readiness_wait_timeout):
                await right_finished.wait()
        return label

    async def branch(label: str, results: dict[str, tuple[str, str, str]]) -> None:
        if attempts and label == 'right':
            with anyio.fail_after(readiness_wait_timeout):
                await left_replayed.wait()
        await prepare(label)
        arrivals.append(label)
        result = await agent.run(label, run_id=f'{label}-run')
        workspace = result.workspace
        assert workspace.ref is not None
        # These calls run after the agent has left its ambient run context.
        owner = await workspace.read_text('owner.txt')
        if label == 'right':
            right_finished.set()
        elif attempts:
            left_replayed.set()
        results[label] = result.run_id, workspace.ref.id, owner

    @flow(retries=1)
    async def retry() -> None:
        results: dict[str, tuple[str, str, str]] = {}
        async with anyio.create_task_group() as group:
            group.start_soon(branch, 'left', results)
            group.start_soon(branch, 'right', results)
        attempts.append([results['left'], results['right']])
        if len(attempts) == 1:
            raise RuntimeError('retry')

    await retry()

    assert arrivals == ['right', 'left', 'left', 'right']
    assert attempts == [[('left-run', 'env-2', 'left'), ('right-run', 'env-1', 'right')]] * 2
    assert tool_runs == ['right', 'left']
    assert operation_runs == (['right-run'] if operation_name is not None else [])
    assert provider.environments == {'env-1': {'/remote/owner.txt': b'right'}, 'env-2': {'/remote/owner.txt': b'left'}}
    assert provider.log == ['create:env-1', 'create:env-2']


async def test_prefect_default_ids_reach_through_a_wrapped_durability_capability() -> None:
    agent = Agent(TestModel(), name='prefect_wrapped_run_id', capabilities=[WrapperCapability(PrefectDurability())])

    @flow
    async def run() -> tuple[str, str]:
        result = await agent.run('Hello.')
        return result.run_id, result.conversation_id

    run_id, conversation_id = await run()
    assert run_id.endswith(':0')
    assert uuid.UUID(conversation_id).version == 5


async def test_prefect_repeated_identical_reads_are_not_served_from_cache() -> None:
    provider.reset()
    attempts: list[tuple[str, str]] = []

    @flow(retries=1)
    async def read_twice() -> tuple[str, str]:
        result = await agents.plain.run('Nothing to do.')
        await result.workspace.write_text('counter.txt', 'one')
        first = await result.workspace.read_text('counter.txt')
        await result.workspace.write_text('counter.txt', 'two')
        second = await result.workspace.read_text('counter.txt')
        attempts.append((first, second))
        if len(attempts) == 1:
            raise RuntimeError('retry')
        return first, second

    assert await read_twice() == ('one', 'two')
    assert attempts == [('one', 'two')] * 2
    assert provider.log == ['create:env-1']
    assert _workspace_tasks() == ['ensure', 'write_bytes', 'read_bytes', 'write_bytes', 'read_bytes'] * 2


async def test_prefect_tool_workspace_timeout_is_not_retried_by_task_engine(tmp_path: Path) -> None:
    """Re-running a timed-out command could repeat side effects it already completed."""
    starts = 0

    async def slow_command(ctx: RunContext) -> str:
        nonlocal starts
        starts += 1
        return (await ctx.workspace.run(['sleep', '30'], timeout=0.05)).stdout

    agent = Agent(
        TestModel(),
        name='prefect_workspace_timeout',
        tools=[slow_command],
        capabilities=[
            LocalWorkspace(tmp_path),
            PrefectDurability(tool_task_config=TaskConfig(retries=2, retry_delay_seconds=0)),
        ],
    )

    @flow
    async def run_agent() -> str:
        return (await agent.run('run')).output

    with pytest.raises(WorkspaceTimeoutError):
        await run_agent()
    assert starts == 1
