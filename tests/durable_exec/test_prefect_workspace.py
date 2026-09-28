"""Workspaces under `PrefectDurability`, against the Prefect test harness.

Not VCR tests: the behavior under test is where each workspace call runs (a Prefect task, or
directly inside one) and what a flow retry replays, which a provider recording could not show.
The shared scenarios live in `workspace_scenarios.py`; this module runs them in a Prefect flow and
adds what only Prefect has.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.durable_exec._workspace import WorkspaceCall
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import WorkspaceTimeoutError

from ..workspace_fakes import InMemoryProvider
from .workspace_scenarios import SCENARIOS, Check, ScenarioFailed, cases, scenario_agents, tool_runs

try:
    from prefect import flow
    from prefect.context import FlowRunContext, TaskRunContext
    from prefect.settings import PREFECT_SERVER_SERVICES_TASK_RUN_RECORDER_ENABLED, temporary_settings
    from prefect.testing.utilities import prefect_test_harness

    from pydantic_ai.durable_exec.prefect import PrefectDurability, TaskConfig
except ImportError:  # pragma: lax no cover
    pytest.skip('Prefect is not installed', allow_module_level=True)


pytestmark = pytest.mark.xdist_group(name='prefect')


@pytest.fixture(autouse=True, scope='session')
def setup_prefect_test_harness() -> Iterator[None]:
    # See `test_prefect.py`: the task-run recorder's background writer contends for the sqlite file.
    with temporary_settings({PREFECT_SERVER_SERVICES_TASK_RUN_RECORDER_ENABLED: False}):
        with prefect_test_harness(server_startup_timeout=60):
            yield


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


async def test_prefect_run_ids_without_a_workspace_are_fresh_uuid7s() -> None:
    agent = Agent(TestModel(), name='prefect_run_id', capabilities=[PrefectDurability()])
    ids: list[str] = []

    @flow(retries=1)
    async def run() -> None:
        ids.append((await agent.run('hello')).run_id)
        if len(ids) == 1:
            raise RuntimeError('retry')

    await run()
    # Without a workspace, a run inside a flow gets the same fresh UUID7 as outside one.
    assert [uuid.UUID(run_id).version for run_id in ids] == [7, 7]
    assert ids[0] != ids[1]


async def test_prefect_repeated_identical_reads_are_not_served_from_cache() -> None:
    provider.reset()

    @flow
    async def read_twice() -> tuple[str, str]:
        result = await agents.plain.run('Nothing to do.')
        await result.workspace.write_text('counter.txt', 'one')
        first = await result.workspace.read_text('counter.txt')
        await result.workspace.write_text('counter.txt', 'two')
        second = await result.workspace.read_text('counter.txt')
        return first, second

    assert await read_twice() == ('one', 'two')
    assert _workspace_tasks() == ['ensure', 'write_bytes', 'read_bytes', 'write_bytes', 'read_bytes']


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
