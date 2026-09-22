"""Workspaces under `PrefectDurability`, against the Prefect test harness.

Not VCR tests: the behavior under test is where each workspace call runs (a Prefect task, or
directly inside one) and what a flow retry replays, which a provider recording could not show.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace
from pydantic_ai.durable_exec._workspace import DurableWorkspace
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import ReadOnlyWorkspace, WorkspaceRef

from ..workspace_fakes import InMemoryProvider

try:
    from prefect import flow
    from prefect.context import TaskRunContext
    from prefect.settings import PREFECT_SERVER_SERVICES_TASK_RUN_RECORDER_ENABLED, temporary_settings
    from prefect.testing.utilities import prefect_test_harness

    from pydantic_ai.durable_exec.prefect import PrefectDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('Prefect is not installed', allow_module_level=True)


pytestmark = [pytest.mark.anyio, pytest.mark.xdist_group(name='prefect')]


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


provider = InMemoryProvider()
task_names: list[str] = []


class WriteInHook(AbstractCapability[Any]):
    async def before_run(self, ctx: RunContext[Any]) -> None:
        assert isinstance(ctx.workspace, DurableWorkspace)
        assert TaskRunContext.get() is None
        await ctx.workspace.write_text('hook.txt', f'hook in {await ctx.workspace.working_dir()}')


def _workspace_tasks() -> list[str]:
    return [name for name in task_names if name.startswith('Workspace')]


@pytest.fixture(autouse=True)
def record_task_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Record every task run's name at the point its body starts, in the order the flow ran them."""
    from pydantic_ai.durable_exec.prefect import _operation_backend

    task_names.clear()
    original = _operation_backend.PrefectOperationBackend.execute

    async def execute(self: Any, **kwargs: Any) -> object:
        task_names.append(kwargs['name'])
        return await original(self, **kwargs)

    monkeypatch.setattr(_operation_backend.PrefectOperationBackend, 'execute', execute)


async def test_prefect_workspace_operations_run_as_tasks_and_a_flow_retry_replays_them() -> None:
    provider.reset()
    agent = Agent(
        TestModel(call_tools=['read_hook']),
        name='prefect_workspace',
        capabilities=[WriteInHook(), provider.capability(), PrefectDurability()],
    )
    tool_runs = 0

    @agent.tool
    async def read_hook(ctx: RunContext[Any]) -> str:
        nonlocal tool_runs
        tool_runs += 1
        # A tool runs inside its own task; its workspace calls go direct, not into nested tasks.
        assert TaskRunContext.get() is not None
        assert isinstance(ctx.workspace, DurableWorkspace)
        await ctx.workspace.write_text('tool.txt', 'from the tool')
        return await ctx.workspace.read_text('hook.txt')

    attempts = 0

    @flow(retries=1)
    async def run_agent() -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        result = await agent.run('Read the hook file.')
        ref = result.workspace.ref
        assert ref is not None
        # Fail after the run once, so the retry replays every task the run recorded.
        if attempts == 1:
            raise RuntimeError('boom')
        return {
            'output': result.output,
            'ref': ref.id,
            'working_dir': await result.workspace.working_dir(),
            'tool': await result.workspace.read_text('tool.txt'),
        }

    output = await run_agent()

    assert output == snapshot(
        {'output': '{"read_hook":"hook in /remote"}', 'ref': 'env-1', 'working_dir': '/remote', 'tool': 'from the tool'}
    )
    assert attempts == 2
    # The tool task and every workspace task replayed on the retry; only the final read, which the
    # first attempt never reached, ran, and it attached to the environment the first attempt made.
    assert tool_runs == 1
    assert provider.log == snapshot(['create:env-1', 'attach:env-1'])
    assert list(provider.environments) == ['env-1']
    assert _workspace_tasks() == snapshot(
        [
            'Workspace: ensure',
            'Workspace: write_text',
            'Workspace: ensure',
            'Workspace: write_text',
            'Workspace: read_text',
        ]
    )


async def test_prefect_repeated_identical_reads_are_not_served_from_cache() -> None:
    provider.reset()
    agent = Agent(TestModel(), name='prefect_rereads', capabilities=[provider.capability(), PrefectDurability()])

    @flow
    async def read_twice() -> tuple[str, str]:
        result = await agent.run('Nothing to do.')
        await result.workspace.write_text('counter.txt', 'one')
        first = await result.workspace.read_text('counter.txt')
        await result.workspace.write_text('counter.txt', 'two')
        second = await result.workspace.read_text('counter.txt')
        return first, second

    assert await read_twice() == ('one', 'two')
    assert _workspace_tasks() == snapshot(
        [
            'Workspace: ensure',
            'Workspace: write_text',
            'Workspace: read_text',
            'Workspace: write_text',
            'Workspace: read_text',
        ]
    )


async def test_prefect_read_only_policy_is_enforced_inside_the_task() -> None:
    provider.reset()
    agent = Agent(
        TestModel(call_tools=['try_write']),
        name='prefect_read_only',
        capabilities=[provider.capability(read_only=True), PrefectDurability()],
    )

    @agent.tool
    async def try_write(ctx: RunContext[Any]) -> str:
        assert isinstance(ctx.workspace, DurableWorkspace)
        assert isinstance(ctx.workspace.wrapped, ReadOnlyWorkspace)
        try:
            await ctx.workspace.write_text('nope.txt', 'x')
        except UserError as error:
            return f'blocked: {str(error)[:28]}'
        return 'wrote'  # pragma: no cover

    @flow
    async def run_agent() -> str:
        result = await agent.run('Try to write.')
        with pytest.raises(UserError, match='read-only'):
            await result.workspace.make_dir('sub')
        return result.output

    assert await run_agent() == snapshot('{"try_write":"blocked: This workspace is read-only:"}')


async def test_prefect_explicit_workspace_inside_a_flow() -> None:
    provider.reset()
    provider.environments['seeded'] = {'/remote/seed.txt': b'seed'}
    agent = Agent(TestModel(), name='prefect_explicit', capabilities=[provider.capability(), PrefectDurability()])
    seeded = WorkspaceRef(provider='fake', id='seeded')

    @flow
    async def run_agent() -> list[str]:
        by_ref = await agent.run('One.', workspace=seeded)
        by_live = await agent.run('Two.', workspace=provider.backend(seeded))
        by_result = await agent.run('Three.', workspace=by_ref.workspace)
        with pytest.raises(UserError, match='A live workspace cannot be passed to `workspace=` inside a Prefect flow'):
            await agent.run('Four.', workspace=provider.backend(None))
        return [await run.workspace.read_text('seed.txt') for run in (by_ref, by_live, by_result)]

    assert await run_agent() == ['seed', 'seed', 'seed']
    assert list(provider.environments) == ['seeded']
    assert 'create:' not in ' '.join(provider.log)


async def test_prefect_local_workspace_end_to_end(tmp_path: Path) -> None:
    agent = Agent(
        TestModel(call_tools=['write_note']),
        name='prefect_local',
        capabilities=[LocalWorkspace(tmp_path), PrefectDurability(workspace_task_config={'retries': 0})],
    )

    @agent.tool
    async def write_note(ctx: RunContext[Any]) -> str:
        await ctx.workspace.write_text('note.txt', 'on disk')
        return (await ctx.workspace.run(['cat', 'note.txt'])).stdout

    @flow
    async def run_agent() -> dict[str, Any]:
        result = await agent.run('Write the note.')
        return {
            'output': result.output,
            'ref': result.workspace.ref,
            'working_dir': await result.workspace.working_dir(),
        }

    assert await run_agent() == {
        'output': '{"write_note":"on disk"}',
        'ref': WorkspaceRef(provider='local', id=str(tmp_path)),
        'working_dir': str(tmp_path.resolve()),
    }
    assert (tmp_path / 'note.txt').read_text() == 'on disk'
    assert _workspace_tasks() == ['Workspace: ensure']
