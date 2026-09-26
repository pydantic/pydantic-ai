"""Workspaces under `DBOSDurability`, against a real DBOS runtime (sqlite).

Not VCR tests: the behavior under test is where each workspace call runs (a DBOS step, or directly
inside one) and what a forked re-execution replays, which only the DBOS system database can show.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace
from pydantic_ai.durable_exec._workspace import DurableWorkspace
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import ReadOnlyWorkspace, WorkspaceReadOnlyError, WorkspaceRef

from ..workspace_fakes import InMemoryProvider

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('DBOS is not installed', allow_module_level=True)


pytestmark = [pytest.mark.anyio, pytest.mark.xdist_group(name='dbos')]


@pytest.fixture(scope='module')
async def dbos(tmp_path_factory: pytest.TempPathFactory) -> AsyncGenerator[DBOS]:
    # An async module-scoped fixture keeps one event loop for the module: DBOS binds its async
    # machinery to the loop it is launched under, and a per-test loop would leave it on a closed one.
    dbos_sqlite_file = tmp_path_factory.mktemp('dbos') / 'dbostest.sqlite'
    dbos_config: DBOSConfig = {
        'name': 'pydantic_dbos_workspace_tests',
        'system_database_url': f'sqlite:///{dbos_sqlite_file}',
        'run_admin_server': False,
        'enable_otlp': False,
    }
    dbos = DBOS(config=dbos_config)
    DBOS.launch()
    try:
        yield dbos
    finally:
        DBOS.destroy()
        # DBOS leaves its log filter on every logger, and the filter's emit path imports
        # `dbos._context`, which is fatal inside the Temporal workflow sandbox when an xdist
        # worker later runs the Temporal suite. See the `dbos` fixture in `test_dbos.py`.
        from dbos import _logger as dbos_logger_module

        for logger in [logging.root, *(logging.getLogger(name) for name in logging.root.manager.loggerDict)]:
            for log_filter in [f for f in logger.filters if isinstance(f, dbos_logger_module.DBOSLogTransformer)]:
                logger.removeFilter(log_filter)


provider = InMemoryProvider()


class WriteInHook(AbstractCapability[Any]):
    async def before_run(self, ctx: RunContext[Any]) -> None:
        assert isinstance(ctx.workspace, DurableWorkspace)
        await ctx.workspace.write_text('hook.txt', f'hook in {await ctx.workspace.working_dir()}')


fresh_agent = Agent(
    TestModel(call_tools=['read_hook']),
    name='dbos_workspace',
    capabilities=[WriteInHook(), provider.capability(), DBOSDurability()],
)


@fresh_agent.tool
async def read_hook(ctx: RunContext[Any]) -> str:
    # DBOS runs function tools inline in the workflow, so the tool's call is a step of its own.
    assert isinstance(ctx.workspace, DurableWorkspace)
    await ctx.workspace.write_text('tool.txt', 'from the tool')
    return await ctx.workspace.read_text('hook.txt')


@DBOS.workflow()
async def fresh_workflow() -> dict[str, Any]:
    result = await fresh_agent.run('Read the hook file.')
    ref = result.workspace.ref
    assert ref is not None
    return {
        'output': result.output,
        'ref': ref.id,
        'working_dir': await result.workspace.working_dir(),
        'tool': await result.workspace.read_text('tool.txt'),
    }


async def test_dbos_default_run_id_is_workflow_id(dbos: DBOS) -> None:
    workflow_id = f'run-id-{uuid.uuid4()}'

    @DBOS.workflow()
    async def run() -> tuple[str, str]:
        result = await fresh_agent.run('Read the hook file.')
        return result.run_id, (await fresh_agent.run('Read the hook file.', run_id='explicit')).run_id

    with SetWorkflowID(workflow_id):
        assert await run() == (workflow_id, 'explicit')


async def test_dbos_workspace_operations_run_as_steps_and_a_fork_replays_them(dbos: DBOS) -> None:
    provider.reset()
    workflow_id = f'workspace-{uuid.uuid4()}'

    with SetWorkflowID(workflow_id):
        output = await fresh_workflow()

    assert output == snapshot(
        {'output': '{"read_hook":"hook in /remote"}', 'ref': 'env-1', 'working_dir': '/remote', 'tool': 'from the tool'}
    )
    assert provider.log == snapshot(['create:env-1'])
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert [step['function_name'] for step in steps] == snapshot(
        [
            'dbos_workspace__capability__workspace.call',
            'dbos_workspace__capability__workspace.call',
            'dbos_workspace__model.request',
            'dbos_workspace__capability__workspace.call',
            'dbos_workspace__capability__workspace.call',
            'dbos_workspace__model.request',
            'dbos_workspace__capability__workspace.call',
        ]
    )

    # Re-execute the workflow function from its last step, the way recovery does: every earlier
    # step replays its recorded output, `ensure` included, so the rebuilt workspace attaches to the
    # environment the original run created instead of creating another, and the re-executed final
    # read reaches the file the original run wrote.
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    forked = await handle.get_result()
    assert forked == output
    assert provider.log == snapshot(['create:env-1', 'attach:env-1'])
    assert list(provider.environments) == ['env-1']


read_only_agent = Agent(
    TestModel(call_tools=['try_write']),
    name='dbos_read_only',
    capabilities=[provider.capability(read_only=True), DBOSDurability()],
)


@read_only_agent.tool
async def try_write(ctx: RunContext[Any]) -> str:
    try:
        await ctx.workspace.write_text('nope.txt', 'x')
    except WorkspaceReadOnlyError as error:
        return f'blocked: {str(error)[:28]}'
    return 'wrote'  # pragma: no cover


@DBOS.workflow()
async def read_only_workflow() -> str:
    result = await read_only_agent.run('Try to write.')
    assert isinstance(result.workspace, DurableWorkspace)
    assert isinstance(result.workspace.wrapped, ReadOnlyWorkspace)
    try:
        await result.workspace.make_dir('sub')
    except WorkspaceReadOnlyError:
        return f'{result.output}|blocked after the run'
    return result.output  # pragma: no cover


async def test_dbos_read_only_policy_is_enforced_inside_the_step(dbos: DBOS) -> None:
    provider.reset()
    assert await read_only_workflow() == snapshot(
        '{"try_write":"blocked: This workspace is read-only:"}|blocked after the run'
    )


explicit_agent = Agent(TestModel(), name='dbos_explicit', capabilities=[provider.capability(), DBOSDurability()])


@DBOS.workflow()
async def explicit_workflow(kind: str) -> str:
    seeded = WorkspaceRef(provider='fake', id='seeded')
    if kind == 'ref':
        workspace: Any = seeded
    elif kind == 'live_with_ref':
        workspace = provider.backend(seeded)
    elif kind == 'previous_result':
        workspace = (await explicit_agent.run('First.', workspace=seeded)).workspace
    else:
        workspace = provider.backend(None)
    result = await explicit_agent.run('Again.', workspace=workspace)
    return await result.workspace.read_text('seed.txt')


@pytest.mark.parametrize('kind', ['ref', 'live_with_ref', 'previous_result'])
async def test_dbos_explicit_workspace_that_a_capability_recognizes_attaches(dbos: DBOS, kind: str) -> None:
    provider.reset()
    provider.environments['seeded'] = {'/remote/seed.txt': b'seed'}

    assert await explicit_workflow(kind) == 'seed'
    assert list(provider.environments) == ['seeded']
    assert 'create:' not in ' '.join(provider.log)


async def test_dbos_live_workspace_without_a_ref_is_rejected_in_a_workflow(dbos: DBOS) -> None:
    provider.reset()
    with pytest.raises(UserError, match='A live workspace cannot be passed to `workspace=` inside a DBOS workflow'):
        await explicit_workflow('live_fresh')
    assert provider.environments == {}


async def test_dbos_local_workspace_end_to_end(dbos: DBOS, tmp_path: Path) -> None:
    agent = Agent(
        TestModel(call_tools=['write_note']),
        name='dbos_local',
        capabilities=[LocalWorkspace(tmp_path), DBOSDurability()],
    )

    @agent.tool
    async def write_note(ctx: RunContext[Any]) -> str:
        await ctx.workspace.write_text('note.txt', 'on disk')
        return (await ctx.workspace.run(['cat', 'note.txt'])).stdout

    @DBOS.workflow()
    async def local_workflow() -> dict[str, Any]:
        result = await agent.run('Write the note.')
        return {
            'output': result.output,
            'ref': result.workspace.ref,
            'working_dir': await result.workspace.working_dir(),
        }

    output = await local_workflow()
    assert output == {
        'output': '{"write_note":"on disk"}',
        'ref': WorkspaceRef(provider='local', id=str(tmp_path)),
        'working_dir': str(tmp_path.resolve()),
    }
    assert (tmp_path / 'note.txt').read_text() == 'on disk'
