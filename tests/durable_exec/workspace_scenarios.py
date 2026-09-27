"""Workspace scenarios every durable engine must pass, written once.

Each engine's test module builds the scenario agents with its durability capability, runs a scenario
inside a workflow or flow through a small runner, and parametrizes over `CASES`; the checks here say
what the run must produce. A runner reports a failed workflow or flow as `ScenarioFailed`, so a check
can assert the error type and message the same way on every engine.

This module must stay importable inside the Temporal workflow sandbox, which re-imports it: no engine
imports, nothing that touches the filesystem at import time.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace
from pydantic_ai.durable_exec._workspace import DurableWorkspace
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import ReadOnlyWorkspace, Workspace, WorkspaceBackend, WorkspaceReadOnlyError, WorkspaceRef

from ..workspace_fakes import InMemoryProvider


class ScenarioFailed(Exception):
    """A scenario's workflow or flow failed with an error of type `type`."""

    def __init__(self, type: str, message: str) -> None:
        super().__init__(f'{type}: {message}')
        self.type = type
        self.message = message


class WriteInHook(AbstractCapability[Any]):
    """Touches the workspace from a hook, which runs in workflow code: every call must be a durable unit."""

    async def before_run(self, ctx: RunContext[Any]) -> None:
        await ctx.workspace.write_text('hook.txt', f'hook in {await ctx.workspace.working_dir()}')


class ReadMissingInHook(AbstractCapability[Any]):
    async def before_run(self, ctx: RunContext[Any]) -> None:
        await ctx.workspace.read_text('missing.txt')


class AmnesiacWorkspaces(AbstractCapability[Any]):
    """Creates environments but can't reattach to the ones it created."""

    def __init__(self, provider: InMemoryProvider) -> None:
        self.provider = provider

    def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        return self.provider.backend(None) if ref is None else None


tool_runs: list[str] = []
"""The tools that ran, so a test can tell a replayed tool from one that ran again."""


async def write_left(ctx: RunContext[Any]) -> str:
    tool_runs.append('write_left')
    await ctx.workspace.write_text('left.txt', 'L')
    return await ctx.workspace.read_text('hook.txt')


async def write_right(ctx: RunContext[Any]) -> str:
    await ctx.workspace.write_text('right.txt', 'R')
    return (await ctx.workspace.run(['pwd'])).stdout


async def try_write(ctx: RunContext[Any]) -> str:
    try:
        await ctx.workspace.write_text('nope.txt', 'x')
    except WorkspaceReadOnlyError:
        return 'blocked'
    return 'wrote'  # pragma: no cover


async def read_seed(ctx: RunContext[Any]) -> str:
    return await ctx.workspace.read_text('seed.txt')


async def write_note(ctx: RunContext[Any]) -> str:
    await ctx.workspace.write_text('note.txt', 'on disk')
    return (await ctx.workspace.run(['cat', 'note.txt'])).stdout


@dataclass
class ScenarioAgents:
    provider: InMemoryProvider
    local_dir: str
    fresh: Agent[None, str]
    plain: Agent[None, str]
    read_only: Agent[None, str]
    explicit: Agent[None, str]
    local: Agent[None, str]
    amnesiac: Agent[None, str]
    uncaught: Agent[None, str]

    def all(self) -> list[Agent[None, str]]:
        return [self.fresh, self.plain, self.read_only, self.explicit, self.local, self.amnesiac, self.uncaught]


def scenario_agents(
    durability: Callable[[], AbstractCapability[Any]], *, prefix: str, provider: InMemoryProvider
) -> ScenarioAgents:
    """The agents every scenario uses, named `<prefix><role>`, each with a fresh `durability()`."""
    workspaces = provider.capability
    local_dir = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'pydantic_ai_{prefix}local_workspace')
    return ScenarioAgents(
        provider=provider,
        local_dir=local_dir,
        fresh=Agent(
            TestModel(call_tools=['write_left', 'write_right']),
            name=f'{prefix}fresh',
            tools=[write_left, write_right],
            capabilities=[WriteInHook(), workspaces(), durability()],
        ),
        plain=Agent(TestModel(), name=f'{prefix}plain', capabilities=[workspaces(), durability()]),
        read_only=Agent(
            TestModel(call_tools=['try_write']),
            name=f'{prefix}read_only',
            tools=[try_write],
            capabilities=[workspaces(read_only=True), durability()],
        ),
        explicit=Agent(
            TestModel(call_tools=['read_seed']),
            name=f'{prefix}explicit',
            tools=[read_seed],
            capabilities=[workspaces(), durability()],
        ),
        local=Agent(
            TestModel(call_tools=['write_note']),
            name=f'{prefix}local',
            tools=[write_note],
            capabilities=[LocalWorkspace(local_dir), durability()],
        ),
        amnesiac=Agent(
            TestModel(), name=f'{prefix}amnesiac', capabilities=[AmnesiacWorkspaces(provider), durability()]
        ),
        uncaught=Agent(
            TestModel(), name=f'{prefix}uncaught', capabilities=[ReadMissingInHook(), workspaces(), durability()]
        ),
    )


# --- Scenarios: run inside the workflow or flow ------------------------------------------------

Scenario = Callable[[ScenarioAgents, 'str | None', str], Awaitable[Any]]


async def fresh(agents: ScenarioAgents, arg: str | None, engine_id: str) -> dict[str, Any]:
    result = await agents.fresh.run('Use both tools.')
    workspace = result.workspace
    assert workspace.ref is not None
    return {
        'output': result.output,
        'ref': workspace.ref.id,
        'response_refs': [
            message.workspace_ref.id if message.workspace_ref else None
            for message in result.all_messages()
            if message.kind == 'response'
        ],
        'working_dir': await workspace.working_dir(),
        'resolved': await workspace.resolve('nested/file.txt'),
        'files': [entry.path for entry in await workspace.list_dir('.')],
        'left': await workspace.read_text('left.txt'),
    }


async def multi_turn(agents: ScenarioAgents, arg: str | None, engine_id: str) -> list[Any]:
    first = await agents.plain.run('First.')
    second = await agents.plain.run('Second.', message_history=first.all_messages())
    return [first.run_id, second.run_id, first.workspace.ref == second.workspace.ref]


async def two_agents(agents: ScenarioAgents, arg: str | None, engine_id: str) -> list[str | None]:
    first = await agents.plain.run('One.')
    second = await agents.read_only.run('Two.')
    return [workspace.ref.id if workspace.ref else None for workspace in (first.workspace, second.workspace)]


async def run_id(agents: ScenarioAgents, arg: str | None, engine_id: str) -> list[str]:
    generated = (await agents.plain.run('Hi.')).run_id
    explicit = (await agents.plain.run('Hi.', run_id='explicit')).run_id
    return [generated, explicit, engine_id]


async def read_only(agents: ScenarioAgents, arg: str | None, engine_id: str) -> dict[str, Any]:
    result = await agents.read_only.run('Try to write.')
    workspace = result.workspace
    try:
        await workspace.make_dir('sub')
    except WorkspaceReadOnlyError:
        after = 'blocked'
    else:  # pragma: no cover
        after = 'wrote'
    return {
        'output': result.output,
        'after': after,
        'wrapped': isinstance(workspace, DurableWorkspace) and isinstance(workspace.wrapped, ReadOnlyWorkspace),
    }


async def explicit(agents: ScenarioAgents, arg: str | None, engine_id: str) -> dict[str, Any]:
    provider = agents.provider
    seeded = WorkspaceRef(provider=provider.name, id='seeded')
    workspace: Any
    if arg == 'ref':
        workspace = seeded
    elif arg == 'live_with_ref':
        workspace = provider.backend(seeded)
    elif arg == 'live_read_only':
        workspace = ReadOnlyWorkspace(Workspace(provider.backend(seeded)))
    elif arg == 'live_fresh':
        workspace = provider.backend(None)
    elif arg == 'foreign_ref':
        workspace = WorkspaceRef(provider='other', id='x')
    elif arg == 'dead_ref':
        workspace = WorkspaceRef(provider=provider.name, id='expired')
    else:
        assert arg == 'previous_result'
        workspace = (await agents.explicit.run('Read the seed.', workspace=seeded)).workspace
    result = await agents.explicit.run('Read the seed.', workspace=workspace)
    assert result.workspace.ref is not None
    return {'output': result.output, 'ref': result.workspace.ref.id}


_BINARY = b'\xff\xfe\x00\x01binary\x80'


async def _error_of(call: Awaitable[Any]) -> str:
    try:
        await call
    except (OSError, UnicodeDecodeError, TypeError) as error:
        return type(error).__name__
    return 'none'  # pragma: no cover


async def binary(agents: ScenarioAgents, arg: str | None, engine_id: str) -> dict[str, Any]:
    workspace = (await agents.plain.run('Nothing to do.')).workspace
    await workspace.write_bytes('blob.bin', _BINARY)
    await workspace.make_dir('sub')
    await workspace.write_text('sub/gone.txt', 'x')
    await workspace.remove('sub/gone.txt')
    try:
        await workspace.read_text('blob.bin')
    except UnicodeDecodeError as error:
        decoded = error.object == _BINARY
    else:  # pragma: no cover
        decoded = False
    return {
        'round_trip': (await workspace.read_bytes('blob.bin')) == _BINARY,
        'size': (await workspace.stat('blob.bin')).size,
        'exists': [await workspace.exists('blob.bin'), await workspace.exists('nope')],
        'decode_error_keeps_bytes': decoded,
        'errors': [
            await _error_of(workspace.stat('sub/gone.txt')),
            await _error_of(workspace.remove('missing.txt')),
            await _error_of(workspace.run('echo hi')),
        ],
    }


async def local(agents: ScenarioAgents, arg: str | None, engine_id: str) -> dict[str, Any]:
    result = await agents.local.run('Write the note.')
    assert result.workspace.ref is not None
    return {
        'output': result.output,
        'ref': result.workspace.ref.id,
        'working_dir': await result.workspace.working_dir(),
        'note': await result.workspace.read_text('note.txt'),
    }


async def amnesiac(agents: ScenarioAgents, arg: str | None, engine_id: str) -> str:
    return (await agents.amnesiac.run('Nothing to do.')).output


async def uncaught(agents: ScenarioAgents, arg: str | None, engine_id: str) -> str:
    return (await agents.uncaught.run('Nothing to do.')).output


SCENARIOS: dict[str, Scenario] = {
    scenario.__name__: scenario
    for scenario in (fresh, multi_turn, two_agents, run_id, read_only, explicit, binary, local, amnesiac, uncaught)
}


# --- Checks: run by the test, through an engine's runner -----------------------------------------

Runner = Callable[[str, 'str | None'], Awaitable[Any]]
"""Runs `SCENARIOS[name]` with `arg` inside a workflow or flow; raises `ScenarioFailed` if it fails."""


async def check_fresh(run: Runner, agents: ScenarioAgents) -> None:
    """One environment, created by one `ensure`, reached by the hook, both tools and the result."""
    assert await run('fresh', None) == {
        'output': '{"write_left":"hook in /remote","write_right":"ran:pwd"}',
        'ref': 'env-1',
        'response_refs': ['env-1', 'env-1'],
        'working_dir': '/remote',
        'resolved': '/remote/nested/file.txt',
        'files': ['/remote/hook.txt', '/remote/left.txt', '/remote/right.txt'],
        'left': 'L',
    }
    assert list(agents.provider.environments) == ['env-1']
    assert [entry for entry in agents.provider.log if entry.startswith('create:')] == ['create:env-1']


async def check_multi_turn(run: Runner, agents: ScenarioAgents) -> None:
    first_id, second_id, same_ref = await run('multi_turn', None)
    assert first_id != second_id and same_ref
    # Outside a workflow or flow, the agent keeps a plain random run ID.
    assert ':' not in (await agents.plain.run('Outside.')).run_id


async def check_two_agents(run: Runner, agents: ScenarioAgents) -> None:
    """Two agents in one workflow or flow each get their own fresh environment."""
    assert await run('two_agents', None) == ['env-1', 'env-2']


async def check_run_id(run: Runner, agents: ScenarioAgents) -> None:
    generated, explicit_id, engine_id = await run('run_id', None)
    assert generated != engine_id
    assert explicit_id == 'explicit'


async def check_read_only(run: Runner, agents: ScenarioAgents) -> None:
    """Enforced inside the unit (the tool) and from workflow code (the result), and never lost."""
    assert await run('read_only', None) == {'output': '{"try_write":"blocked"}', 'after': 'blocked', 'wrapped': True}
    assert agents.provider.environments == {'env-1': {}}


async def check_explicit_attaches(kind: str, run: Runner, agents: ScenarioAgents) -> None:
    agents.provider.environments['seeded'] = {'/remote/seed.txt': b'seed'}
    assert await run('explicit', kind) == {'output': '{"read_seed":"seed"}', 'ref': 'seeded'}
    assert list(agents.provider.environments) == ['seeded']
    assert not any(entry.startswith('create:') for entry in agents.provider.log)


async def check_explicit_rejected(kind: str, type: str, fragment: str, run: Runner, agents: ScenarioAgents) -> None:
    agents.provider.environments['seeded'] = {'/remote/seed.txt': b'seed'}
    try:
        await run('explicit', kind)
    except ScenarioFailed as failure:
        assert (failure.type, fragment in failure.message) == (type, True), failure
    else:  # pragma: no cover
        raise AssertionError(f'{kind} was accepted')
    assert list(agents.provider.environments) == ['seeded']


async def check_binary(run: Runner, agents: ScenarioAgents) -> None:
    """Bytes and the expected errors cross the durable boundary unchanged."""
    assert await run('binary', None) == {
        'round_trip': True,
        'size': 11,
        'exists': [True, False],
        'decode_error_keeps_bytes': True,
        'errors': ['FileNotFoundError', 'FileNotFoundError', 'TypeError'],
    }
    assert agents.provider.environments['env-1']['/remote/blob.bin'] == _BINARY


async def check_local(run: Runner, agents: ScenarioAgents) -> None:
    shutil.rmtree(agents.local_dir, ignore_errors=True)
    os.makedirs(agents.local_dir)
    try:
        assert await run('local', None) == {
            'output': '{"write_note":"on disk"}',
            'ref': agents.local_dir,
            'working_dir': os.path.realpath(agents.local_dir),
            'note': 'on disk',
        }
        with open(os.path.join(agents.local_dir, 'note.txt'), encoding='utf-8') as note:
            assert note.read() == 'on disk'
    finally:
        shutil.rmtree(agents.local_dir, ignore_errors=True)


async def check_failure(name: str, type: str, fragment: str, run: Runner, agents: ScenarioAgents) -> None:
    """An error the run can't recover from fails the workflow or flow with that same error."""
    try:
        await run(name, None)
    except ScenarioFailed as failure:
        assert (failure.type, fragment in failure.message) == (type, True), failure
    else:  # pragma: no cover
        raise AssertionError(f'{name} succeeded')


Check = Callable[[Runner, ScenarioAgents], Awaitable[None]]

CASES: dict[str, Check] = {
    'fresh': check_fresh,
    'multi_turn': check_multi_turn,
    'two_agents': check_two_agents,
    'run_id': check_run_id,
    'read_only': check_read_only,
    **{
        f'explicit_{kind}': partial(check_explicit_attaches, kind)
        for kind in ('ref', 'live_with_ref', 'previous_result')
    },
    'explicit_live_fresh': partial(
        check_explicit_rejected, 'live_fresh', 'UserError', 'A live workspace cannot be passed to `workspace=`'
    ),
    'explicit_foreign_ref': partial(
        check_explicit_rejected, 'foreign_ref', 'UserError', "none of the agent's workspace capabilities recognized it"
    ),
    'explicit_live_read_only': partial(
        check_explicit_rejected, 'live_read_only', 'UserError', 'policy would be lost across durable units'
    ),
    'explicit_dead_ref': partial(
        check_explicit_rejected, 'dead_ref', 'WorkspaceUnavailableError', "environment 'expired' does not exist"
    ),
    'binary': check_binary,
    'local': check_local,
    'amnesiac': partial(check_failure, 'amnesiac', 'UserError', 'which the run just created'),
    'uncaught': partial(check_failure, 'uncaught', 'FileNotFoundError', '/remote/missing.txt'),
}


def cases() -> list[Any]:
    """`CASES` as pytest params, one per scenario."""
    return [pytest.param(check, id=name) for name, check in CASES.items()]
