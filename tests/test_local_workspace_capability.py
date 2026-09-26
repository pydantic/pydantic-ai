"""Tests for the built-in [`LocalWorkspace`][pydantic_ai.capabilities.LocalWorkspace] capability."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import (
    LocalWorkspaceBackend,
    ReadOnlyWorkspace,
    WorkspaceReadOnlyError,
    WorkspaceRef,
)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(os.name != 'posix', reason='`LocalWorkspaceBackend` only supports POSIX platforms'),
]


async def test_override_workspace_precedence(tmp_path: Path) -> None:
    original, replacement, explicit = (tmp_path / part for part in ('original', 'replacement', 'explicit'))
    for path in (original, replacement, explicit):
        path.mkdir()
    agent = Agent(TestModel(), capabilities=[LocalWorkspace(original)])
    replacement_backend = LocalWorkspaceBackend(replacement)
    with agent.override(workspace=replacement_backend):
        assert (await agent.run('go')).workspace.ref == replacement_backend.ref
        assert (await agent.run('go', workspace=LocalWorkspaceBackend(explicit))).workspace.ref == WorkspaceRef(
            provider='local', id=str(explicit)
        )
    assert (await agent.run('go')).workspace.ref == WorkspaceRef(provider='local', id=str(original))


@pytest.mark.parametrize('invalid', [{'provider': 'local', 'id': '/tmp'}, '/tmp', Path('/tmp'), 'old', 42])
async def test_invalid_workspace_argument_fails_before_model_call(invalid: object) -> None:
    agent = Agent(TestModel())
    with pytest.raises(TypeError, match=r'workspace=.*WorkspaceRef.*LocalWorkspaceBackend'):
        await agent.run('go', workspace=invalid)  # type: ignore[arg-type]


async def test_resolver_cannot_substitute_a_different_workspace(tmp_path: Path) -> None:
    class MisleadingWorkspace(AbstractCapability[Any]):
        def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> LocalWorkspaceBackend | None:
            return LocalWorkspaceBackend(tmp_path)

    agent = Agent(TestModel(), capabilities=[MisleadingWorkspace()])
    with pytest.raises(UserError, match='different workspace'):
        await agent.run('go', workspace=WorkspaceRef(provider='local', id='/wrong'))


def test_local_workspace_rejects_deferred_loading(tmp_path: Path) -> None:
    with pytest.raises(UserError, match='workspace is chosen at run setup'):
        LocalWorkspace(tmp_path, defer_loading=True)


async def test_unattached_placeholder_does_not_shadow_child_capability(tmp_path: Path) -> None:
    child = Agent(TestModel(call_tools=['write']), capabilities=[LocalWorkspace(tmp_path)])

    @child.tool
    async def write(ctx: RunContext[Any]) -> str:
        await ctx.workspace.write_text('child.txt', 'ready')
        return 'ready'

    parent = Agent(TestModel(call_tools=['delegate']))

    @parent.tool
    async def delegate(ctx: RunContext[Any]) -> str:
        result = await child.run('go', workspace=ctx.workspace)
        return result.output

    await parent.run('go')
    assert (tmp_path / 'child.txt').read_text() == 'ready'


async def test_tools_use_the_local_workspace(tmp_path: Path) -> None:
    agent = Agent(TestModel(call_tools=['write_and_run']), capabilities=[LocalWorkspace(tmp_path)])

    @agent.tool
    async def write_and_run(ctx: RunContext[Any]) -> str:
        await ctx.workspace.write_text('greeting.txt', 'hello')
        return (await ctx.workspace.run(['cat', 'greeting.txt'])).stdout

    result = await agent.run('go')

    assert result.output == '{"write_and_run":"hello"}'
    assert (tmp_path / 'greeting.txt').read_text() == 'hello'
    assert isinstance(result.workspace.backend, LocalWorkspaceBackend)
    assert await result.workspace.working_dir() == str(tmp_path.resolve())


async def test_env_reaches_every_command(tmp_path: Path) -> None:
    agent = Agent(TestModel(), capabilities=[LocalWorkspace(tmp_path, env={'GREETING': 'hello'})])

    result = await agent.run('go')

    assert (await result.workspace.run(['sh', '-c', 'printf %s "$GREETING"'])).stdout == 'hello'


async def test_working_dir_expands_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HOME', str(tmp_path))
    (tmp_path / 'project').mkdir()
    agent = Agent(TestModel(), capabilities=[LocalWorkspace('~/project')])

    result = await agent.run('go')

    assert await result.workspace.working_dir() == str((tmp_path / 'project').resolve())


async def test_dot_supplies_the_directory_the_capability_was_constructed_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    agent = Agent(TestModel(), capabilities=[LocalWorkspace('.')])
    monkeypatch.chdir('/')

    result = await agent.run('go')

    assert result.workspace.ref == WorkspaceRef(provider='local', id=str(tmp_path))


async def test_read_only_allows_reads_and_refuses_writes_and_commands(tmp_path: Path) -> None:
    (tmp_path / 'notes.txt').write_text('read me')
    agent = Agent(TestModel(call_tools=['probe']), capabilities=[LocalWorkspace(tmp_path, read_only=True)])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        with pytest.raises(WorkspaceReadOnlyError, match='read-only'):
            await ctx.workspace.write_text('notes.txt', 'changed')
        with pytest.raises(WorkspaceReadOnlyError, match='read-only'):
            await ctx.workspace.run(['rm', 'notes.txt'])
        return await ctx.workspace.read_text('notes.txt')

    result = await agent.run('go')

    assert result.output == '{"probe":"read me"}'
    assert isinstance(result.workspace, ReadOnlyWorkspace)
    assert (tmp_path / 'notes.txt').read_text() == 'read me'


async def test_foreign_ref_without_another_capability_raises(tmp_path: Path) -> None:
    agent = Agent(TestModel(), capabilities=[LocalWorkspace(tmp_path)])

    with pytest.raises(UserError, match="none of the agent's workspace capabilities recognized it"):
        await agent.run('go', workspace=WorkspaceRef(provider='fake', id='remote'))


async def test_responses_carry_the_local_ref_and_the_next_run_continues_in_the_same_directory(tmp_path: Path) -> None:
    ref = WorkspaceRef(provider='local', id=str(tmp_path))
    agent = Agent(TestModel(), capabilities=[LocalWorkspace(tmp_path)])

    first = await agent.run('go')
    second = await agent.run('again', message_history=first.all_messages())

    assert [m.workspace_ref for m in second.all_messages() if isinstance(m, ModelResponse)] == [ref, ref]
    assert isinstance(second.workspace.backend, LocalWorkspaceBackend)
    assert second.workspace.ref == ref


async def test_own_ref_is_claimed_whatever_the_spelling_of_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('HOME', str(tmp_path))
    agent = Agent(TestModel(), capabilities=[LocalWorkspace('~/project')])

    result = await agent.run('go', workspace=WorkspaceRef(provider='local', id=f'{tmp_path}/project'))

    assert isinstance(result.workspace.backend, LocalWorkspaceBackend)


@pytest.mark.parametrize('blockbuster_enabled', [False])
@pytest.mark.parametrize('read_only', [False, True])
async def test_local_ref_with_symlink_spelling_attaches_to_same_directory(
    tmp_path: Path, blockbuster_enabled: bool, read_only: bool
) -> None:
    root = tmp_path / 'root'
    root.mkdir()
    (tmp_path / 'alias').symlink_to(root)
    agent = Agent(TestModel(), capabilities=[LocalWorkspace(root, read_only=read_only)])
    for spelling in (tmp_path / 'alias', tmp_path / 'root' / '..' / 'root'):
        ref = WorkspaceRef(provider='local', id=str(spelling))
        result = await agent.run('go', message_history=[ModelResponse(parts=[TextPart('old')], workspace_ref=ref)])
        assert await result.workspace.working_dir() == str(root.resolve())
        assert result.workspace.ref == WorkspaceRef(provider='local', id=str(root))
        assert result.workspace.read_only is read_only


async def test_local_ref_for_another_directory_is_never_followed(tmp_path: Path) -> None:
    """A ref in message history must not be able to point the agent at an arbitrary host directory."""
    elsewhere = WorkspaceRef(provider='local', id=str(tmp_path / 'elsewhere'))
    historical = ModelResponse(parts=[TextPart('old')], workspace_ref=elsewhere)
    agent = Agent(TestModel(), capabilities=[LocalWorkspace(tmp_path / 'configured')])

    with pytest.raises(UserError, match="none of the agent's workspace capabilities recognized it"):
        await agent.run('go', workspace=elsewhere)
    with pytest.raises(UserError, match='The message history continues in workspace `local:'):
        await agent.run('go', message_history=[historical])

    fresh = await agent.run('go', message_history=[historical], workspace='new')
    assert fresh.workspace.ref == WorkspaceRef(provider='local', id=str(tmp_path / 'configured'))


async def test_a_repeated_local_workspace_resolves_to_the_later_one(tmp_path: Path) -> None:
    """The default `id` makes a repeat one configuration stated twice; a distinct `id` keeps both, and the first answers."""
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    merged = Agent(TestModel(), capabilities=[LocalWorkspace(first), LocalWorkspace(second)])
    both = Agent(TestModel(), capabilities=[LocalWorkspace(first), LocalWorkspace(second, id='scratch')])

    assert await (await merged.run('go')).workspace.working_dir() == str(second.resolve())
    assert await (await both.run('go')).workspace.working_dir() == str(first.resolve())
    continued = await both.run('go', workspace=WorkspaceRef(provider='local', id=str(second.resolve())))
    assert await continued.workspace.working_dir() == str(second.resolve())


async def test_agent_spec_builds_a_local_workspace(tmp_path: Path) -> None:
    agent = Agent.from_spec(
        {'model': 'test', 'capabilities': [{'LocalWorkspace': {'working_dir': str(tmp_path), 'read_only': True}}]}
    )

    result = await agent.run('go')

    assert isinstance(result.workspace, ReadOnlyWorkspace)
    assert await result.workspace.working_dir() == str(tmp_path.resolve())
