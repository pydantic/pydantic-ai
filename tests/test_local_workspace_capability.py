"""Tests for the built-in [`LocalWorkspace`][pydantic_ai.capabilities.LocalWorkspace] capability."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import (
    LocalWorkspaceBackend,
    ReadOnlyWorkspace,
    UnavailableWorkspace,
    WorkspaceReadOnlyError,
    WorkspaceRef,
)

from .workspace_fakes import ConnectOnlyWorkspaceCapability, FakeWorkspace, WorkspaceCapability

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(os.name != 'posix', reason='`LocalWorkspaceBackend` only supports POSIX platforms'),
]


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


async def test_explicit_workspace_overrides_the_capability(tmp_path: Path) -> None:
    explicit = FakeWorkspace('explicit')
    agent = Agent(TestModel(), capabilities=[LocalWorkspace(tmp_path)])

    result = await agent.run('go', workspace=explicit)

    assert result.workspace.backend is explicit


@pytest.mark.parametrize('local_first', [True, False])
async def test_first_workspace_capability_wins(tmp_path: Path, local_first: bool) -> None:
    other = WorkspaceCapability()
    local = LocalWorkspace[Any](tmp_path)
    agent = Agent(TestModel(), capabilities=[local, other] if local_first else [other, local])

    result = await agent.run('go')

    assert isinstance(result.workspace.backend, LocalWorkspaceBackend) is local_first
    assert other.refs == ([] if local_first else [None])


@pytest.mark.parametrize('source', ['explicit', 'history'])
async def test_foreign_ref_is_declined_so_a_later_capability_can_claim_it(tmp_path: Path, source: str) -> None:
    provider = ConnectOnlyWorkspaceCapability()
    agent = Agent(TestModel(), capabilities=[LocalWorkspace(tmp_path), provider])
    ref = WorkspaceRef(provider='fake', id='remote')

    if source == 'explicit':
        result = await agent.run('go', workspace=ref)
    else:
        result = await agent.run('go', message_history=[ModelResponse(parts=[TextPart('old')], workspace_ref=ref)])

    assert result.workspace.ref == ref
    assert provider.ids == ['remote']


async def test_foreign_ref_without_another_capability_raises(tmp_path: Path) -> None:
    agent = Agent(TestModel(), capabilities=[LocalWorkspace(tmp_path)])

    with pytest.raises(UserError, match="No capability can supply workspace 'remote'"):
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


async def test_local_ref_for_another_directory_is_never_followed(tmp_path: Path) -> None:
    """A ref in message history must not be able to point the agent at an arbitrary host directory."""
    elsewhere = WorkspaceRef(provider='local', id=str(tmp_path / 'elsewhere'))
    historical = ModelResponse(parts=[TextPart('old')], workspace_ref=elsewhere)
    agent = Agent(TestModel(), capabilities=[LocalWorkspace(tmp_path / 'configured')])

    with pytest.raises(UserError, match='No capability can supply workspace'):
        await agent.run('go', workspace=elsewhere)

    continued = await agent.run('go', message_history=[historical])
    assert isinstance(continued.workspace.backend, UnavailableWorkspace)

    fresh = await agent.run('go', message_history=[historical], workspace='new')
    assert fresh.workspace.ref == WorkspaceRef(provider='local', id=str(tmp_path / 'configured'))


async def test_new_workspace_ignores_a_foreign_ref_in_history(tmp_path: Path) -> None:
    provider = ConnectOnlyWorkspaceCapability()
    agent = Agent(TestModel(), capabilities=[provider, LocalWorkspace(tmp_path)])
    historical = ModelResponse(parts=[TextPart('old')], workspace_ref=WorkspaceRef(provider='fake', id='remote'))

    result = await agent.run('go', message_history=[historical], workspace='new')

    assert isinstance(result.workspace.backend, LocalWorkspaceBackend)
    assert provider.ids == []


async def test_a_repeated_local_workspace_resolves_to_the_later_one(tmp_path: Path) -> None:
    """The default `id` makes a repeat one configuration stated twice; a distinct `id` keeps both."""
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    merged = Agent(TestModel(), capabilities=[LocalWorkspace(first), LocalWorkspace(second)])
    distinct = Agent(TestModel(), capabilities=[LocalWorkspace(first), LocalWorkspace(second, id='scratch')])

    assert await (await merged.run('go')).workspace.working_dir() == str(second.resolve())
    assert await (await distinct.run('go')).workspace.working_dir() == str(first.resolve())


async def test_agent_spec_builds_a_local_workspace(tmp_path: Path) -> None:
    agent = Agent.from_spec(
        {'model': 'test', 'capabilities': [{'LocalWorkspace': {'working_dir': str(tmp_path), 'read_only': True}}]}
    )

    result = await agent.run('go')

    assert isinstance(result.workspace, ReadOnlyWorkspace)
    assert await result.workspace.working_dir() == str(tmp_path.resolve())
