"""Tests for per-path workspace authority: `AuthorityWorkspace(path_authority=...)` and `Workspace.authority_of`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import Authority
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import AuthorityWorkspace, LocalWorkspaceBackend, ReadOnlyWorkspace, Workspace
from pydantic_ai.workspaces.workspace import workspace_layers

pytestmark = [
    pytest.mark.skipif(os.name != 'posix', reason='`LocalWorkspaceBackend` only supports POSIX platforms'),
]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / 'repo'
    (root / 'vendor' / 'ours').mkdir(parents=True)
    (root / 'trusted').mkdir()
    return root


def workspace(
    root: Path, authority: Authority = 'operator', path_authority: dict[str, Authority] | None = None
) -> Workspace:
    return AuthorityWorkspace(
        Workspace(LocalWorkspaceBackend(root)), authority=authority, path_authority=path_authority
    )


async def test_without_path_authority_files_in_the_working_directory_have_the_workspace_authority(root: Path) -> None:
    plain = Workspace(LocalWorkspaceBackend(root))
    assert await plain.authority_of('AGENTS.md') == 'operator'
    assert await plain.authority_of(str(root / 'vendor' / 'AGENTS.md')) == 'operator'
    assert await plain.authority_of(str(root)) == 'operator'

    users = AuthorityWorkspace(plain, authority='user')
    assert await users.authority_of('AGENTS.md') == 'user'
    assert await users.authority_of('vendor/ours/AGENTS.md') == 'user'


async def test_a_path_outside_the_working_directory_is_outside_the_workspace(root: Path) -> None:
    plain = Workspace(LocalWorkspaceBackend(root))
    assert await plain.authority_of(str(root.parent / 'elsewhere.md')) == 'none'
    assert await plain.authority_of('../elsewhere.md') == 'none'
    # A sibling whose name starts with the working directory's isn't inside it.
    assert await plain.authority_of(f'{root}-other/AGENTS.md') == 'none'


async def test_the_longest_matching_path_wins(root: Path) -> None:
    ws = workspace(root, path_authority={'vendor': 'none', 'vendor/ours': 'user'})

    assert await ws.authority_of('AGENTS.md') == 'operator'
    assert await ws.authority_of('vendor') == 'none'
    assert await ws.authority_of('vendor/lib/AGENTS.md') == 'none'
    assert await ws.authority_of('vendor/ours') == 'user'
    assert await ws.authority_of('vendor/ours/AGENTS.md') == 'user'
    # `vendor-other` shares a prefix with `vendor` as text, not as a path.
    assert await ws.authority_of('vendor-other/AGENTS.md') == 'operator'


async def test_a_lower_path_inside_a_higher_one(root: Path) -> None:
    ws = workspace(root, path_authority={'vendor': 'none'})

    assert await ws.authority_of('trusted/AGENTS.md') == 'operator'
    assert await ws.authority_of('vendor/AGENTS.md') == 'none'


async def test_a_higher_path_inside_a_lower_one(root: Path) -> None:
    ws = workspace(root, 'none', {'trusted': 'operator', 'vendor/ours': 'user'})

    assert await ws.authority_of('AGENTS.md') == 'none'
    assert await ws.authority_of('vendor/AGENTS.md') == 'none'
    assert await ws.authority_of('trusted/AGENTS.md') == 'operator'
    assert await ws.authority_of('vendor/ours/AGENTS.md') == 'user'
    assert ws.authority == 'none'


async def test_path_authority_outside_the_working_directory_brings_that_path_in(root: Path) -> None:
    config = root.parent / 'config'
    ws = workspace(root, 'none', {str(config): 'operator'})

    assert await ws.authority_of(str(config / 'AGENTS.md')) == 'operator'
    assert await ws.authority_of(str(root / 'AGENTS.md')) == 'none'
    assert await ws.authority_of(str(root.parent / 'other' / 'AGENTS.md')) == 'none'


async def test_a_link_from_a_lower_path_into_a_higher_one_keeps_the_lower_level(root: Path) -> None:
    (root / 'trusted' / 'AGENTS.md').write_text('trusted')
    (root / 'vendor' / 'AGENTS.md').symlink_to(root / 'trusted' / 'AGENTS.md')
    (root / 'vendor' / 'dir').symlink_to(root / 'trusted')
    ws = workspace(root, path_authority={'vendor': 'none'})

    assert await ws.authority_of('trusted/AGENTS.md') == 'operator'
    # The link itself, and a directory link part-way along the path.
    assert await ws.authority_of('vendor/AGENTS.md') == 'none'
    assert await ws.authority_of('vendor/dir/AGENTS.md') == 'none'


async def test_a_link_from_a_higher_path_into_a_lower_one_takes_the_lower_level(root: Path) -> None:
    (root / 'vendor' / 'AGENTS.md').write_text('vendored')
    (root / 'trusted' / 'AGENTS.md').symlink_to(root / 'vendor' / 'AGENTS.md')
    (root / 'trusted' / 'outside.md').symlink_to(root.parent / 'outside.md')
    ws = workspace(root, path_authority={'vendor': 'none'})

    assert await ws.authority_of('trusted/AGENTS.md') == 'none'
    # A link out of the workspace leads outside it.
    assert await ws.authority_of('trusted/outside.md') == 'none'


async def test_a_link_above_the_workspace_is_not_the_workspace_s(root: Path) -> None:
    # Like `/tmp` linking to `/private/tmp` on macOS: the link is the machine's, not the workspace's.
    alias = root.parent / 'alias'
    alias.symlink_to(root)
    ws = workspace(root, 'user', {'vendor': 'none'})

    assert await ws.authority_of(str(alias / 'AGENTS.md')) == 'user'
    assert await ws.authority_of(str(alias / 'vendor' / 'AGENTS.md')) == 'none'


async def test_path_authority_keys_resolve_through_links(root: Path) -> None:
    (root / 'third_party').symlink_to(root / 'vendor')
    # Two keys naming one directory: the lower level wins, whichever comes first.
    ws = workspace(root, path_authority={'third_party': 'none', 'vendor': 'user'})
    reversed_ws = workspace(root, path_authority={'vendor': 'user', 'third_party': 'none'})

    assert await ws.authority_of('vendor/AGENTS.md') == 'none'
    assert await reversed_ws.authority_of('vendor/AGENTS.md') == 'none'


async def test_wrappers_only_lower_the_level(root: Path) -> None:
    inner = workspace(root, path_authority={'vendor': 'none', 'trusted': 'user'})

    capped = AuthorityWorkspace(inner, authority='user')
    assert await capped.authority_of('AGENTS.md') == 'user'
    assert await capped.authority_of('vendor/AGENTS.md') == 'none'
    assert await capped.authority_of('trusted/AGENTS.md') == 'user'

    # An outer wrapper's path can't raise what an inner wrapper set.
    raising = AuthorityWorkspace(workspace(root, 'none'), authority='operator', path_authority={'trusted': 'operator'})
    assert await raising.authority_of('trusted/AGENTS.md') == 'none'
    assert raising.authority == 'none'

    # Wrappers that don't set authority pass it through.
    assert await ReadOnlyWorkspace(inner).authority_of('vendor/AGENTS.md') == 'none'
    assert await Workspace(ReadOnlyWorkspace(capped)).authority_of('AGENTS.md') == 'user'


def test_path_authority_is_part_of_the_durable_policy(root: Path) -> None:
    assert workspace(root, 'user').durable_policy() == ('user', {})
    assert workspace(root, 'user', {'vendor': 'none'}).durable_policy() == ('user', {'vendor': 'none'})
    assert workspace(root, 'user', {'vendor': 'none'}).durable_policy() != workspace(root, 'user').durable_policy()


async def test_local_workspace_path_authority(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = root.parent / 'home'
    (home / '.config' / 'app').mkdir(parents=True)
    monkeypatch.setenv('HOME', str(home))
    capability = LocalWorkspace(
        root, authority='none', path_authority={'~/.config/app': 'operator', 'vendor/ours': 'user'}
    )
    assert capability.path_authority == {str(home / '.config' / 'app'): 'operator', 'vendor/ours': 'user'}
    agent = Agent(TestModel(call_tools=['probe']), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> list[Authority]:
        return [
            await ctx.workspace.authority_of(path)
            for path in ('AGENTS.md', 'vendor/ours/AGENTS.md', str(home / '.config' / 'app' / 'AGENTS.md'))
        ]

    result = await agent.run('go')

    assert result.output == '{"probe":["none","user","operator"]}'
    assert workspace_layers(result.workspace) == [AuthorityWorkspace, LocalWorkspaceBackend]


async def test_local_workspace_path_authority_alone_adds_the_wrapper(root: Path) -> None:
    agent = Agent(TestModel(), capabilities=[LocalWorkspace(root, path_authority={'vendor': 'none'})])

    result = await agent.run('go')

    assert workspace_layers(result.workspace) == [AuthorityWorkspace, LocalWorkspaceBackend]
    assert result.workspace.authority == 'operator'
    assert await result.workspace.authority_of('vendor/AGENTS.md') == 'none'
    assert workspace_layers((await Agent(TestModel(), capabilities=[LocalWorkspace(root)]).run('go')).workspace) == [
        LocalWorkspaceBackend
    ]
