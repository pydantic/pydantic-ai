"""Tests for the read-only workspace policy wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import UserError
from pydantic_ai.workspaces import LocalWorkspace, ReadOnlyWorkspace, Workspace, WorkspaceRef

from .workspace_fakes import FakeWorkspace, RunOnlyWorkspaceBackend

pytestmark = pytest.mark.anyio


async def test_read_only_workspace_forwards_reads_and_refuses_run_and_writes() -> None:
    ref = WorkspaceRef(provider='fake', id='existing')
    backend = FakeWorkspace('read-only', {'/workspace/data.txt': b'original'}, ref=ref)
    workspace = Workspace(ReadOnlyWorkspace(Workspace(backend)))

    assert await workspace.read_text('data.txt') == 'original'
    assert workspace.ref == ref

    with pytest.raises(UserError, match='read-only'):
        await workspace.run(['rm', 'data.txt'])
    with pytest.raises(UserError, match='read-only'):
        await workspace.write_text('data.txt', 'changed')

    assert backend.files['/workspace/data.txt'] == b'original'


async def test_read_only_workspace_forwards_every_read_and_refuses_every_write() -> None:
    backend = FakeWorkspace('read-only', {'/workspace/data.txt': b'original'})
    workspace = Workspace(ReadOnlyWorkspace(Workspace(backend)))

    assert (await workspace.stat('data.txt')).name == 'data.txt'
    assert [entry.name for entry in await workspace.list_dir('/workspace')] == ['data.txt']
    assert await workspace.exists('data.txt') is True
    assert await workspace.working_dir() == '/workspace'

    with pytest.raises(UserError, match='read-only'):
        await workspace.make_dir('new-dir')
    with pytest.raises(UserError, match='read-only'):
        await workspace.remove('data.txt')

    assert backend.files == {'/workspace/data.txt': b'original'}


async def test_read_only_workspace_over_a_run_only_backend_uses_the_shell_fallback(tmp_path: Path) -> None:
    """The wrapper can read through the inner shell fallback without exposing command execution."""
    (tmp_path / 'data.txt').write_text('hello')
    backend = RunOnlyWorkspaceBackend(LocalWorkspace(tmp_path))
    workspace = Workspace(ReadOnlyWorkspace(Workspace(backend)))

    assert await workspace.read_text('data.txt') == 'hello'
    assert any(isinstance(command, str) and command.startswith('base64 <') for command in backend.commands)
    with pytest.raises(UserError, match='read-only'):
        await workspace.run(['ls'])
    with pytest.raises(UserError, match='read-only'):
        await workspace.write_text('data.txt', 'changed')
