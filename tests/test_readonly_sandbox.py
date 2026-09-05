"""Tests for the read-only sandbox policy wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import UserError
from pydantic_ai.sandboxes import LocalSandbox, ReadOnlySandbox, Sandbox, SandboxRef

from .sandbox_fakes import FakeSandbox, RunOnlySandboxBackend

pytestmark = pytest.mark.anyio


async def test_read_only_sandbox_forwards_reads_and_refuses_run_and_writes() -> None:
    ref = SandboxRef(sandbox_id='existing')
    backend = FakeSandbox('read-only', {'/workspace/data.txt': b'original'}, ref=ref)
    sandbox = Sandbox(ReadOnlySandbox(backend))

    assert await sandbox.read_text('data.txt') == 'original'
    assert sandbox.ref == ref

    with pytest.raises(UserError, match='read-only'):
        await sandbox.run(['rm', 'data.txt'])
    with pytest.raises(UserError, match='read-only'):
        await sandbox.write_text('data.txt', 'changed')

    assert backend.files['/workspace/data.txt'] == b'original'


async def test_read_only_sandbox_forwards_every_read_and_refuses_every_write() -> None:
    backend = FakeSandbox('read-only', {'/workspace/data.txt': b'original'})
    sandbox = Sandbox(ReadOnlySandbox(backend))

    assert (await sandbox.stat('data.txt')).name == 'data.txt'
    assert [entry.name for entry in await sandbox.list_dir('/workspace')] == ['data.txt']
    assert await sandbox.exists('data.txt') is True
    assert await sandbox.working_dir() == '/workspace'

    with pytest.raises(UserError, match='read-only'):
        await sandbox.make_dir('new-dir')
    with pytest.raises(UserError, match='read-only'):
        await sandbox.remove('data.txt')

    assert backend.files == {'/workspace/data.txt': b'original'}


async def test_read_only_sandbox_over_a_run_only_backend_uses_the_shell_fallback(tmp_path: Path) -> None:
    """The wrapper can read through the inner shell fallback without exposing command execution."""
    (tmp_path / 'data.txt').write_text('hello')
    backend = RunOnlySandboxBackend(LocalSandbox(tmp_path))
    sandbox = Sandbox(ReadOnlySandbox(backend))

    assert await sandbox.read_text('data.txt') == 'hello'
    assert any(isinstance(command, str) and command.startswith('base64 <') for command in backend.commands)
    with pytest.raises(UserError, match='read-only'):
        await sandbox.run(['ls'])
    with pytest.raises(UserError, match='read-only'):
        await sandbox.write_text('data.txt', 'changed')
