"""Native transfer behavior through the public Workspace and SDK boundary."""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
from typing import Any

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from render.client.errors import ClientError, RateLimitError, ServerError
    from render.experimental.sandbox import SandboxDownloadError, SandboxFileNotFoundError, SandboxNotFoundError

    from pydantic_ai.workspaces import (
        Workspace,
        WorkspaceError,
        WorkspaceTimeoutError,
        WorkspaceUnavailableError,
    )

    from .workspace_helpers import WorkspaceHarness

pytestmark = pytest.mark.skipif(not imports_successful(), reason='render not installed')


async def test_native_bytes_do_not_travel_in_commands(workspace_harness: WorkspaceHarness) -> None:
    workspace = Workspace(workspace_harness.backend())
    data = os.urandom(8192)
    await workspace.write_bytes('native.bin', data)
    assert await workspace.read_bytes('native.bin') == data
    assert [call[0] for call in workspace_harness.transfers] == ['upload', 'download']
    encodings = [data.hex(), base64.b64encode(data).decode(), data.decode('latin-1')]
    assert not any(encoded in str(command) for command in workspace_harness.commands for encoded in encodings)
    assert not list(workspace_harness.root.glob('.render-pydantic-*'))
    for direction, _, source, target in workspace_harness.transfers:
        remote = target if direction == 'upload' else source
        assert not Path(remote).parent.exists()
        local = source if direction == 'upload' else target
        assert not Path(local).parent.exists()


@pytest.mark.parametrize('data', [b'', b"\0\xff'\n"])
async def test_empty_binary_and_quoted_names(data: bytes, workspace_harness: WorkspaceHarness) -> None:
    workspace = Workspace(workspace_harness.backend())
    path = "nested/' $();\nfile"
    await workspace.write_bytes(path, data)
    assert await workspace.read_bytes(path) == data


async def test_overwrite_preserves_mode_and_dangling_symlink(
    workspace_harness: WorkspaceHarness,
) -> None:
    root = workspace_harness.root
    target = root / 'target'
    target.write_bytes(b'before')
    target.chmod(0o751)
    workspace = Workspace(workspace_harness.backend())
    await workspace.write_bytes('target', b'after')
    assert target.stat().st_mode & 0o777 == 0o751
    link = root / 'link'
    link.symlink_to('missing')
    await workspace.write_bytes('link', b'through')
    assert link.is_symlink()
    assert (root / 'missing').read_bytes() == b'through'


@pytest.mark.parametrize('direction', ['read', 'write'])
@pytest.mark.parametrize(
    ('failure', 'expected'),
    [
        (SandboxFileNotFoundError('missing'), FileNotFoundError),
        (SandboxNotFoundError('gone'), WorkspaceUnavailableError),
        (RateLimitError('quota'), RateLimitError),
        (ClientError('denied'), WorkspaceError),
        (SandboxDownloadError('unsafe archive'), WorkspaceError),
        (ServerError('transient'), ServerError),
        (TimeoutError('deadline'), WorkspaceTimeoutError),
    ],
    ids=repr,
)
async def test_sdk_errors_are_not_replayed(
    direction: str,
    failure: Exception,
    expected: type[Exception],
    workspace_harness: WorkspaceHarness,
) -> None:
    path = workspace_harness.root / 'existing'
    path.write_bytes(b'original')
    workspace_harness.transfer_error = failure
    workspace = Workspace(workspace_harness.backend())
    with pytest.raises(expected):
        if direction == 'read':
            await workspace.read_bytes('existing')
        else:
            await workspace.write_bytes('existing', b'replacement')
    assert len(workspace_harness.transfers) == 1
    assert path.read_bytes() == b'original'
    _, _, source, target = workspace_harness.transfers[0]
    assert not Path(source if direction == 'read' else target).parent.exists()


async def test_cancelled_upload_preserves_old_file(workspace_harness: WorkspaceHarness) -> None:
    workspace_harness.transfer_gate = asyncio.Event()
    path = workspace_harness.root / 'existing'
    path.write_bytes(b'original')
    backend = workspace_harness.backend()
    task = asyncio.create_task(Workspace(backend).write_bytes('existing', b'replacement'))
    await workspace_harness.transfer_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert path.read_bytes() == b'original'
    assert len(workspace_harness.transfers) == 1
    assert not Path(workspace_harness.transfers[0][3]).parent.exists()
    assert (await backend.run(['printf', 'usable'])).stdout == 'usable'


async def test_read_snapshot_does_not_follow_source_growth(
    workspace_harness: WorkspaceHarness,
) -> None:
    source = workspace_harness.root / 'growing'
    source.write_bytes(b'original')
    workspace_harness.transfer_gate = asyncio.Event()
    task = asyncio.create_task(Workspace(workspace_harness.backend()).read_bytes('growing'))
    await workspace_harness.transfer_started.wait()
    source.write_bytes(b'much larger replacement')
    workspace_harness.transfer_gate.set()
    assert await task == b'original'


@pytest.mark.parametrize('direction', ['read', 'write'])
async def test_incomplete_transfer_is_not_success(
    direction: str,
    workspace_harness: WorkspaceHarness,
) -> None:
    path = workspace_harness.root / 'existing'
    path.write_bytes(b'original')
    workspace_harness.transfer_gate = asyncio.Event()
    workspace = Workspace(workspace_harness.backend())
    task = asyncio.create_task(
        workspace.read_bytes('existing') if direction == 'read' else workspace.write_bytes('existing', b'replacement')
    )
    await workspace_harness.transfer_started.wait()
    # Corrupt the source at the SDK boundary, after the expected size was recorded.
    Path(workspace_harness.transfers[0][2]).write_bytes(b'short')
    workspace_harness.transfer_gate.set()
    with pytest.raises(WorkspaceError if direction == 'read' else OSError):
        await task
    assert path.read_bytes() == b'original'


@pytest.mark.parametrize('path', ['relative', '/contains\0nul'])
async def test_backend_file_paths_are_validated_before_creation(workspace_harness: WorkspaceHarness, path: str) -> None:
    backend = workspace_harness.backend()
    with pytest.raises(ValueError, match='absolute and contain no NUL'):
        await backend.read_bytes(path)
    assert workspace_harness.created == []


@pytest.mark.parametrize('response', ['{invalid', '[]', '{"size": -1}', '{"size": "wrong"}'])
async def test_malformed_file_helper_result_cannot_replace_existing_file(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch, response: str
) -> None:
    from pydantic_ai.workspaces import CommandResult, WorkspaceCommand

    original = workspace_harness.execute

    async def malformed(sandbox_id: str, command: WorkspaceCommand, **kwargs: Any) -> CommandResult:
        # Readiness remains normal; only the remote file helper returns unusable metadata.
        if kwargs['working_dir'] == '/':
            return await original(sandbox_id, command, **kwargs)
        return CommandResult(exit_code=0, stdout=response, stderr='')

    monkeypatch.setattr(workspace_harness, 'execute', malformed)
    target = workspace_harness.root / 'existing'
    target.write_bytes(b'original')
    backend = workspace_harness.backend()
    with pytest.raises(WorkspaceError, match='file prepare returned an invalid result'):
        await backend.write_bytes(str(target), b'replacement')
    assert target.read_bytes() == b'original'
    assert workspace_harness.transfers == []


@pytest.mark.parametrize('operation', ['read', 'write'])
async def test_native_file_operations_reject_named_pipes(workspace_harness: WorkspaceHarness, operation: str) -> None:
    import errno

    target = workspace_harness.root / 'pipe'
    os.mkfifo(target)
    backend = workspace_harness.backend()
    # A file operation must reject a pipe promptly, rather than wait for another process.
    async with asyncio.timeout(10):
        with pytest.raises(OSError) as caught:
            if operation == 'read':
                await backend.read_bytes(str(target))
            else:
                await backend.write_bytes(str(target), b'data')
    assert caught.value.errno == errno.EINVAL
    assert workspace_harness.transfers == []


async def test_file_shrinking_during_snapshot_is_not_partial_success(workspace_harness: WorkspaceHarness) -> None:
    import errno

    target = workspace_harness.root / 'shrinking'
    target.write_bytes(b'original')
    original_stat = os.fstat

    def shrink_during_read(child: pytest.MonkeyPatch, action: str) -> None:
        if action == 'read':

            def snapshot_stat(fd: int) -> os.stat_result:
                state = original_stat(fd)
                # Return the observed size, then simulate a concurrent writer shortening the file.
                with target.open('r+b') as writer:
                    writer.truncate(max(0, state.st_size - 1))
                return state

            child.setattr(os, 'fstat', snapshot_stat)

    workspace_harness.helper_fault = shrink_during_read
    backend = workspace_harness.backend()
    with pytest.raises(OSError) as caught:
        await backend.read_bytes(str(target))
    assert caught.value.errno == errno.EIO
    assert workspace_harness.transfers == []


async def test_cleanup_removes_the_temporary_file_a_killed_commit_left(workspace_harness: WorkspaceHarness) -> None:
    import errno

    target = workspace_harness.root / 'existing'
    target.write_bytes(b'original')

    def kill_during_commit(child: pytest.MonkeyPatch, action: str) -> None:
        if action == 'commit':
            # A stop kills the helper before its `finally` runs: the replace fails and the temporary file stays.
            def killed(source: str, destination: str) -> None:
                raise OSError(errno.EIO, 'killed')

            def never_exists(path: str) -> bool:
                return False

            child.setattr(os, 'replace', killed)
            child.setattr(os.path, 'lexists', never_exists)

    workspace_harness.helper_fault = kill_during_commit
    with pytest.raises(OSError):
        await workspace_harness.backend().write_bytes(str(target), b'replacement')
    assert sorted(path.name for path in workspace_harness.root.iterdir()) == ['existing']
    assert target.read_bytes() == b'original'


async def test_new_file_gets_the_default_mode(workspace_harness: WorkspaceHarness) -> None:
    umask = os.umask(0)
    os.umask(umask)
    target = workspace_harness.root / 'new'
    await workspace_harness.backend().write_bytes(str(target), b'data')
    assert target.stat().st_mode & 0o777 == 0o666 & ~umask


async def test_replacement_stays_private_until_it_takes_the_old_mode(workspace_harness: WorkspaceHarness) -> None:
    import shutil
    import stat

    target = workspace_harness.root / 'existing'
    target.write_bytes(b'original')
    target.chmod(0o644)
    modes_while_filling: list[int] = []
    copy = shutil.copyfileobj

    def observe_commit(child: pytest.MonkeyPatch, action: str) -> None:
        if action == 'commit':

            def observed(source: Any, output: Any, length: int = 0) -> None:
                modes_while_filling.append(stat.S_IMODE(os.fstat(output.fileno()).st_mode))
                copy(source, output, length)

            child.setattr(shutil, 'copyfileobj', observed)

    workspace_harness.helper_fault = observe_commit
    await workspace_harness.backend().write_bytes(str(target), b'replacement')
    assert modes_while_filling == [0o600]
    assert target.stat().st_mode & 0o777 == 0o644
    assert target.read_bytes() == b'replacement'
