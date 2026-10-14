"""Native transfer behavior through the public Workspace and SDK boundary."""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import ParamSpec, TypeVar

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from render.client.errors import ClientError, RateLimitError, ServerError
    from render.experimental.sandbox import SandboxFileNotFoundError, SandboxNotFoundError

    from pydantic_ai.workspaces import (
        SupportsFilesystem,
        Workspace,
        WorkspaceError,
        WorkspaceTimeoutError,
        WorkspaceUnavailableError,
    )
    from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite

    from .workspace_helpers import WorkspaceHarness

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='render not installed'),
    pytest.mark.subprocess(reason='Exercise real POSIX workspace commands and file helper processes', conditional=True),
]

_P = ParamSpec('_P')
_T = TypeVar('_T')


# Exercise the released rules directly. The complete suite also runs live; these
# fast cases catch file semantics before creating any remote resources.
_RULES = [
    'test_large_file_round_trip',
    'test_bytes_round_trip_and_write_creates_parents',
    'test_exists',
    'test_stat_and_list_dir',
    'test_make_dir_creates_parents_and_is_idempotent',
    'test_commands_and_files_share_one_environment',
    'test_missing_paths_raise_file_not_found',
    'test_reading_a_directory_raises_is_a_directory',
    'test_listing_a_file_raises_not_a_directory',
    'test_writing_to_a_directory_raises_is_a_directory',
    'test_symlink_loop_does_not_break_listing',
    'test_fifo_read_does_not_wait_for_writer',
    'test_permission_denied_uses_builtin_error',
    'test_file_as_parent_raises_not_a_directory',
    'test_making_a_directory_over_a_file_raises_file_exists',
    'test_remove_deletes_a_file_or_a_tree',
    'test_remove_refuses_the_working_dir_and_its_ancestors',
    'test_realpath_and_entries_follow_symlinks',
    'test_writes_go_through_a_symlink',
]


@pytest.mark.parametrize('rule', _RULES)
async def test_public_file_contract(rule: str, workspace_harness: WorkspaceHarness) -> None:
    backend = workspace_harness.backend()
    assert isinstance(backend, SupportsFilesystem)
    method = getattr(WorkspaceBackendSuite(), rule)
    arguments = {name: True for name in inspect.signature(method).parameters if name != 'backend'}
    await method(backend=backend, **arguments)


async def test_native_bytes_do_not_travel_in_commands(workspace_harness: WorkspaceHarness) -> None:
    workspace = Workspace(workspace_harness.backend())
    data = os.urandom(8192)
    await workspace.write_bytes('native.bin', data)
    assert await workspace.read_bytes('native.bin') == data
    assert [call[0] for call in workspace_harness.transfers] == ['upload', 'download']
    # Keep the control-request budget constant instead of chunking the payload.
    assert len(workspace_harness.commands) <= 8
    assert all(len(str(command)) < 16_000 for command in workspace_harness.commands)
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
        (ServerError('transient'), WorkspaceError),
    ],
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


@pytest.mark.parametrize('stop', ['timeout', 'cancel', 'destroy'])
async def test_interrupted_upload_preserves_old_file(
    stop: str,
    workspace_harness: WorkspaceHarness,
) -> None:
    workspace_harness.config = replace(workspace_harness.config, operation_timeout=1)
    workspace_harness.transfer_gate = asyncio.Event()
    path = workspace_harness.root / 'existing'
    path.write_bytes(b'original')
    backend = workspace_harness.backend()
    task = asyncio.create_task(Workspace(backend).write_bytes('existing', b'replacement'))
    await workspace_harness.transfer_started.wait()
    if stop == 'cancel':
        task.cancel()
    elif stop == 'destroy':
        await backend.destroy()
    with pytest.raises(WorkspaceTimeoutError if stop == 'timeout' else asyncio.CancelledError):
        await task
    assert path.read_bytes() == b'original'
    assert len(workspace_harness.transfers) == 1
    if stop == 'destroy':
        assert workspace_harness.terminated == ['sb-1']
        # The fake termination has no real remote filesystem to delete.
        import shutil

        shutil.rmtree(Path(workspace_harness.transfers[0][3]).parent)
    else:
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


async def test_transfers_share_command_pacing(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from itertools import pairwise

    interval = 0.015
    workspace_harness.config = replace(workspace_harness.config, command_interval=interval)
    dispatches: list[float] = []
    control = workspace_harness.client.experimental.sandboxes
    command = control.exec
    upload, download = control.copy_to, control.copy_from

    def record(operation: Callable[_P, Awaitable[_T]]) -> Callable[_P, Awaitable[_T]]:
        async def recorded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            dispatches.append(asyncio.get_running_loop().time())
            return await operation(*args, **kwargs)

        return recorded

    def record_exec(identifier: str, launch: str):
        dispatches.append(asyncio.get_running_loop().time())
        return command(identifier, launch)

    monkeypatch.setattr(control, 'exec', record_exec)
    monkeypatch.setattr(control, 'copy_to', record(upload))
    monkeypatch.setattr(control, 'copy_from', record(download))
    first = Workspace(workspace_harness.backend())
    second = Workspace(workspace_harness.backend())
    await asyncio.gather(first.write_bytes('one', b'one'), second.write_bytes('two', b'two'))
    assert await first.read_bytes('one') == b'one'
    assert all(b - a >= interval - 0.001 for a, b in pairwise(dispatches))


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
    from render import RenderAsync

    from pydantic_ai.workspaces import CommandResult, WorkspaceCommand

    original = workspace_harness.execute

    async def malformed(
        client: RenderAsync, sandbox_id: str, command: WorkspaceCommand, **kwargs: object
    ) -> CommandResult:
        # Readiness remains normal; only the remote file helper returns unusable metadata.
        if kwargs['working_dir'] == '/':
            return await original(client, sandbox_id, command, **kwargs)
        return CommandResult(exit_code=0, stdout=response, stderr='')

    monkeypatch.setattr(workspace_harness, 'execute', malformed)
    target = workspace_harness.root / 'existing'
    target.write_bytes(b'original')
    backend = workspace_harness.backend()
    try:
        with pytest.raises(WorkspaceError, match='file prepare returned an invalid result'):
            await backend.write_bytes(str(target), b'replacement')
        assert target.read_bytes() == b'original'
        assert workspace_harness.transfers == []
    finally:
        await backend.destroy()


async def test_expired_sandbox_during_upload_remains_unavailable(workspace_harness: WorkspaceHarness) -> None:
    import shutil

    from pydantic_ai.workspaces import WorkspaceUnavailableError

    # Let preparation finish, then expire the provider before its upload completes.
    workspace_harness.transfer_gate = asyncio.Event()
    backend = workspace_harness.backend()
    target = workspace_harness.root / 'existing'
    target.write_bytes(b'original')
    pending = asyncio.create_task(backend.write_bytes(str(target), b'replacement'))
    try:
        await workspace_harness.transfer_started.wait()
        assert backend.ref is not None
        workspace_harness.sandboxes[backend.ref.id].status = 'terminated'
        workspace_harness.transfer_gate.set()
        with pytest.raises(WorkspaceUnavailableError):
            await pending
        assert target.read_bytes() == b'original'
        assert len(workspace_harness.created) == 1
    finally:
        workspace_harness.transfer_gate.set()
        await asyncio.gather(pending, return_exceptions=True)
        await backend.destroy()
        # The local SDK double does not remove remote staging when it marks termination.
        shutil.rmtree(Path(workspace_harness.transfers[0][3]).parent, ignore_errors=True)


@pytest.mark.skipif(os.geteuid() == 0, reason='root bypasses file write permissions')
async def test_native_write_respects_existing_file_permissions(workspace_harness: WorkspaceHarness) -> None:
    target = workspace_harness.root / 'protected'
    target.write_bytes(b'original')
    target.chmod(0o400)
    backend = workspace_harness.backend()
    try:
        with pytest.raises(PermissionError):
            await backend.write_bytes(str(target), b'replacement')
        assert target.read_bytes() == b'original'
    finally:
        target.chmod(0o600)
        await backend.destroy()


@pytest.mark.parametrize('operation', ['read', 'write'])
async def test_native_file_operations_reject_named_pipes(workspace_harness: WorkspaceHarness, operation: str) -> None:
    import errno

    target = workspace_harness.root / 'pipe'
    os.mkfifo(target)
    backend = workspace_harness.backend()
    try:
        # A file operation must reject a pipe promptly, rather than wait for another process.
        async with asyncio.timeout(10):
            with pytest.raises(OSError) as caught:
                if operation == 'read':
                    await backend.read_bytes(str(target))
                else:
                    await backend.write_bytes(str(target), b'data')
        assert caught.value.errno == errno.EINVAL
        assert workspace_harness.transfers == []
    finally:
        await backend.destroy()


def interpret_file_helpers(
    workspace_harness: WorkspaceHarness,
    monkeypatch: pytest.MonkeyPatch,
    fault: Callable[[pytest.MonkeyPatch, str], None],
) -> None:
    """Execute the exact dispatched program while controlling its local OS boundary."""
    import io
    import runpy
    import sys
    from contextlib import redirect_stdout
    from importlib.resources import files

    from render import RenderAsync

    from pydantic_ai.workspaces import CommandResult, WorkspaceCommand

    original = workspace_harness.execute
    program = files('pydantic_ai_harness.render_sandbox').joinpath('_file_transfer.py')

    async def interpret(
        client: RenderAsync, sandbox_id: str, command: WorkspaceCommand, **kwargs: object
    ) -> CommandResult:
        if isinstance(command, str) or list(command[:3]) != ['python3', '-c', program.read_text()]:
            return await original(client, sandbox_id, command, **kwargs)
        # Keep OS injection inside this one synchronous program invocation, so it
        # cannot affect Pydantic, SDK transport, or another test's file operations.
        output = io.StringIO()
        exit_code = 0
        with monkeypatch.context() as child:
            child.setattr(sys, 'argv', [str(program), *command[3:]])
            fault(child, command[3])
            with redirect_stdout(output):
                try:
                    runpy.run_path(str(program), run_name='__main__')
                except SystemExit as error:
                    assert isinstance(error.code, int)
                    exit_code = error.code
        return CommandResult(exit_code=exit_code, stdout=output.getvalue(), stderr='')

    monkeypatch.setattr(workspace_harness, 'execute', interpret)


async def test_file_shrinking_during_snapshot_is_not_partial_success(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    interpret_file_helpers(workspace_harness, monkeypatch, shrink_during_read)
    backend = workspace_harness.backend()
    try:
        with pytest.raises(OSError) as caught:
            await backend.read_bytes(str(target))
        assert caught.value.errno == errno.EIO
        assert workspace_harness.transfers == []
    finally:
        await backend.destroy()


async def test_destination_becoming_dangling_link_does_not_create_its_target(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    target = workspace_harness.root / 'changing'
    target.write_bytes(b'original')
    original_stat = os.stat

    def replace_destination(child: pytest.MonkeyPatch, action: str) -> None:
        if action == 'prepare':

            def changed_stat(path: str, *, dir_fd: int | None = None, follow_symlinks: bool = True) -> os.stat_result:
                if path == str(target):  # pragma: no branch - other interpreter file checks only pass through
                    # The write resolved a regular file; another writer replaces it
                    # with a dangling link before the subsequent file check.
                    target.unlink()
                    target.symlink_to('must-not-create')
                return original_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

            child.setattr(os, 'stat', changed_stat)

    interpret_file_helpers(workspace_harness, monkeypatch, replace_destination)
    backend = workspace_harness.backend()
    try:
        with pytest.raises(OSError) as caught:
            await backend.write_bytes(str(target), b'replacement')
        assert caught.value.errno == errno.ELOOP
        assert not (workspace_harness.root / 'must-not-create').exists()
        assert workspace_harness.transfers == []
    finally:
        await backend.destroy()
