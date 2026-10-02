"""Tests for the workspace interface and its lazy backend contract."""

from __future__ import annotations

import asyncio
import base64
import math
import os
import re
import shutil
import stat
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio
import anyio.to_thread
import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent, RunContext, UserError, capture_run_messages
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, LocalWorkspace, WrapperCapability
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    LocalWorkspaceBackend,
    ReadOnlyWorkspace,
    SupportsCommands,
    SupportsFilesystem,
    UnavailableWorkspace,
    Workspace,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceReadOnlyError,
    WorkspaceRef,
    WorkspaceUnavailableError,
    WrapperWorkspace,
    local as local_module,
)
from pydantic_ai.workspaces.workspace import workspace_layers

from .workspace_fakes import (
    ConnectOnlyWorkspaceCapability,
    DecliningWorkspaceCapability,
    FakeWorkspace,
    FilesystemOnlyWorkspaceBackend,
    InMemoryProvider,
    RunOnlyWorkspaceBackend,
    WorkspaceCapability,
)


@pytest.mark.parametrize('timeout', [-1, 0, math.nan, math.inf, '5'])
async def test_facade_rejects_invalid_timeout_before_backend(timeout: Any) -> None:
    with pytest.raises(ValueError, match='timeout must be a positive finite number or None'):
        await Workspace(FakeWorkspace('invalid-timeout')).run(['true'], timeout=timeout)


async def test_wrapper_overrides_apply_to_text_reads():
    backend = FakeWorkspace('wrapper')

    class ReadingWrapper(WrapperWorkspace):
        async def read_bytes(self, path: str) -> bytes:
            return b'outer\nvalue\n'

    workspace = ReadingWrapper(Workspace(backend))
    assert await workspace.read_text('file.txt') == 'outer\nvalue\n'


async def test_wrapper_overrides_apply_to_text_writes():
    backend = FakeWorkspace('wrapper')
    writes: list[tuple[str, bytes]] = []

    class WritingWrapper(WrapperWorkspace):
        async def write_bytes(self, path: str, data: bytes) -> None:
            writes.append((path, data))

    workspace = WritingWrapper(Workspace(backend))
    await workspace.write_text('file.txt', 'outer')
    assert writes == [('file.txt', b'outer')]
    assert backend.files == {}


async def test_stacked_wrappers_preserve_delegation_identity_and_refs():
    ref = WorkspaceRef(provider='fake', id='stacked')
    backend = FakeWorkspace('wrapper', {'/workspace/file.txt': b'inner'}, ref=ref)
    inner = WrapperWorkspace(Workspace(backend))
    outer = WrapperWorkspace(inner)

    assert outer.wrapped is inner
    assert outer.backend is backend
    assert Workspace(outer).backend is backend
    assert outer.ref == ref

    events: list[str] = []

    class LoggedWorkspace(WrapperWorkspace):
        def __init__(self, wrapped: Workspace, name: str):
            super().__init__(wrapped)
            self.name = name

        async def read_bytes(self, path: str) -> bytes:
            events.append(f'{self.name} before')
            data = await self.wrapped.read_bytes(path)
            events.append(f'{self.name} after')
            return data

    inner_logged = LoggedWorkspace(Workspace(backend), 'inner')
    outer_logged = LoggedWorkspace(inner_logged, 'outer')
    assert events == []
    assert await outer_logged.read_text('file.txt') == 'inner'
    assert events == ['outer before', 'inner before', 'inner after', 'outer after']


def _tool_call_model(tool_name: str = 'probe') -> FunctionModel:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, {})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(model)


async def test_resolve_normalizes_paths_against_the_working_directory() -> None:
    workspace = Workspace(FakeWorkspace('resolve'))

    assert await workspace.resolve('sub/../notes.txt') == '/workspace/notes.txt'
    assert await workspace.resolve('/abs/./x') == '/abs/x'
    assert await workspace.resolve('x', base='/elsewhere') == '/elsewhere/x'


async def test_resolve_rejects_a_relative_base() -> None:
    with pytest.raises(ValueError, match="base must be an absolute path, got 'relative'"):
        await Workspace(FakeWorkspace('resolve')).resolve('file.txt', base='relative')


@pytest.mark.parametrize(
    'path',
    [
        'dir_link/missing/new.txt',
        'file_link',
        # `..` climbs from the link's target, where the kernel would, not from the link's own directory.
        'dir_link/..',
        # A missing component is climbed out of, and the symlink after it still resolves.
        'missing/../dir_link/escape.txt',
        # A target whose name ends in a newline must not collapse onto its sibling without one.
        'newline_link/escape.txt',
        'dangling_link',
        'deep_dangling_link/x',
        '/pydantic-ai-missing/file.txt',
    ],
)
@pytest.mark.parametrize('native', [True, False], ids=['native', 'shell'])
async def test_realpath_resolves_symlinks_the_way_the_environment_does(tmp_path: Path, native: bool, path: str) -> None:
    """Native and shell-derived `realpath` both agree with `os.path.realpath(strict=False)`."""
    data = tmp_path / 'elsewhere' / 'data'
    data.mkdir(parents=True)
    (data / 'file.txt').write_text('x')
    (tmp_path / 'safe\n').mkdir()
    root = tmp_path / 'safe'
    root.mkdir()
    (root / 'dir_link').symlink_to(data)
    (root / 'file_link').symlink_to(data / 'file.txt')
    (root / 'newline_link').symlink_to(tmp_path / 'safe\n')
    (root / 'dangling_link').symlink_to(tmp_path / 'gone.txt')
    (root / 'deep_dangling_link').symlink_to(tmp_path / 'gone' / 'deeper')
    backend = LocalWorkspaceBackend(root)
    workspace = Workspace(backend if native else RunOnlyWorkspaceBackend(backend))

    assert await workspace.realpath(path) == os.path.realpath(root.resolve() / path)


@pytest.mark.parametrize('native', [True, False], ids=['native', 'shell'])
async def test_realpath_of_resolve_names_the_file_a_file_method_opens(tmp_path: Path, native: bool) -> None:
    """File methods collapse `..` as text, so `realpath(resolve(p))`, not `realpath(p)`, names the file they open."""
    root = tmp_path / 'root'
    (root / 'a' / 'b').mkdir(parents=True)
    (root / 'inlink').symlink_to(root / 'a' / 'b')
    (tmp_path / 'x').write_text('beside the root')
    (root / 'x').write_text('inside the root')
    backend = LocalWorkspaceBackend(root)
    workspace = Workspace(backend if native else RunOnlyWorkspaceBackend(backend))
    path = 'inlink/../../x'

    assert await workspace.read_text(path) == 'beside the root'
    assert await workspace.realpath(await workspace.resolve(path)) == str((tmp_path / 'x').resolve())
    # On the raw path, `..` climbs from the link's target, as a command would.
    assert await workspace.realpath(path) == str((root / 'x').resolve())


async def test_run_rejects_an_empty_argv(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='command must not be empty'):
        await Workspace(LocalWorkspaceBackend(tmp_path)).run([])


async def test_shell_read_output_limit_names_file_operation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'large').write_bytes(b'x' * (80 * 1024))
    monkeypatch.setattr(local_module, '_MAX_CAPTURE_BYTES', 50 * 1024)
    workspace = Workspace(RunOnlyWorkspaceBackend(LocalWorkspaceBackend(tmp_path)))
    with pytest.raises(WorkspaceError, match='shell filesystem read exceeded command output limit'):
        await workspace.read_bytes('large')


async def test_shell_listing_output_limit_names_listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for index in range(400):
        (tmp_path / (f'file-{index:04d}-' + 'x' * 100)).write_bytes(b'')
    monkeypatch.setattr(local_module, '_MAX_CAPTURE_BYTES', 50 * 1024)
    workspace = Workspace(RunOnlyWorkspaceBackend(LocalWorkspaceBackend(tmp_path)))
    with pytest.raises(WorkspaceError, match='shell filesystem listing exceeded command output limit'):
        await workspace.list_dir('.')


async def test_shell_listing_larger_than_command_output_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for index in range(600):
        (tmp_path / (f'file-{index:04d}-' + 'x' * 100)).write_bytes(b'')
    monkeypatch.setattr(local_module, '_MAX_CAPTURE_BYTES', 120 * 1024)
    workspace = Workspace(RunOnlyWorkspaceBackend(LocalWorkspaceBackend(tmp_path)))
    assert len(await workspace.list_dir('.')) == 600


async def test_shell_listing_uses_configured_temporary_directory(tmp_path: Path) -> None:
    temporary = tmp_path / 'temp'
    temporary.mkdir()

    class TemporaryBackend(RunOnlyWorkspaceBackend):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> CommandResult:
            assert '/tmp/.pydantic-ai-' not in str(command)
            return await super().run(command, shell=shell, env={'TMPDIR': str(temporary)}, timeout=timeout)

    workspace = Workspace(TemporaryBackend(LocalWorkspaceBackend(tmp_path)))
    assert [entry.name for entry in await workspace.list_dir('.')] == ['temp']
    assert list(temporary.iterdir()) == []


async def test_shell_listing_removes_scratch_file_on_cancel(tmp_path: Path) -> None:
    started = asyncio.Event()
    temporary = tmp_path / 'temp'
    temporary.mkdir()

    class InterruptedBackend(RunOnlyWorkspaceBackend):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> CommandResult:
            if isinstance(command, str) and 'find ' in command:
                match = re.search(r'\.pydantic-ai-[a-f0-9]+\.list', command)
                assert match is not None
                await anyio.to_thread.run_sync((temporary / match.group()).write_bytes, b'partial')
                started.set()
                await asyncio.Event().wait()
            return await super().run(command, shell=shell, env={'TMPDIR': str(temporary)}, timeout=timeout)

    workspace = Workspace(InterruptedBackend(LocalWorkspaceBackend(tmp_path)))
    task = asyncio.create_task(workspace.list_dir('.'))
    try:
        with anyio.fail_after(30):
            await started.wait()
    finally:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert list(temporary.iterdir()) == []


async def test_shell_listing_preserves_non_utf8_filename(tmp_path: Path) -> None:
    class ByteListingBackend(RunOnlyWorkspaceBackend):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> CommandResult:
            listing = b'-/workspace/file-\xff\0'
            return CommandResult(exit_code=0, stdout=f'{len(listing)}\n{base64.b64encode(listing).decode()}', stderr='')

    workspace = Workspace(ByteListingBackend(LocalWorkspaceBackend(tmp_path)))
    assert [entry.name for entry in await workspace.list_dir('.')] == ['file-\udcff']


async def test_shell_realpath_keeps_a_non_utf8_filename(tmp_path: Path) -> None:
    # `list_dir` returns such a name with surrogate escapes; `write_bytes` resolves it through `realpath`.
    workspace = Workspace(RunOnlyWorkspaceBackend(LocalWorkspaceBackend(tmp_path)))
    assert await workspace.realpath('file-\udcff') == f'{os.path.realpath(tmp_path)}/file-\udcff'


async def test_shell_filesystem_reports_permission_denied(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip('root bypasses filesystem permissions')  # pragma: no cover
    (tmp_path / 'unreadable').write_bytes(b'x')
    (tmp_path / 'unreadable').chmod(0)
    workspace = Workspace(RunOnlyWorkspaceBackend(LocalWorkspaceBackend(tmp_path)))
    with pytest.raises(PermissionError):
        await workspace.read_bytes('unreadable')
    with pytest.raises(PermissionError):
        await workspace.write_bytes('unreadable', b'x')
    (tmp_path / 'unwritable').mkdir()
    (tmp_path / 'unwritable').chmod(0o500)
    with pytest.raises(PermissionError):
        await workspace.make_dir('unwritable/child')


async def test_shell_write_never_replaces_a_file_it_cannot_copy_the_mode_of(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip('root bypasses filesystem permissions')  # pragma: no cover
    target = tmp_path / 'write-only'
    target.write_bytes(b'original')
    target.chmod(0o200)
    workspace = Workspace(RunOnlyWorkspaceBackend(LocalWorkspaceBackend(tmp_path)))
    with pytest.raises(WorkspaceError):
        await workspace.write_bytes('write-only', b'replacement')
    assert target.stat().st_mode & 0o777 == 0o200
    target.chmod(0o600)
    assert target.read_bytes() == b'original'
    assert not list(tmp_path.glob('.pydantic-ai-*'))


async def test_shell_filesystem_refuses_to_remove_workspace_root(tmp_path: Path) -> None:
    workspace = Workspace(RunOnlyWorkspaceBackend(LocalWorkspaceBackend(tmp_path)))
    (tmp_path / 'safe').write_bytes(b'safe')
    for path in ('.', str(tmp_path.parent)):
        with pytest.raises(ValueError, match='workspace root'):
            await workspace.remove(path)
    assert (tmp_path / 'safe').read_bytes() == b'safe'


async def test_shell_filesystem_refuses_fifo_without_opening_it(tmp_path: Path) -> None:
    fifo = tmp_path / 'fifo'
    os.mkfifo(fifo)
    workspace = Workspace(RunOnlyWorkspaceBackend(LocalWorkspaceBackend(tmp_path)))
    with anyio.fail_after(30):
        for operation in (workspace.read_bytes, workspace.stat):
            with pytest.raises(OSError, match='not a regular file'):
                await operation('fifo')
        with pytest.raises(OSError, match='not a regular file'):
            await workspace.write_bytes('fifo', b'data')
    assert stat.S_ISFIFO(fifo.lstat().st_mode)


async def test_local_write_refuses_fifo_without_opening_it(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / 'fifo')
    workspace = Workspace(LocalWorkspaceBackend(tmp_path))
    with anyio.fail_after(30):
        with pytest.raises(OSError, match='not a regular file'):
            await workspace.write_bytes('fifo', b'data')


async def test_shell_realpath_leaves_a_symlink_loop_unresolved(tmp_path: Path) -> None:
    (tmp_path / 'loop').symlink_to(tmp_path / 'loop')
    workspace = Workspace(RunOnlyWorkspaceBackend(LocalWorkspaceBackend(tmp_path)))

    (tmp_path / 'loop1').symlink_to('loop2')
    (tmp_path / 'loop2').symlink_to('loop1')
    # Like `os.path.realpath`: a loop is left unresolved rather than failing the call.
    with anyio.fail_after(30):
        assert await workspace.realpath('loop') == str(tmp_path / 'loop')
        assert await workspace.realpath('loop1/q') == str(tmp_path / 'loop1' / 'q')


async def test_shell_realpath_raises_on_a_link_readlink_cannot_read(tmp_path: Path) -> None:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for tool in ('base64', 'wc'):
        found = shutil.which(tool)
        assert found is not None
        (bin_dir / tool).symlink_to(found)
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'escape').symlink_to(tmp_path)
    workspace = Workspace(RunOnlyWorkspaceBackend(LocalWorkspaceBackend(root, env={'PATH': str(bin_dir)})))

    assert await workspace.realpath('plain/file.txt') == str(root / 'plain' / 'file.txt')
    with pytest.raises(WorkspaceError, match='readlink'):
        await workspace.realpath('escape/secret.txt')


async def test_realpath_only_normalizes_on_a_filesystem_only_backend() -> None:
    workspace = Workspace(FilesystemOnlyWorkspaceBackend(FakeWorkspace('files-only')))

    assert await workspace.realpath('sub/../notes.txt') == '/workspace/notes.txt'


async def test_flat_file_operations_use_the_backend_filesystem() -> None:
    backend = FakeWorkspace('files', {'/workspace/data.txt': b'hello'})
    workspace = Workspace(backend)

    assert await workspace.read_bytes('data.txt') == b'hello'
    assert (await workspace.stat('data.txt')).path == '/workspace/data.txt'
    assert await workspace.exists('data.txt')
    assert (await workspace.list_dir('.'))[0].path == '/workspace/data.txt'
    await workspace.make_dir('new-dir')
    await workspace.write_bytes('new.txt', b'new')
    await workspace.write_text('data.txt', 'updated')
    await workspace.remove('new.txt')

    assert backend.files['/workspace/data.txt'] == b'updated'
    assert not await workspace.exists('new.txt')


async def test_filesystem_only_backend_works_without_command_execution() -> None:
    inner = FakeWorkspace('files-only', {'/workspace/data.txt': b'one\ntwo\nthree\n'})
    backend = FilesystemOnlyWorkspaceBackend(inner)
    workspace = Workspace(backend)

    assert isinstance(backend, WorkspaceBackend)
    assert isinstance(backend, SupportsFilesystem)
    assert not isinstance(backend, SupportsCommands)
    await workspace.write_text('new.txt', 'content')
    assert await workspace.read_text('new.txt') == 'content'
    assert (await workspace.stat('new.txt')).size == len('content')
    await workspace.make_dir('nested')
    assert sorted(entry.name for entry in await workspace.list_dir('.')) == ['data.txt', 'nested', 'new.txt']
    await workspace.remove('new.txt')
    assert not await workspace.exists('new.txt')
    assert workspace.ref == WorkspaceRef(provider='fake', id='fake-files-only')
    assert inner.commands == []

    with pytest.raises(UserError, match='does not support command execution'):
        await workspace.run(['true'])


async def test_backend_without_commands_or_filesystem_explains_what_to_attach() -> None:
    class IdentityOnlyBackend(WorkspaceBackend):
        @property
        def ref(self) -> None:
            return None

        async def working_dir(self) -> str:
            return '/workspace'

    workspace = Workspace(IdentityOnlyBackend())

    assert workspace.ref is None
    assert await workspace.working_dir() == '/workspace'
    with pytest.raises(UserError, match='does not support filesystem operations'):
        await workspace.read_text('data.txt')


async def test_backend_with_part_of_the_filesystem_protocol_names_what_it_lacks() -> None:
    class NoExistsBackend(WorkspaceBackend):
        @property
        def ref(self) -> None:
            return None  # pragma: no cover

        async def working_dir(self) -> str:
            return '/workspace'  # pragma: no cover

        async def read_bytes(self, path: str) -> bytes:
            raise NotImplementedError

        async def write_bytes(self, path: str, data: bytes) -> None:
            raise NotImplementedError

        async def stat(self, path: str) -> FileEntry:
            raise NotImplementedError

        async def list_dir(self, path: str) -> Sequence[FileEntry]:
            raise NotImplementedError

        async def make_dir(self, path: str) -> None:
            raise NotImplementedError

        async def remove(self, path: str) -> None:
            raise NotImplementedError

    with pytest.raises(UserError, match=r'part of `SupportsFilesystem` and lacks `exists`\.$'):
        await Workspace(NoExistsBackend()).read_text('data.txt')


@pytest.mark.parametrize(
    ('exit_code', 'stderr', 'match'),
    [(255, 'ssh: connect to host x port 22: Connection refused\n', 'refused'), (127, '', 'exit code 127')],
)
async def test_shell_exists_raises_when_the_command_itself_fails(exit_code: int, stderr: str, match: str) -> None:
    """Only `test`'s own "no" (exit 1) means missing; a broken shell or connection must not read as absent."""

    class BrokenShellBackend(WorkspaceBackend, SupportsCommands):
        @property
        def ref(self) -> None:
            return None  # pragma: no cover

        async def working_dir(self) -> str:
            return '/workspace'

        async def run(
            self,
            command: WorkspaceCommand,
            *,
            shell: bool = False,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> CommandResult:
            return CommandResult(exit_code=exit_code, stdout='', stderr=stderr)

    workspace = Workspace(BrokenShellBackend())
    with pytest.raises(WorkspaceError, match=match):
        await workspace.exists('data.txt')
    with pytest.raises(WorkspaceError, match=match):
        await workspace.stat('data.txt')


async def test_run_only_backend_writes_binary_and_odd_names_through_the_shell(tmp_path: Path) -> None:
    backend = RunOnlyWorkspaceBackend(LocalWorkspaceBackend(tmp_path))
    workspace = Workspace(backend)
    payload = bytes(range(256)) * 800
    filename = "nested/weird '\n blob.bin"

    assert backend.ref == WorkspaceRef(provider='local', id=str(tmp_path))
    await workspace.write_bytes(filename, payload)

    assert await workspace.read_bytes(filename) == payload
    entries = await workspace.list_dir('nested')
    assert [(entry.name, entry.is_dir) for entry in entries] == [("weird '\n blob.bin", False)]
    # The encoded write is chunked below Linux's independent per-argument limit.
    assert max(len(command.encode()) for command in backend.commands if isinstance(command, str)) < 128 * 1024


@pytest.mark.parametrize('native', [True, False], ids=['native', 'shell'])
async def test_entries_follow_symlinks_for_is_dir(tmp_path: Path, native: bool) -> None:
    target = tmp_path / 'target'
    child = target / 'child'
    child.mkdir(parents=True)
    (child / 'file.txt').write_text('content')
    (target / 'child-link').symlink_to(child, target_is_directory=True)
    (target / 'file-link').symlink_to(child / 'file.txt')
    (target / 'dangling-link').symlink_to(tmp_path / 'gone')
    (tmp_path / 'root-link').symlink_to(target, target_is_directory=True)
    backend = LocalWorkspaceBackend(tmp_path)
    workspace = Workspace(backend if native else RunOnlyWorkspaceBackend(backend))

    # Listed through a symlinked directory, which is followed.
    entries = await workspace.list_dir('root-link')

    assert {entry.name: entry.is_dir for entry in entries} == {
        'child': True,
        'child-link': True,
        'dangling-link': False,
        'file-link': False,
    }
    link = await workspace.stat('root-link/file-link')
    assert (link.is_dir, link.size) == (False, len('content'))
    assert (await workspace.stat('root-link/child-link')).is_dir is True


async def test_run_only_filesystem_raises_builtin_path_errors_in_one_command(tmp_path: Path) -> None:
    """Classifying the path inside the read or listing command keeps the error to one round trip.

    `base64 < directory` exits 0 with empty output on macOS and fails generically on GNU, so the
    derived read must check the path itself to raise the documented `IsADirectoryError`.
    """
    (tmp_path / 'directory').mkdir()
    (tmp_path / 'file.txt').write_text('content')
    backend = RunOnlyWorkspaceBackend(LocalWorkspaceBackend(tmp_path))
    workspace = Workspace(backend)

    cases: list[tuple[type[OSError], Callable[[], Awaitable[object]]]] = [
        (IsADirectoryError, lambda: workspace.read_bytes('directory')),
        (FileNotFoundError, lambda: workspace.read_bytes('missing')),
        (NotADirectoryError, lambda: workspace.list_dir('file.txt')),
        (FileNotFoundError, lambda: workspace.list_dir('missing')),
    ]
    for error_type, action in cases:
        backend.commands.clear()
        with pytest.raises(error_type) as exc_info:
            await action()
        assert type(exc_info.value) is error_type
        assert len(backend.commands) == 1


async def test_shell_symlink_write_is_atomic_and_preserves_target_mode(tmp_path: Path) -> None:
    target = tmp_path / 'target'
    target.write_bytes(b'original')
    target.chmod(0o640)
    link = tmp_path / 'link'
    link.symlink_to(target)

    class FailedTransfer(RunOnlyWorkspaceBackend):
        fail = True

        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> CommandResult:
            if self.fail and isinstance(command, str) and 'base64 -d' in command and 'mv -f' in command:
                self.fail = False
                command = command.replace('mv -f', 'false && mv -f', 1)
            return await super().run(command, shell=shell, env=env, timeout=timeout)

    backend = FailedTransfer(LocalWorkspaceBackend(tmp_path))
    workspace = Workspace(backend)
    with pytest.raises(WorkspaceError):
        await workspace.write_bytes('link', b'replacement')
    assert target.read_bytes() == b'original'
    assert target.stat().st_mode & 0o777 == 0o640
    assert link.is_symlink()
    assert not list(tmp_path.glob('.pydantic-ai-*'))

    await workspace.write_bytes('link', b'replacement')
    assert target.read_bytes() == b'replacement'
    assert target.stat().st_mode & 0o777 == 0o640
    assert link.is_symlink()

    dangling = tmp_path / 'dangling'
    dangling.symlink_to(tmp_path / 'new-target')
    await workspace.write_bytes('dangling', b'created')
    assert dangling.is_symlink()
    assert (tmp_path / 'new-target').read_bytes() == b'created'


@pytest.mark.parametrize('cleanup_fails', [False, True])
async def test_shell_write_preserves_the_original_error_when_cleanup_fails(tmp_path: Path, cleanup_fails: bool) -> None:
    cleanup_attempted = False

    class FailedWriteBackend(RunOnlyWorkspaceBackend):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> CommandResult:
            nonlocal cleanup_attempted
            if isinstance(command, str) and command.startswith('rm -f '):
                cleanup_attempted = True
                if cleanup_fails:
                    raise RuntimeError('cleanup failed')
            if isinstance(command, str) and 'base64 -d' in command:
                raise RuntimeError('write failed')
            return await super().run(command, shell=shell, env=env, timeout=timeout)

    with pytest.raises(RuntimeError, match='write failed'):
        await Workspace(FailedWriteBackend(LocalWorkspaceBackend(tmp_path))).write_bytes('data.bin', b'data')

    assert cleanup_attempted


def _truncate(stdout: str) -> str:
    """Drop the end of the output, the way a connection lost mid-transfer does."""
    return stdout.strip()[:-4]


def _garble(stdout: str) -> str:
    """A plausible size, then base64 whose padding is broken."""
    return '4\nAAA'


@pytest.mark.parametrize(
    ('operation', 'corrupt'),
    [
        *(
            (operation, corrupt)
            for operation in ('read_bytes', 'list_dir', 'realpath')
            for corrupt in (_truncate, _garble)
        ),
        # A cut-short size is still a number, so only a garbled one is detectable.
        ('stat', _garble),
    ],
)
async def test_shell_filesystem_refuses_damaged_output(
    tmp_path: Path, operation: str, corrupt: Callable[[str], str]
) -> None:
    """Damaged output raises; it never becomes a shorter file, a different listing or another path."""

    class DamagingBackend(RunOnlyWorkspaceBackend):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> CommandResult:
            result = await super().run(command, shell=shell, env=env, timeout=timeout)
            return replace(result, stdout=corrupt(result.stdout))

    (tmp_path / 'directory').mkdir()
    (tmp_path / 'directory' / 'data.bin').write_bytes(bytes(range(256)) * 100)
    workspace = Workspace(DamagingBackend(LocalWorkspaceBackend(tmp_path)))
    with pytest.raises(WorkspaceError, match='damaged or incomplete output'):
        await getattr(workspace, operation)('directory/data.bin' if operation != 'list_dir' else 'directory')


async def test_shell_filesystem_rereads_a_file_replaced_mid_read(tmp_path: Path) -> None:
    """A file swapped for a different size between the size check and the read is read again, not refused."""
    status = anyio.Path(tmp_path / 'status.json')
    await status.write_text('{"exit_code": null}')

    class PublishingBackend(RunOnlyWorkspaceBackend):
        replacements: list[str] = []

        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> CommandResult:
            if isinstance(command, str) and command.startswith('dd ') and self.replacements:
                # The job publishes a new status right after the size was read.
                await status.write_text(self.replacements.pop(0))
            return await super().run(command, shell=shell, env=env, timeout=timeout)

    backend = PublishingBackend(LocalWorkspaceBackend(tmp_path))
    backend.replacements = ['{"exit_code": 0}']
    assert await Workspace(backend).read_text('status.json') == '{"exit_code": 0}'

    # A file that keeps changing size is refused rather than read forever.
    backend.replacements = ['1', '22', '333']
    with pytest.raises(WorkspaceError, match='damaged or incomplete output'):
        await Workspace(backend).read_text('status.json')


async def test_shell_list_dir_does_not_hide_find_failure(tmp_path: Path) -> None:
    class FailedFindBackend(RunOnlyWorkspaceBackend):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> CommandResult:
            if isinstance(command, str) and 'find ' in command:
                command = command.replace('find ', 'false ', 1)
            return await super().run(command, shell=shell, env=env, timeout=timeout)

    with pytest.raises(WorkspaceError):
        await Workspace(FailedFindBackend(LocalWorkspaceBackend(tmp_path))).list_dir('.')


async def test_unavailable_workspace_uses_the_configured_reason_for_every_operation() -> None:
    reason = 'workspace disabled by policy'
    backend = Workspace(UnavailableWorkspace(reason))
    # No environment exists, so there is no identity a later run could reconnect to.
    assert backend.ref is None
    operations = [
        backend.run(['true']),
        backend.working_dir(),
        backend.read_bytes('/file'),
        backend.write_bytes('/file', b'data'),
        backend.stat('/file'),
        backend.list_dir('/'),
        backend.make_dir('/dir'),
        backend.remove('/file'),
        backend.exists('/file'),
    ]

    # A workspace error, not a `UserError`, so a tool can catch it and tell the model instead of ending the run.
    for operation in operations:
        with pytest.raises(WorkspaceUnavailableError, match='workspace disabled by policy'):
            await operation


async def test_attached_is_false_only_for_an_unavailable_workspace_even_through_wrappers() -> None:
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())

    assert not ctx.workspace.attached
    assert not ReadOnlyWorkspace(Workspace(UnavailableWorkspace('disabled by policy'))).attached
    assert not Workspace(Workspace(UnavailableWorkspace('disabled by policy'))).attached
    assert ReadOnlyWorkspace(Workspace(FakeWorkspace('attached'))).attached


def test_a_plain_workspace_around_a_wrapper_keeps_its_policy() -> None:
    nested = Workspace(ReadOnlyWorkspace(Workspace(FakeWorkspace('nested'))))

    assert nested.read_only
    assert workspace_layers(nested) == [ReadOnlyWorkspace, FakeWorkspace]


async def test_bare_run_context_workspace_explains_how_to_attach_one() -> None:
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())

    with pytest.raises(
        WorkspaceUnavailableError, match=r"LocalWorkspace\('\.'\).*https://pydantic\.dev/docs/ai/workspace/"
    ):
        await ctx.workspace.run(['true'])


async def test_explicit_backend_wins_over_a_capability_backend() -> None:
    capability = WorkspaceCapability()
    explicit = FakeWorkspace('explicit')
    observed: list[Workspace] = []
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.workspace)
        return 'ok'

    await agent.run('go', workspace=explicit)

    assert observed[0].backend is explicit
    assert capability.refs == []


async def test_for_run_sees_the_workspace_the_run_uses(tmp_path: Path) -> None:
    (tmp_path / 'catalog.txt').write_text('from the workspace')
    seen: list[str] = []

    class ReadsInForRun(AbstractCapability[Any]):
        async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
            seen.append(await ctx.workspace.read_text('catalog.txt'))
            return self

    result = await Agent(TestModel(), capabilities=[ReadsInForRun(), LocalWorkspace(tmp_path)]).run('go')

    assert seen == ['from the workspace']
    assert result.workspace.ref == WorkspaceRef(provider='local', id=str(tmp_path))


async def test_a_run_workspace_capability_overrides_the_agents(tmp_path: Path) -> None:
    agent = Agent(TestModel(), capabilities=[LocalWorkspace(tmp_path / 'agent')])

    result = await agent.run('go', capabilities=[LocalWorkspace(tmp_path)])

    assert result.workspace.ref == WorkspaceRef(provider='local', id=str(tmp_path))


async def test_a_capability_function_supplies_the_workspace_after_for_run(tmp_path: Path) -> None:
    agent = Agent(TestModel(), deps_type=Path, capabilities=[lambda ctx: LocalWorkspace(ctx.deps)])

    result = await agent.run('go', deps=tmp_path)

    assert result.workspace.ref == WorkspaceRef(provider='local', id=str(tmp_path))


async def test_for_run_cannot_change_the_workspace_selected_before_it(tmp_path: Path) -> None:
    class ReadOnlyPerRun(LocalWorkspace[Any]):
        async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
            return LocalWorkspace(self.working_dir, read_only=True)

    with pytest.raises(UserError, match="A capability's `for_run` changed the workspace"):
        await Agent(TestModel(), capabilities=[ReadOnlyPerRun(tmp_path)]).run('go')


async def test_failed_run_error_hook_exposes_workspace_ref_for_cleanup() -> None:
    backend = FakeWorkspace('failed-run')
    seen: list[WorkspaceRef | None] = []

    class Cleanup(AbstractCapability[Any]):
        async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
            await ctx.workspace.working_dir()
            return await handler()

        async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            seen.append(ctx.workspace.ref)
            raise error

    def fail_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ValueError('failure')

    agent = Agent(FunctionModel(fail_model), capabilities=[WorkspaceCapability(backend), Cleanup()])
    with pytest.raises(ValueError, match='failure'):
        await agent.run('go')
    assert seen == [backend.ref]


async def test_the_result_carries_the_workspace_the_run_used() -> None:
    """`result.workspace` is the same object tools saw, so a caller can keep working in it."""
    capability = WorkspaceCapability()
    observed: list[Workspace] = []
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.workspace)
        return (await ctx.workspace.run(['true'])).stdout

    result = await agent.run('go')

    assert result.workspace is observed[0]
    assert result.workspace.ref == WorkspaceRef(provider='fake', id='fake-capability')

    # Handing it to a second run continues in the same environment rather than making a new one.
    second = await agent.run('again', workspace=result.workspace)
    assert second.workspace is result.workspace
    assert capability.refs == [None]
    assert capability.backend.create_calls == 1


async def test_workspace_ref_is_persisted_and_reused_from_agent_history() -> None:
    seen_refs: list[WorkspaceRef | None] = []
    backends: list[FakeWorkspace] = []

    class HistoryCapability(AbstractCapability[Any]):
        id = 'history-workspace'

        def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
            seen_refs.append(ref)
            backend = FakeWorkspace('history', ref=ref)
            backends.append(backend)
            return backend

    class RepeatingTestModel(TestModel):
        def _request(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            if isinstance(messages[-1], ModelRequest) and any(
                isinstance(part, UserPromptPart) for part in messages[-1].parts
            ):
                messages = []
            return super()._request(messages, model_settings, model_request_parameters)

    agent = Agent(RepeatingTestModel(call_tools=['probe']), deps_type=type(None), capabilities=[HistoryCapability()])

    @agent.tool
    async def probe(ctx: RunContext[None]) -> str:
        return (await ctx.workspace.run(['true'])).stdout

    first = await agent.run('first')
    history = ModelMessagesTypeAdapter.validate_json(first.all_messages_json())
    second = await agent.run('second', message_history=history)

    ref = WorkspaceRef(provider='fake', id='fake-history')
    assert seen_refs == [None, ref]
    assert first.response.workspace_ref == ref
    assert second.response.workspace_ref == ref
    assert second.workspace.ref == ref
    assert backends[0].create_calls == 1
    assert backends[1].attach_calls == 1
    assert backends[1].create_calls == 0


async def test_explicit_workspace_facade_wins_over_historical_ref_without_mutating_history() -> None:
    backend = FakeWorkspace('explicit-history')
    explicit = ReadOnlyWorkspace(Workspace(backend))
    historical = ModelResponse(
        parts=[TextPart('old')], metadata={'keep': True}, workspace_ref=WorkspaceRef(provider='fake', id='old')
    )
    agent = Agent(TestModel(custom_output_text='done'), deps_type=type(None))

    result = await agent.run('new', message_history=[historical], workspace=explicit)

    assert result.workspace is explicit
    assert historical.metadata == {'keep': True}
    assert historical.workspace_ref == WorkspaceRef(provider='fake', id='old')
    with pytest.raises(WorkspaceReadOnlyError, match='read-only'):
        await result.workspace.run(['true'])


async def test_latest_none_workspace_ref_suppresses_an_older_historical_ref() -> None:
    """A backend that has no environment yet stamps `None`, which is newer than the older ref."""
    seen: list[WorkspaceRef | None] = []

    class HistoryCapability(AbstractCapability[Any]):
        def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
            seen.append(ref)
            return FakeWorkspace('latest-none', ref=ref)

    agent = Agent(TestModel(custom_output_text='done'), deps_type=type(None), capabilities=[HistoryCapability()])

    older = ModelResponse(parts=[TextPart('old')], workspace_ref=WorkspaceRef(provider='fake', id='old'))
    first = await agent.run('first', message_history=[older], workspace=FakeWorkspace('not-created-yet'))
    await agent.run('new', message_history=first.all_messages())

    assert seen == [None]


@pytest.mark.parametrize('middle', ['no_workspace_agent', 'unavailable_workspace'])
async def test_a_turn_without_a_workspace_keeps_the_conversations_ref(middle: str) -> None:
    capability = ProviderWorkspaceCapability('provider')
    agent = Agent(TestModel(), capabilities=[capability])
    ref = WorkspaceRef(provider='provider', id='existing')
    history: list[ModelMessage] = [ModelResponse(parts=[TextPart('old')], workspace_ref=ref)]

    if middle == 'no_workspace_agent':
        middle_turn = await Agent(TestModel()).run('chat', message_history=history)
    else:
        middle_turn = await agent.run('chat', message_history=history, workspace=UnavailableWorkspace('off'))
    assert middle_turn.response.workspace_ref == ref

    third = await agent.run('go', message_history=middle_turn.all_messages())
    assert third.workspace.ref == ref


async def test_historical_workspace_ref_without_capability_stays_unavailable() -> None:
    historical = ModelResponse(
        parts=[ToolCallPart('probe', {})], workspace_ref=WorkspaceRef(provider='missing', id='remote')
    )
    agent = Agent(TestModel(call_tools=['probe']), deps_type=type(None))

    @agent.tool
    async def probe(ctx: RunContext[None]) -> str:
        with pytest.raises(WorkspaceUnavailableError, match='No workspace is attached'):
            await ctx.workspace.run(['true'])
        return 'unavailable'

    result = await agent.run(None, message_history=[historical])
    assert result.output == '{"probe":"unavailable"}'


async def test_a_result_still_round_trips_through_json_when_a_workspace_was_used() -> None:
    """The workspace is a live handle, so it is left out of the serialized result rather than breaking it."""
    agent = Agent(_tool_call_model(), capabilities=[WorkspaceCapability()])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        return (await ctx.workspace.run(['true'])).stdout

    result = await agent.run('go')
    adapter = TypeAdapter(AgentRunResult[str])
    restored = adapter.validate_json(adapter.dump_json(result))

    assert restored == result
    with pytest.raises(WorkspaceUnavailableError, match='No workspace is attached'):
        await restored.workspace.run(['true'])


async def test_a_result_built_outside_a_run_explains_that_no_workspace_is_attached() -> None:
    result = AgentRunResult[str]('output')

    with pytest.raises(WorkspaceUnavailableError, match='No workspace is attached') as exc_info:
        await result.workspace.run(['true'])
    # A bare `WorkspaceRef` is refused without a capability to resolve it, so the hint names one.
    assert 'with a capability that can reconnect to an existing environment, pass its `WorkspaceRef`' in str(
        exc_info.value
    )


class ProviderWorkspaceCapability(AbstractCapability[Any]):
    """Creates a fresh backend without a ref, and claims only refs from its own provider."""

    def __init__(self, provider: str) -> None:
        self.id = self.provider = provider
        self.refs: list[WorkspaceRef | None] = []
        self.supplied: list[WorkspaceBackend] = []

    def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        self.refs.append(ref)
        if ref is not None and ref.provider != self.provider:
            return None
        backend = FakeWorkspace('fresh', ref=ref)
        self.supplied.append(backend)
        return backend


async def test_the_first_workspace_capability_that_answers_wins() -> None:
    first, second = ProviderWorkspaceCapability('first'), ProviderWorkspaceCapability('second')
    agent = Agent(TestModel(), capabilities=[first, second])

    fresh = await agent.run('go')
    assert [fresh.workspace.backend] == first.supplied
    assert second.refs == []

    # A ref the first capability declines falls through to the one that owns it.
    ref = WorkspaceRef(provider='second', id='existing')
    historical = ModelResponse(parts=[TextPart('old')], workspace_ref=ref)
    continued = await agent.run('go', message_history=[historical])
    assert [continued.workspace.backend] == second.supplied
    assert first.refs == [None, ref]


async def test_a_run_workspace_capability_joins_the_agents() -> None:
    agent = Agent(TestModel(), capabilities=[ProviderWorkspaceCapability('first')])
    second = ProviderWorkspaceCapability('second')

    result = await agent.run('go', capabilities=[second], workspace=WorkspaceRef(provider='second', id='existing'))

    assert [result.workspace.backend] == second.supplied


async def test_a_run_workspace_capability_is_asked_before_the_agents(tmp_path: Path) -> None:
    agent = Agent(TestModel(), capabilities=[LocalWorkspace(tmp_path)])
    run_level = ProviderWorkspaceCapability('run')

    result = await agent.run('go', capabilities=[run_level])

    assert [result.workspace.backend] == run_level.supplied


async def test_a_run_workspace_capability_function_is_asked_before_the_agents_after_for_run(tmp_path: Path) -> None:
    agent = Agent(TestModel(), capabilities=[lambda ctx: LocalWorkspace(tmp_path)])
    run_level = ProviderWorkspaceCapability('run')

    result = await agent.run('go', capabilities=[lambda ctx: run_level])

    assert [result.workspace.backend] == run_level.supplied


async def test_an_unrecognized_history_ref_is_an_error_when_the_agent_has_workspace_capabilities() -> None:
    """Silently dropping the ref would put the conversation in a different environment than it continued from."""
    historical = ModelResponse(parts=[TextPart('old')], workspace_ref=WorkspaceRef(provider='gone', id='sb-1'))
    agent = Agent(TestModel(), capabilities=[ProviderWorkspaceCapability('first')])

    with pytest.raises(
        UserError,
        match=(
            r"The message history continues in workspace `gone:sb-1`, but none of the agent's workspace "
            r"capabilities recognized it\. Pass `workspace='new'` to start a fresh workspace, or pass the "
            r'workspace to continue in with `workspace=`\.'
        ),
    ):
        await agent.run('go', message_history=[historical])

    fresh = await agent.run('go', message_history=[historical], workspace='new')
    assert fresh.workspace.attached


@pytest.mark.parametrize('source', ['explicit', 'history'])
async def test_a_resolver_cannot_answer_a_ref_with_a_backend_that_would_create_a_fresh_one(source: str) -> None:
    class LazyResolver(AbstractCapability[Any]):
        def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
            return FakeWorkspace('lazy')

    ref = WorkspaceRef(provider='fake', id='existing')
    agent = Agent(TestModel(), capabilities=[LazyResolver()])
    with pytest.raises(
        UserError,
        match=r"`get_workspace` returned a different workspace than requested: asked for WorkspaceRef\(.*'existing'.*\), got ",
    ):
        if source == 'explicit':
            await agent.run('go', workspace=ref)
        else:
            await agent.run('go', message_history=[ModelResponse(parts=[TextPart('old')], workspace_ref=ref)])


async def test_a_gone_workspace_fails_on_first_use_and_new_recovers() -> None:
    def probe_each_turn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(part, UserPromptPart) for part in messages[-1].parts):
            return ModelResponse(parts=[ToolCallPart('probe', {})])
        return ModelResponse(parts=[TextPart('done')])

    provider = InMemoryProvider()
    agent = Agent(FunctionModel(probe_each_turn), capabilities=[provider.capability()])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        return await ctx.workspace.working_dir()

    first = await agent.run('go')
    provider.environments.clear()  # the environment is destroyed between turns

    with pytest.raises(WorkspaceUnavailableError, match="environment 'env-1' does not exist"):
        await agent.run('go', message_history=first.all_messages())

    fresh = await agent.run('go', message_history=first.all_messages(), workspace='new')
    assert fresh.workspace.ref == WorkspaceRef(provider='fake', id='env-1')
    assert provider.log == ['create:env-1', 'create:env-1']


async def test_a_history_ref_is_offered_to_a_workspace_capability_that_exists_only_after_for_run(
    tmp_path: Path,
) -> None:
    ref = WorkspaceRef(provider='local', id=str(tmp_path))
    historical = ModelResponse(parts=[TextPart('old')], workspace_ref=ref)
    agent = Agent(TestModel(), deps_type=Path, capabilities=[lambda ctx: LocalWorkspace(ctx.deps)])

    continued = await agent.run('go', deps=tmp_path, message_history=[historical])
    assert continued.workspace.ref == ref

    with pytest.raises(UserError, match='The message history continues in workspace `local:'):
        await agent.run('go', deps=tmp_path / 'other', message_history=[historical])


async def test_an_explicit_ref_without_a_workspace_capability_is_an_error() -> None:
    agent = Agent(TestModel())

    with pytest.raises(
        UserError,
        match='Workspace `fake:remote` was passed to the run, but the agent has no workspace capability to resolve it',
    ):
        await agent.run('go', workspace=WorkspaceRef(provider='fake', id='remote'))


def test_has_get_workspace_mirrors_the_capability_tree() -> None:
    supplier = WorkspaceCapability()

    class Policy(WrapperCapability[Any]):
        def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
            return None  # pragma: no cover

    class Bundle(CombinedCapability[Any]):
        def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
            return None  # pragma: no cover

    assert supplier._has_get_workspace  # pyright: ignore[reportPrivateUsage]
    assert not AbstractCapability[Any]()._has_get_workspace  # pyright: ignore[reportPrivateUsage]
    assert CombinedCapability([AbstractCapability[Any](), supplier])._has_get_workspace  # pyright: ignore[reportPrivateUsage]
    assert WrapperCapability(supplier)._has_get_workspace  # pyright: ignore[reportPrivateUsage]
    assert not WrapperCapability(AbstractCapability[Any]())._has_get_workspace  # pyright: ignore[reportPrivateUsage]
    assert Policy(AbstractCapability[Any]())._has_get_workspace  # pyright: ignore[reportPrivateUsage]
    assert not CombinedCapability([AbstractCapability[Any]()])._has_get_workspace  # pyright: ignore[reportPrivateUsage]
    assert Bundle([AbstractCapability[Any]()])._has_get_workspace  # pyright: ignore[reportPrivateUsage]


async def test_new_workspace_ignores_the_ref_in_history() -> None:
    """`workspace='new'` mirrors `conversation_id='new'`: history's ref is not offered, so a fresh one is created."""
    capability = ProviderWorkspaceCapability('provider')
    agent = Agent(TestModel(), capabilities=[capability])
    historical = ModelResponse(parts=[TextPart('old')], workspace_ref=WorkspaceRef(provider='provider', id='existing'))

    result = await agent.run('go', message_history=[historical], workspace='new')

    assert capability.refs == [None]
    # Nothing used the fresh workspace, so the old ref is not carried forward either.
    assert result.response.workspace_ref is None


async def test_new_workspace_without_a_supplier_raises() -> None:
    """Unlike `None`, `'new'` is an explicit request, so an agent that cannot create a workspace says so."""
    agent = Agent(TestModel(), capabilities=[DecliningWorkspaceCapability()])

    with pytest.raises(UserError, match="`workspace='new'` needs a capability that can create a workspace"):
        await agent.run('go', workspace='new')


async def test_a_workspace_capability_cannot_be_deferred(tmp_path: Path) -> None:
    message = "supplies the run's workspace, which is chosen when the run starts, so it can't be deferred"
    with pytest.raises(UserError, match=f'`WrapperCapability` {message}'):
        Agent(TestModel(), capabilities=[WrapperCapability(LocalWorkspace(tmp_path), defer_loading=True, id='ws')])

    with pytest.raises(UserError, match=f'`LocalWorkspace` {message}'):
        Agent(TestModel(), capabilities=[LocalWorkspace(tmp_path, defer_loading=True)])


async def test_wrapper_composes_workspace_policy_over_combined_capability() -> None:
    provider = WorkspaceCapability()

    class Policy(WrapperCapability[Any]):
        def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
            backend = super().get_workspace(ctx, ref=ref)
            return ReadOnlyWorkspace(Workspace(backend)) if backend is not None else None

    capability = Policy(CombinedCapability([provider]))
    agent = Agent(TestModel(call_tools=['probe']), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> None:
        await ctx.workspace.write_text('blocked.txt', 'nope')

    with pytest.raises(WorkspaceReadOnlyError, match='read-only'):
        await agent.run('go')
    assert provider.refs == [None]


class _Tagged(WrapperWorkspace):
    def __init__(self, wrapped: Workspace, tag: str) -> None:
        super().__init__(wrapped)
        self.tag = tag


class _WrapHook(AbstractCapability[Any]):
    """Records what the private wrap hook sees and tags the workspace it returns."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.seen: list[tuple[type[Workspace], bool, bool]] = []

    def _prepare_workspace(self, ctx: RunContext[Any], workspace: Workspace, *, explicit: bool) -> Workspace:
        self.seen.append((type(workspace), explicit, ctx.root_capability is not None))
        return _Tagged(workspace, self.tag)


async def test_prepare_workspace_hook_wraps_innermost_last_and_reports_explicit_workspaces() -> None:
    """The private hook composes like middleware and sees the composed per-run tree on the context.

    Not reachable through a public API: nothing first-party overrides it besides the durability
    capabilities, whose engine suites cover the observable behavior. This pins the plumbing they
    build on: the last capability wraps first (innermost), a wrapper capability forwards, the
    result is what tools and `result.workspace` see, and `explicit` tells a caller-supplied
    workspace from a capability-selected one.
    """
    first, last = _WrapHook('first'), _WrapHook('last')
    provider = WorkspaceCapability()
    agent = Agent(TestModel(), capabilities=[first, WrapperCapability(last), provider])

    result = await agent.run('go')

    outer = result.workspace
    assert isinstance(outer, _Tagged) and outer.tag == 'first'
    inner = outer.wrapped
    assert isinstance(inner, _Tagged) and inner.tag == 'last'
    assert inner.wrapped.backend is provider.backend
    assert first.seen == [(_Tagged, False, True)]
    assert last.seen == [(Workspace, False, True)]

    explicit = Workspace(FakeWorkspace('explicit'))
    second = await agent.run('again', workspace=explicit)
    second_inner = second.workspace
    assert isinstance(second_inner, _Tagged)
    second_inner = second_inner.wrapped
    assert isinstance(second_inner, _Tagged) and second_inner.wrapped is explicit
    assert last.seen[-1] == (Workspace, True, True)


async def test_declining_capability_leaves_the_run_workspace_unavailable() -> None:
    capability = DecliningWorkspaceCapability()
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        assert not ctx.workspace.attached
        await ctx.workspace.run(['true'])
        return 'unreachable'  # pragma: no cover

    with pytest.raises(WorkspaceUnavailableError, match='No workspace is attached'):
        await agent.run('go')
    assert capability.calls == 1


async def test_capability_backend_is_available_without_connecting_during_run_setup() -> None:
    capability = WorkspaceCapability()
    seen: list[Workspace] = []
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        seen.append(ctx.workspace)
        return 'ok'

    await agent.run('go')

    assert seen[0].backend is capability.backend
    assert capability.backend.create_calls == 0


async def test_failed_run_stamps_the_workspace_ref() -> None:
    backend = FakeWorkspace('failed')
    agent = Agent(_tool_call_model('explode'))

    @agent.tool
    async def explode(ctx: RunContext[Any]) -> str:
        await ctx.workspace.run(['true'])
        raise RuntimeError('boom')

    with capture_run_messages() as messages, pytest.raises(RuntimeError, match='boom'):
        await agent.run('go', workspace=backend)

    response = next(message for message in reversed(messages) if isinstance(message, ModelResponse))
    assert response.workspace_ref == backend.ref
    assert backend.ref is not None


async def test_cancelled_run_stamps_the_workspace_ref() -> None:
    backend = FakeWorkspace('cancelled')
    agent = Agent(_tool_call_model())
    entered = anyio.Event()
    captured_messages: list[ModelMessage] = []

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        await ctx.workspace.run(['true'])
        entered.set()
        await anyio.sleep_forever()
        return 'unreachable'  # pragma: no cover

    async with anyio.create_task_group() as tg:

        async def run_agent() -> None:
            with capture_run_messages() as captured:
                try:
                    await agent.run('go', workspace=backend)
                finally:
                    captured_messages.extend(captured)

        tg.start_soon(run_agent)
        await entered.wait()
        tg.cancel_scope.cancel()

    response = next(message for message in reversed(captured_messages) if isinstance(message, ModelResponse))
    assert response.workspace_ref == backend.ref
    assert backend.ref is not None


async def test_streamed_responses_keep_the_workspace_ref() -> None:
    ref = WorkspaceRef(provider='fake', id='streamed')
    backend = FakeWorkspace('streamed', ref=ref)
    agent = Agent(TestModel(custom_output_text='streamed'))

    async with agent.run_stream('go', workspace=Workspace(backend)) as result:
        responses = [response async for response in result.stream_response(debounce_by=None)]

    assert responses
    assert all(response.workspace_ref == ref for response in responses)


async def test_streamed_result_keeps_the_workspace_identity() -> None:
    backend = FakeWorkspace('streamed-result')
    workspace = ReadOnlyWorkspace(Workspace(backend))
    agent = Agent(TestModel(custom_output_text='streamed'))

    async with agent.run_stream('go', workspace=workspace) as result:
        await result.get_output()
        assert result.workspace is workspace


def test_sync_streamed_result_keeps_the_workspace_identity() -> None:
    workspace = ReadOnlyWorkspace(Workspace(FakeWorkspace('sync-streamed-result')))
    agent = Agent(TestModel(custom_output_text='streamed'))

    with agent.run_stream_sync('go', workspace=workspace) as result:
        result.get_output()
        assert result.workspace is workspace


async def test_result_workspace_survives_after_run_replacement() -> None:
    workspace = ReadOnlyWorkspace(Workspace(FakeWorkspace('after-run')))

    class ReplaceResult(AbstractCapability[Any]):
        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            return replace(result, output='replaced')

    agent = Agent(TestModel(custom_output_text='original'), capabilities=[ReplaceResult()])
    result = await agent.run('go', workspace=workspace)

    assert result.output == 'replaced'
    assert result.workspace is workspace


async def test_result_workspace_is_the_outer_runs_when_after_run_returns_another_runs_result() -> None:
    inner_workspace = Workspace(FakeWorkspace('inner'))
    inner = Agent(TestModel(custom_output_text='inner'))

    class ReturnInnerResult(AbstractCapability[Any]):
        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            return await inner.run('go', workspace=inner_workspace)

    outer_workspace = Workspace(FakeWorkspace('outer'))
    agent = Agent(TestModel(custom_output_text='outer'), capabilities=[ReturnInnerResult()])
    result = await agent.run('go', workspace=outer_workspace)

    assert result.output == 'inner'
    assert result.workspace is outer_workspace


@pytest.mark.parametrize('fail', [False, True], ids=['success', 'error'])
async def test_late_after_run_workspace_ref_is_stamped_on_latest_response(fail: bool) -> None:
    backend = FakeWorkspace('late-after-run')

    class LateUse(AbstractCapability[Any]):
        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            assert ctx.workspace.ref is None
            await ctx.workspace.run(['after-run'])
            if fail:
                raise RuntimeError('late failure')
            return result

    agent = Agent(TestModel(custom_output_text='done'), capabilities=[LateUse()])
    with capture_run_messages() as captured:
        if fail:
            with pytest.raises(RuntimeError, match='late failure'):
                await agent.run('go', workspace=backend)
        else:
            await agent.run('go', workspace=backend)
    response = next(message for message in reversed(captured) if isinstance(message, ModelResponse))

    assert response.workspace_ref == backend.ref


async def test_no_prompt_response_clone_gets_late_workspace_ref() -> None:
    old_response = ModelResponse(parts=[TextPart('old')], workspace_ref=WorkspaceRef(provider='fake', id='old'))
    backend = FakeWorkspace('late-no-prompt')

    class LateUse(AbstractCapability[Any]):
        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            assert ctx.workspace.ref is None
            await ctx.workspace.run(['after-run'])
            return result

    result = await Agent(TestModel(custom_output_text='done'), capabilities=[LateUse()]).run(
        message_history=[old_response], workspace=backend
    )
    response = next(message for message in reversed(result.all_messages()) if isinstance(message, ModelResponse))

    assert response is not old_response
    assert response.workspace_ref == backend.ref
    assert old_response.workspace_ref == WorkspaceRef(provider='fake', id='old')


async def test_borrowed_short_circuit_response_keeps_its_original_workspace_ref() -> None:
    old_ref = WorkspaceRef(provider='fake', id='old')
    old_response = ModelResponse(parts=[TextPart('old')], workspace_ref=old_ref)
    backend = FakeWorkspace('borrowed-short-circuit')

    class ShortCircuit(AbstractCapability[Any]):
        async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[str]:
            await ctx.workspace.run(['cached'])
            return AgentRunResult('cached')

    agent = Agent(TestModel(), capabilities=[ShortCircuit()])
    result = await agent.run('new', message_history=[old_response], workspace=backend)

    assert result.output == 'cached'
    assert old_response.workspace_ref == old_ref


async def test_copied_history_response_keeps_its_workspace_ref_when_the_model_fails() -> None:
    old_ref = WorkspaceRef(provider='fake', id='old')
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('old')]),
        ModelResponse(parts=[TextPart('old')], workspace_ref=old_ref),
    ]
    backend = FakeWorkspace('copied-history')

    class CopyHistory(AbstractCapability[Any]):
        async def before_model_request(self, ctx: RunContext[Any], request_context: Any) -> Any:
            await ctx.workspace.run(['used'])
            request_context.messages = [replace(message) for message in request_context.messages]
            return request_context

    def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('model failed')

    agent = Agent(FunctionModel(fail), capabilities=[CopyHistory()])
    with capture_run_messages() as captured:
        with pytest.raises(RuntimeError, match='model failed'):
            await agent.run('new', message_history=history, workspace=backend)
    copied = next(message for message in reversed(captured) if isinstance(message, ModelResponse))

    assert copied.parts == [TextPart('old')]
    assert copied.workspace_ref == old_ref


async def test_streamed_short_circuit_result_keeps_the_workspace_identity() -> None:
    workspace = ReadOnlyWorkspace(Workspace(FakeWorkspace('short-circuit')))

    class ShortCircuit(AbstractCapability[Any]):
        async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[str]:
            return AgentRunResult('short-circuited')

    agent = Agent(TestModel(), capabilities=[ShortCircuit()])

    async with agent.run_stream('go', workspace=workspace) as result:
        assert await result.get_output() == 'short-circuited'
        assert result.workspace is workspace


async def test_interrupted_stream_history_keeps_the_workspace_ref() -> None:
    ref = WorkspaceRef(provider='fake', id='interrupted')
    backend = FakeWorkspace('interrupted', ref=ref)
    agent = Agent(TestModel(custom_output_text='hello world'))

    async with agent.run_stream('go', workspace=backend) as result:
        await anext(result.stream_response(debounce_by=None))
        await result.cancel()

    assert result.response.state == 'interrupted'
    assert result.response.workspace_ref == ref
    response = next(message for message in reversed(result.all_messages()) if isinstance(message, ModelResponse))
    assert response.workspace_ref == ref


async def test_no_prompt_history_response_is_copied_before_stamping_workspace_ref() -> None:
    original = ModelResponse(parts=[TextPart('finished')])
    backend = FakeWorkspace('history-copy', ref=WorkspaceRef(provider='fake', id='history-copy'))
    agent = Agent(TestModel(custom_output_text='unused'))

    result = await agent.run(message_history=[original], workspace=Workspace(backend))

    assert result.output == 'finished'
    assert original.workspace_ref is None
    response = next(message for message in reversed(result.all_messages()) if isinstance(message, ModelResponse))
    assert response is not original
    assert response.workspace_ref == backend.ref


async def test_no_prompt_unavailable_history_ref_is_copied() -> None:
    ref = WorkspaceRef(provider='missing', id='remote')
    original = ModelResponse(parts=[TextPart('finished')], workspace_ref=ref)
    result = await Agent(TestModel()).run(message_history=[original])

    assert result.response is not original
    assert original.workspace_ref == ref
    assert result.response.workspace_ref == ref


async def test_no_prompt_pending_tool_call_history_is_copied_before_execution() -> None:
    original = ModelResponse(parts=[ToolCallPart('probe', {})])
    history = [ModelRequest(parts=[UserPromptPart('go')]), original]
    backend = FakeWorkspace('pending')
    agent = Agent(TestModel(custom_output_text='done'))

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        return (await ctx.workspace.run(['true'])).stdout

    result = await agent.run(message_history=history, workspace=Workspace(backend))

    assert result.output == 'done'
    assert original.workspace_ref is None
    copied = next(message for message in result.all_messages() if isinstance(message, ModelResponse))
    assert copied is not original
    assert copied.workspace_ref == backend.ref


async def test_tool_result_event_sees_workspace_ref_after_lazy_acquisition() -> None:
    backend = FakeWorkspace('event')
    agent = Agent(TestModel(call_tools=['probe']))
    observed: list[WorkspaceRef | None] = []

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        await ctx.workspace.run(['true'])
        return 'ok'

    @agent.on_event(FunctionToolResultEvent)
    async def observe(ctx: RunContext[Any], event: FunctionToolResultEvent) -> None:
        response = next(message for message in reversed(ctx.messages) if isinstance(message, ModelResponse))
        observed.append(response.workspace_ref)

    await agent.run('go', workspace=backend)

    assert observed == [backend.ref]
    assert backend.ref is not None


async def test_deferred_approval_stamps_copied_response_after_workspace_acquisition() -> None:
    original_response: ModelResponse | None = None

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal original_response
        if len(messages) == 1:
            original_response = ModelResponse(parts=[ToolCallPart('probe', {})])
            return original_response
        return ModelResponse(parts=[TextPart('approved')])

    agent = Agent(FunctionModel(model), deps_type=type(None), output_type=[str, DeferredToolRequests])

    @agent.tool
    async def probe(ctx: RunContext[None]) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return (await ctx.workspace.run(['true'])).stdout

    first = await agent.run('go')
    assert isinstance(first.output, DeferredToolRequests)
    assert original_response is not None
    assert original_response.workspace_ref is None

    backend = FakeWorkspace('approved')
    source_tool_response = next(message for message in first.all_messages() if isinstance(message, ModelResponse))
    second = await agent.run(
        message_history=first.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={first.output.approvals[0].tool_call_id: ToolApproved()}),
        workspace=Workspace(backend),
    )

    assert second.output == 'approved'
    assert original_response.workspace_ref is None
    copied_tool_response = next(message for message in second.all_messages() if isinstance(message, ModelResponse))
    assert copied_tool_response is not source_tool_response
    assert copied_tool_response.workspace_ref == backend.ref


async def test_capability_can_supply_a_backend_for_an_explicit_ref() -> None:
    capability = ConnectOnlyWorkspaceCapability()
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        return (await ctx.workspace.run(['true'])).stdout

    result: AgentRunResult[Any] = await agent.run('go', workspace=WorkspaceRef(provider='fake', id='existing'))

    assert result.output == 'done'
    assert capability.ids == ['existing']
    assert result.workspace.ref == WorkspaceRef(provider='fake', id='existing')
    assert await result.workspace.working_dir() == '/workspace'

    # This capability only attaches: with no ref it declines and the run gets the unavailable default.
    without_ref: AgentRunResult[Any] = await Agent(TestModel(), capabilities=[capability]).run('go')

    assert not without_ref.workspace.attached
    assert capability.ids == ['existing']


async def test_shell_remove_refuses_canonical_root_via_alias(tmp_path: Path) -> None:
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'safe').write_bytes(b'safe')
    alias_parent = tmp_path / 'alias_parent'
    alias_parent.symlink_to(tmp_path, target_is_directory=True)
    workspace = Workspace(RunOnlyWorkspaceBackend(LocalWorkspaceBackend(root)))
    with pytest.raises(ValueError, match='workspace root'):
        await workspace.remove(str(alias_parent / 'root'))
    assert (root / 'safe').read_bytes() == b'safe'


async def test_shell_remove_symlink_to_root_only_removes_link(tmp_path: Path) -> None:
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'safe').write_bytes(b'safe')
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    workspace = Workspace(RunOnlyWorkspaceBackend(LocalWorkspaceBackend(root)))
    await workspace.remove(str(alias))
    assert not alias.is_symlink()


async def test_shell_read_names_a_missing_utility(tmp_path: Path) -> None:
    """Without `dd`, reads fail naming it rather than as damaged output."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for tool in ('base64', 'cp', 'find', 'mkdir', 'mv', 'readlink', 'rm', 'wc', 'cat', 'stat', 'head', 'tr', 'sed'):
        found = shutil.which(tool)
        assert found is not None
        (bin_dir / tool).symlink_to(found)
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'a.txt').write_text('hello')
    workspace = Workspace(RunOnlyWorkspaceBackend(LocalWorkspaceBackend(root, env={'PATH': str(bin_dir)})))
    with pytest.raises(WorkspaceError, match=r'^The workspace has no `dd`, which its file operations need'):
        await workspace.read_text('a.txt')
