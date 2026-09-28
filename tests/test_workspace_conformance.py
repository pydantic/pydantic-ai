"""The public workspace conformance suite, run once per kind of backend core supports."""

from __future__ import annotations

import os
import posixpath
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path

import anyio.to_thread
import pytest

from pydantic_ai.workspaces import (
    BubblewrapWorkspace,
    LocalWorkspaceBackend,
    SSHWorkspaceBackend,
    Workspace,
    WorkspaceBackend,
    WorkspaceRef,
)
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite

from .fake_remote_tools import install_fake_remote_tools
from .workspace_fakes import (
    FakeWorkspace,
    FilesystemOnlyWorkspaceBackend,
    InMemoryProvider,
    RunOnlyWorkspaceBackend,
)

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='workspace conformance command rules use POSIX sh')


def _local_destroy_environment() -> Callable[[WorkspaceBackend], Awaitable[None]]:
    async def destroy(backend: WorkspaceBackend) -> None:
        assert backend.ref is not None
        await anyio.to_thread.run_sync(shutil.rmtree, backend.ref.id)

    return destroy


class TestLocalWorkspaceBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, tmp_path: Path) -> LocalWorkspaceBackend:
        return LocalWorkspaceBackend(tmp_path)

    @pytest.fixture
    def destructive_backend(self, tmp_path_factory: pytest.TempPathFactory) -> Callable[[], WorkspaceBackend]:
        path = tmp_path_factory.mktemp('fresh-ws')
        return lambda: LocalWorkspaceBackend(path)

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: LocalWorkspaceBackend(ref.id)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _local_destroy_environment()


class TestSSHWorkspaceBackend(WorkspaceBackendSuite):
    """Runs against a fake `ssh` that executes the remote command on this machine."""

    @pytest.fixture(autouse=True)
    def fake_ssh(self, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_remote_tools(tmp_path_factory.mktemp('fake-remote'), monkeypatch)

    @pytest.mark.skip(reason='the same shell paging as `TestRunOnlyWorkspaceBackend`, at a connection per 64 KiB')
    async def test_large_file_round_trip(self, backend: WorkspaceBackend) -> None: ...

    @pytest.fixture
    def can_detect_exit_with_inherited_output_pipes(self) -> bool:
        # A real `sshd` keeps the session open until every copy of the command's output is closed;
        # the fake `ssh` runs locally, so it would pass where a real host hangs.
        return False

    @staticmethod
    def attach(ref: WorkspaceRef) -> WorkspaceBackend:
        destination, _, working_dir = ref.id.partition(':')
        return SSHWorkspaceBackend(destination, working_dir=working_dir)

    @pytest.fixture
    def backend(self, tmp_path: Path) -> WorkspaceBackend:
        return SSHWorkspaceBackend('box', working_dir=str(tmp_path))

    @pytest.fixture
    def destructive_backend(self, tmp_path_factory: pytest.TempPathFactory) -> Callable[[], WorkspaceBackend]:
        path = tmp_path_factory.mktemp('fresh-ssh-ws')
        return lambda: self.attach(WorkspaceRef(provider='ssh', id=f'box:{path}'))

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return self.attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await anyio.to_thread.run_sync(shutil.rmtree, backend.ref.id.partition(':')[2])

        return destroy


class TestBubblewrapAroundSSH(TestSSHWorkspaceBackend):
    """The same rules hold with every command going through (a fake) `bwrap` on the SSH host."""

    @staticmethod
    def attach(ref: WorkspaceRef) -> WorkspaceBackend:
        return BubblewrapWorkspace(Workspace(TestSSHWorkspaceBackend.attach(ref)))

    @pytest.fixture
    def backend(self, tmp_path: Path) -> WorkspaceBackend:
        return BubblewrapWorkspace(Workspace(SSHWorkspaceBackend('box', working_dir=str(tmp_path))))


class _StrictLoopBackend(LocalWorkspaceBackend):
    async def realpath(self, path: str) -> str:
        def resolve() -> str:
            try:
                return os.path.realpath(path, strict=True)
            except FileNotFoundError:
                return os.path.realpath(path)

        return await anyio.to_thread.run_sync(resolve)


class TestStrictLoopBackend:
    @pytest.fixture
    def backend(self, tmp_path: Path) -> LocalWorkspaceBackend:
        return _StrictLoopBackend(tmp_path)

    test_realpath_and_entries_follow_symlinks = WorkspaceBackendSuite.test_realpath_and_entries_follow_symlinks

    async def test_realpath_loop_raises_oserror(self, backend: LocalWorkspaceBackend, tmp_path: Path) -> None:
        (tmp_path / 'loop').symlink_to('loop')
        with pytest.raises(OSError):
            await backend.realpath(str(tmp_path / 'loop' / 'child'))


class TestFilesystemOnlyWorkspaceBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def enforces_parent_file_errors(self) -> bool:
        return False  # The in-memory fake records paths without traversing parent directories.

    @pytest.fixture
    def has_real_posix_shell(self) -> bool:
        return False  # No shell: only a dict-backed filesystem.

    @pytest.fixture
    def backend(self) -> FilesystemOnlyWorkspaceBackend:
        return FilesystemOnlyWorkspaceBackend(FakeWorkspace('filesystem-only-conformance'))


class TestRunOnlyWorkspaceBackend(WorkspaceBackendSuite):
    """Certifies the file operations `Workspace` derives through the shell for a command-only backend."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> RunOnlyWorkspaceBackend:
        return RunOnlyWorkspaceBackend(LocalWorkspaceBackend(tmp_path))

    @pytest.fixture
    def destructive_backend(self, tmp_path_factory: pytest.TempPathFactory) -> Callable[[], WorkspaceBackend]:
        path = tmp_path_factory.mktemp('fresh-run-ws')
        return lambda: RunOnlyWorkspaceBackend(LocalWorkspaceBackend(path))

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: RunOnlyWorkspaceBackend(LocalWorkspaceBackend(ref.id))

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _local_destroy_environment()


class TestProviderBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def enforces_parent_file_errors(self) -> bool:
        return False  # Its in-memory provider uses the same simplified file map.

    @pytest.fixture
    def has_real_posix_shell(self) -> bool:
        return False  # Its `run` is a stub, not a POSIX process.

    @pytest.fixture
    def provider(self) -> InMemoryProvider:
        return InMemoryProvider('conformance-provider')

    @pytest.fixture
    def backend(self, provider: InMemoryProvider) -> WorkspaceBackend:
        return FilesystemOnlyWorkspaceBackend(provider.backend(None))

    @pytest.fixture
    def fresh_backend(self, provider: InMemoryProvider) -> Callable[[], WorkspaceBackend]:
        return lambda: FilesystemOnlyWorkspaceBackend(provider.backend(None))

    @pytest.fixture
    def attach_backend(self, provider: InMemoryProvider) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: FilesystemOnlyWorkspaceBackend(provider.backend(ref))

    @pytest.fixture
    def destroy_environment(self, provider: InMemoryProvider) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            ref = backend.ref
            assert ref is not None
            provider.environments.pop(ref.id, None)
            provider.directories.pop(ref.id, None)

        return destroy


class _NormalizingRealpathBackend(FilesystemOnlyWorkspaceBackend):
    async def realpath(self, path: str) -> str:
        return posixpath.normpath(path)


class _RelativeRealpathBackend(FilesystemOnlyWorkspaceBackend):
    async def realpath(self, path: str) -> str:
        return path.lstrip('/')


async def test_native_realpath_rule_runs_without_commands() -> None:
    """The other `realpath` rules need `ln -s`, so a filesystem-only backend's `realpath` needs its own."""
    rule = WorkspaceBackendSuite().test_native_realpath_keeps_the_working_dir_and_missing_names
    await rule(_NormalizingRealpathBackend(FakeWorkspace('normalizing-realpath')))
    with pytest.raises(AssertionError):
        await rule(_RelativeRealpathBackend(FakeWorkspace('relative-realpath')))
