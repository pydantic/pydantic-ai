"""The public workspace conformance suite, run once per kind of backend core supports."""

from __future__ import annotations

import os
import shutil
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

import anyio.to_thread
import pytest

from pydantic_ai.workspaces import (
    LocalWorkspaceBackend,
    WorkspaceBackend,
    WorkspaceFileEntry,
    WorkspaceRef,
)
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite

from .workspace_fakes import (
    FakeWorkspace,
    FilesystemOnlyWorkspaceBackend,
    InMemoryProvider,
    ProviderBackend,
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


class _FilesystemProviderBackend:
    def __init__(self, backend: ProviderBackend) -> None:
        self.backend = backend

    @property
    def ref(self) -> WorkspaceRef | None:
        return self.backend.ref

    async def working_dir(self) -> str:
        return await self.backend.working_dir()

    async def read_bytes(self, path: str) -> bytes:
        return await self.backend.read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self.backend.write_bytes(path, data)

    async def stat(self, path: str) -> WorkspaceFileEntry:
        return await self.backend.stat(path)

    async def list_dir(self, path: str) -> Sequence[WorkspaceFileEntry]:
        return await self.backend.list_dir(path)

    async def make_dir(self, path: str) -> None:
        await self.backend.make_dir(path)

    async def remove(self, path: str) -> None:
        await self.backend.remove(path)

    async def exists(self, path: str) -> bool:
        return await self.backend.exists(path)


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
        return _FilesystemProviderBackend(provider.backend(None))

    @pytest.fixture
    def fresh_backend(self, provider: InMemoryProvider) -> Callable[[], WorkspaceBackend]:
        return lambda: _FilesystemProviderBackend(provider.backend(None))

    @pytest.fixture
    def attach_backend(self, provider: InMemoryProvider) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: _FilesystemProviderBackend(provider.backend(ref))

    @pytest.fixture
    def destroy_environment(self, provider: InMemoryProvider) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            ref = backend.ref
            assert ref is not None
            provider.environments.pop(ref.id, None)
            provider.directories.pop(ref.id, None)

        return destroy
