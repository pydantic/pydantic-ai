"""The public workspace conformance suite, run once per kind of backend core supports."""

from __future__ import annotations

import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path

import anyio.to_thread
import pytest

from pydantic_ai.workspaces import LocalWorkspaceBackend, WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.testing import WorkspaceBackendSuite

from .workspace_fakes import (
    FakeWorkspace,
    FilesystemOnlyWorkspaceBackend,
    InMemoryProvider,
    ProviderBackend,
    RunOnlyWorkspaceBackend,
)

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='workspace conformance command rules use POSIX sh')


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
        async def destroy(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await anyio.to_thread.run_sync(shutil.rmtree, backend.ref.id)

        return destroy


class TestSharedLocalWorkspaceBackend(WorkspaceBackendSuite):
    @pytest.fixture(scope='class')
    @classmethod
    def backend(cls, tmp_path_factory: pytest.TempPathFactory) -> LocalWorkspaceBackend:
        return LocalWorkspaceBackend(tmp_path_factory.mktemp('shared-ws'))

    @pytest.fixture
    def destructive_backend(self, tmp_path_factory: pytest.TempPathFactory) -> Callable[[], WorkspaceBackend]:
        path = tmp_path_factory.mktemp('destructive-ws')
        return lambda: LocalWorkspaceBackend(path)

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: LocalWorkspaceBackend(ref.id)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await anyio.to_thread.run_sync(shutil.rmtree, backend.ref.id)

        return destroy


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
        async def destroy(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await anyio.to_thread.run_sync(shutil.rmtree, backend.ref.id)

        return destroy


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
    def backend(self, provider: InMemoryProvider) -> ProviderBackend:
        return provider.backend(None)

    @pytest.fixture
    def fresh_backend(self, provider: InMemoryProvider) -> Callable[[], WorkspaceBackend]:
        return lambda: provider.backend(None)

    @pytest.fixture
    def attach_backend(self, provider: InMemoryProvider) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return provider.backend

    @pytest.fixture
    def destroy_environment(self, provider: InMemoryProvider) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            ref = backend.ref
            assert ref is not None
            provider.environments.pop(ref.id, None)
            provider.directories.pop(ref.id, None)

        return destroy
