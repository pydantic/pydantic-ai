"""Run the public workspace backend conformance suite over the built-in test backends."""

from __future__ import annotations

import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

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


def test_suite_requires_backend_fixture() -> None:
    fixture = cast(
        Callable[[WorkspaceBackendSuite], WorkspaceBackend], getattr(WorkspaceBackendSuite.backend, '__wrapped__')
    )
    with pytest.raises(NotImplementedError, match='provide a `backend` fixture'):
        fixture(WorkspaceBackendSuite())


class _BrokenWorkingDir(FakeWorkspace):
    async def working_dir(self) -> str:
        raise RuntimeError('broken working directory')


class _CleanupFake(FakeWorkspace):
    def __init__(self, cleanup_error: Exception) -> None:
        super().__init__('cleanup')
        self.cleanup_error = cleanup_error

    async def remove(self, path: str) -> None:
        # The exists rule only removes the probe root, so every call here is the probe cleanup.
        raise self.cleanup_error


async def test_suite_failures_quote_the_rule() -> None:
    suite = WorkspaceBackendSuite()
    with pytest.raises(AssertionError, match='Rule: "Structural protocol:'):
        await suite.test_required_members(cast(WorkspaceBackend, object()))
    with pytest.raises(AssertionError, match=r'Rule: "The workspace.*default working directory'):
        await suite.test_default_working_dir_is_canonical(_BrokenWorkingDir('broken'))

    await suite.test_filesystem_exists_is_truthful(_CleanupFake(FileNotFoundError('already removed')))
    with pytest.raises(AssertionError, match=r'probe cleanup raised RuntimeError[\s\S]*Rule: "Whether a file'):
        await suite.test_filesystem_exists_is_truthful(_CleanupFake(RuntimeError('cleanup failed')))


class _IdentityOnlyBackend:
    @property
    def ref(self) -> WorkspaceRef | None:
        return None

    async def working_dir(self) -> str:
        return '/'


async def test_filesystem_rules_skip_without_commands_or_filesystem() -> None:
    with pytest.raises(pytest.skip.Exception, match='neither SupportsFilesystem nor SupportsCommands'):
        await WorkspaceBackendSuite().test_filesystem_exists_is_truthful(_IdentityOnlyBackend())


class TestLocalWorkspaceBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, tmp_path: Path) -> LocalWorkspaceBackend:
        return LocalWorkspaceBackend(tmp_path)

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: LocalWorkspaceBackend(ref.id)

    @pytest.fixture
    def destroy_environment(self, tmp_path: Path) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            await anyio.to_thread.run_sync(shutil.rmtree, tmp_path)

        return destroy


class TestClassScopedLocalWorkspaceBackend(WorkspaceBackendSuite):
    @pytest.fixture(scope='class')
    @classmethod
    def backend(cls, tmp_path_factory: pytest.TempPathFactory) -> LocalWorkspaceBackend:
        return LocalWorkspaceBackend(working_dir=tmp_path_factory.mktemp('ws'))


class TestFakeWorkspace(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self) -> FakeWorkspace:
        return FakeWorkspace('conformance')


class TestFilesystemOnlyWorkspaceBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self) -> FilesystemOnlyWorkspaceBackend:
        return FilesystemOnlyWorkspaceBackend(FakeWorkspace('filesystem-only-conformance'))


class TestRunOnlyWorkspaceBackend(WorkspaceBackendSuite):
    """Certifies the file operations `Workspace` derives through the shell for a command-only backend."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> RunOnlyWorkspaceBackend:
        return RunOnlyWorkspaceBackend(LocalWorkspaceBackend(tmp_path))

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: RunOnlyWorkspaceBackend(LocalWorkspaceBackend(ref.id))

    @pytest.fixture
    def destroy_environment(self, tmp_path: Path) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            await anyio.to_thread.run_sync(shutil.rmtree, tmp_path)

        return destroy


class TestProviderBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def provider(self) -> InMemoryProvider:
        return InMemoryProvider('conformance-provider')

    @pytest.fixture
    def backend(self, provider: InMemoryProvider) -> ProviderBackend:
        return provider.backend(None)

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
