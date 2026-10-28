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
    ComputerAction,
    LocalWorkspaceBackend,
    Point,
    Region,
    SupportsCursorPosition,
    SupportsScaledScreenshot,
    SupportsZoom,
    WorkspaceBackend,
    WorkspaceRef,
    local as local_module,
)
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite

from .workspace_fakes import (
    DisplayOnlyBackend,
    FakeWorkspace,
    FilesystemOnlyWorkspaceBackend,
    InMemoryProvider,
    RunOnlyWorkspaceBackend,
)

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='workspace conformance command rules use POSIX sh')


@pytest.fixture(autouse=True)
def short_output_drain_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Don't wait out `LocalWorkspaceBackend`'s 2s drain grace in the background-child rule.

    The rule only asserts that the command returns while its background child holds stdout open, not
    how long the backend keeps reading first.
    """
    monkeypatch.setattr(local_module, '_OUTPUT_DRAIN_GRACE', 0.05)


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


class _ProviderFilesAndDisplay(DisplayOnlyBackend, SupportsZoom, SupportsScaledScreenshot, SupportsCursorPosition):
    """The provider's files and display, without its stub `run`."""

    async def zoom(self, region: Region) -> bytes:
        return await self._display.zoom(region)

    async def scaled_screenshot(self, width: int, height: int) -> bytes:
        return await self._display.scaled_screenshot(width, height)

    async def cursor_position(self) -> Point:
        return await self._display.cursor_position()


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
        return _ProviderFilesAndDisplay(provider.backend(None))

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


@pytest.mark.anyio
async def test_the_zoom_and_pointer_checks_skip_a_display_without_them() -> None:
    suite = WorkspaceBackendSuite()
    backend = DisplayOnlyBackend(InMemoryProvider().backend(None))
    for check in (
        suite.test_zoom_is_a_png_of_the_region,
        suite.test_a_scaled_screenshot_is_a_png_of_the_size_asked,
        suite.test_the_cursor_position_is_on_the_screen,
        suite.test_moving_the_mouse_puts_the_pointer_there,
    ):
        with pytest.raises(pytest.skip.Exception):
            await check(backend)


@pytest.mark.anyio
async def test_the_display_check_fails_a_backend_with_a_new_display_each_time() -> None:
    class NewDisplayEachTime(DisplayOnlyBackend):
        @property
        def display(self) -> DisplayOnlyBackend:
            return DisplayOnlyBackend(self._display)

    backend = NewDisplayEachTime(InMemoryProvider().backend(None))
    with pytest.raises(AssertionError):
        await WorkspaceBackendSuite().test_display_is_the_same_object_each_time(backend)


@pytest.mark.anyio
async def test_the_zoom_check_allows_a_full_resolution_capture_of_the_region() -> None:
    """On a display whose screenshots are scaled down, a zoom at full resolution is larger than the region."""

    class ScaledZoom(_ProviderFilesAndDisplay):
        async def zoom(self, region: Region) -> bytes:
            return await super().zoom(Region(left=0, top=0, right=region.right * 2, bottom=region.bottom * 2))

    class StretchedZoom(_ProviderFilesAndDisplay):
        async def zoom(self, region: Region) -> bytes:
            return await super().zoom(Region(left=0, top=0, right=region.right * 2, bottom=region.bottom))

    rule = WorkspaceBackendSuite().test_zoom_is_a_png_of_the_region
    await rule(ScaledZoom(InMemoryProvider().backend(None)))
    with pytest.raises(AssertionError):
        await rule(StretchedZoom(InMemoryProvider().backend(None)))


@pytest.mark.anyio
async def test_the_pointer_check_fails_a_display_that_ignores_actions() -> None:
    class InertDisplay(_ProviderFilesAndDisplay):
        async def act(self, action: ComputerAction) -> None:
            pass

    rule = WorkspaceBackendSuite().test_moving_the_mouse_puts_the_pointer_there
    await rule(_ProviderFilesAndDisplay(InMemoryProvider().backend(None)))
    with pytest.raises(AssertionError, match='pointer at'):
        await rule(InertDisplay(InMemoryProvider().backend(None)))


@pytest.mark.anyio
async def test_the_scaled_screenshot_check_fails_a_display_that_ignores_the_size() -> None:
    class FullSize(_ProviderFilesAndDisplay):
        async def scaled_screenshot(self, width: int, height: int) -> bytes:
            return await self.screenshot()

    with pytest.raises(AssertionError):
        await WorkspaceBackendSuite().test_a_scaled_screenshot_is_a_png_of_the_size_asked(
            FullSize(InMemoryProvider().backend(None))
        )
