"""Pydantic AI's workspace backend conformance suite, run against `E2BSandboxBackend`.

`TestFakeE2BSandboxBackend` runs on POSIX hosts, over the fake E2B SDK in host mode, where
commands and file operations act on a temporary host directory. `TestLiveE2BSandboxBackend`
runs the same rules against a real sandbox and is gated like `test_e2b_live.py`.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import Unsupported, WorkspaceBackendSuite
from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend

from .fake_e2b import FakeE2B

# Strict expected failures: each one fails CI once its backend is fixed, so the mark comes off with the fix.
_QUOTED_TWICE = pytest.mark.xfail(strict=True, reason='E2B quotes an argv item twice, past the limit on one argument')
_READS_DEVICES = pytest.mark.xfail(strict=True, reason='E2B reads a device file instead of refusing it')


def _attach(ref: WorkspaceRef) -> WorkspaceBackend:
    return E2BSandboxBackend(ref=ref)


async def _kill(backend: WorkspaceBackend) -> None:
    assert isinstance(backend, E2BSandboxBackend)
    sandbox = await backend.get_sandbox()
    await sandbox.kill()


@pytest.mark.skipif(os.name != 'posix', reason='needs a POSIX host')
class TestFakeE2BSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, fake_e2b: FakeE2B, tmp_path: Path) -> E2BSandboxBackend:
        # Resolved so the working directory `pwd -P` reports matches it on hosts whose temp
        # directory sits behind a symlink.
        fake_e2b.host_root = tmp_path.resolve()
        return E2BSandboxBackend()

    @pytest.fixture
    def fresh_backend(self, backend: E2BSandboxBackend) -> Callable[[], WorkspaceBackend]:
        # `backend` points the fake E2B at the host directory; each call starts another sandbox there.
        return E2BSandboxBackend

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _kill

    @_QUOTED_TWICE
    async def test_a_large_quote_heavy_argv_item_is_literal(self, backend: WorkspaceBackend) -> None:
        await super().test_a_large_quote_heavy_argv_item_is_literal(backend)

    @_READS_DEVICES
    async def test_reading_a_device_raises_os_error(
        self, backend: WorkspaceBackend, has_real_posix_shell: bool
    ) -> None:
        await super().test_reading_a_device_raises_os_error(backend, has_real_posix_shell)


# The live tier runs without coverage; in CI only its gate fixtures run, to skip it.
@pytest.mark.e2b_live
class TestLiveE2BSandboxBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    # Class-scoped so the rules share one sandbox instead of starting one each. The teardown
    # uses E2B's blocking API because a class-scoped fixture outlives each test's event loop.
    @pytest.fixture(scope='class')
    @classmethod
    def backend(cls) -> Iterator[E2BSandboxBackend]:
        backend = E2BSandboxBackend(sandbox_timeout=600)
        yield backend
        if backend.ref is not None:
            import e2b

            e2b.Sandbox.kill(backend.ref.id)

    async def test_symlink_loop_does_not_break_listing(
        self, backend: WorkspaceBackend, has_real_posix_shell: bool
    ) -> None:
        # envd's list omits a looping symlink entirely (live probe, 2026-09-26);
        # get_info cannot restore an entry the SDK never returns.
        pytest.skip('E2B envd omits looping symlinks from directory listings')

    @pytest.fixture
    def can_detect_exit_with_inherited_output_pipes(self) -> bool:
        # Live envd (2026-09-26) closes the command stream only after a background child holding
        # stdout exits, so `run()` waits for it. Documented with a redirect workaround.
        return False

    @pytest.fixture
    def filesystem_honors_shell_permissions(self) -> bool:
        # envd opens files with its own root privileges; `user=` only sets ownership and `~`
        # (envd internal/api/download.go), so a shell chmod 000 does not stop the file API.
        return False

    @pytest.fixture
    def destructive_backend(self) -> Callable[[], WorkspaceBackend]:
        # Destructive rules need their own sandbox, not the shared class-scoped fixture.
        return lambda: E2BSandboxBackend(sandbox_timeout=600)

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _kill

    @pytest.fixture
    def fresh_backend(self) -> Unsupported:
        return Unsupported('the fake tier checks concurrent first use, so the live tier creates no sandbox for it')

    @_QUOTED_TWICE
    async def test_a_large_quote_heavy_argv_item_is_literal(self, backend: WorkspaceBackend) -> None:
        await super().test_a_large_quote_heavy_argv_item_is_literal(backend)

    @_READS_DEVICES
    async def test_reading_a_device_raises_os_error(
        self, backend: WorkspaceBackend, has_real_posix_shell: bool
    ) -> None:
        await super().test_reading_a_device_raises_os_error(backend, has_real_posix_shell)
