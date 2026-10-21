"""Pydantic AI's workspace backend conformance suite, run against `DaytonaSandboxBackend`.

`TestFakeDaytonaSandboxBackend` runs everywhere, over the fake Daytona SDK in host mode, where
commands and file operations act on a temporary host directory. `TestLiveDaytonaSandboxBackend`
runs the same rules against a real sandbox and is gated like `test_daytona_live.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox, DaytonaSandboxBackend

from .conftest import LIVE_AUTO_STOP_INTERVAL
from .fake_daytona import FakeDaytona


def _attach(ref: WorkspaceRef) -> WorkspaceBackend:
    return DaytonaSandboxBackend(ref=ref)


async def _delete(backend: WorkspaceBackend) -> None:
    assert isinstance(backend, DaytonaSandboxBackend)
    sandbox = await backend.get_sandbox()
    await sandbox.delete()


async def _delete_if_live(sandbox_id: str) -> None:  # pragma: no cover - live tier runs without coverage
    # Quiet when the destroy rule already deleted it, and closes the client it opens.
    await DaytonaSandbox().destroy(WorkspaceRef(provider='daytona', id=sandbox_id))


class TestFakeDaytonaSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, fake_daytona: FakeDaytona, tmp_path: Path) -> DaytonaSandboxBackend:
        # Resolved so the working directory `pwd -P` reports matches it on hosts whose temp
        # directory sits behind a symlink.
        fake_daytona.host_root = tmp_path.resolve()
        return DaytonaSandboxBackend()

    @pytest.fixture
    def fresh_backend(self, fake_daytona: FakeDaytona, tmp_path: Path) -> Callable[[], WorkspaceBackend]:
        fake_daytona.host_root = tmp_path.resolve()
        return DaytonaSandboxBackend

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete


@pytest.mark.daytona_live
# The live tier runs without coverage; in CI only its gate fixtures run, to skip it.
class TestLiveDaytonaSandboxBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    # Destructive rules need their own sandbox, not the shared class-scoped one. Each is deleted
    # here if a failed rule left it behind, and the client it opened is closed rather than left to
    # the garbage collector with a connection still open.
    @pytest.fixture
    async def destructive_backend(self) -> AsyncIterator[Callable[[], WorkspaceBackend]]:
        created: list[DaytonaSandboxBackend] = []

        def create() -> WorkspaceBackend:
            backend = DaytonaSandboxBackend(auto_stop_interval=LIVE_AUTO_STOP_INTERVAL)
            created.append(backend)
            return backend

        yield create
        for backend in created:
            try:
                if backend.ref is not None:
                    await _delete_if_live(backend.ref.id)
            finally:
                await backend.aclose()

    # Class-scoped so the rules share one sandbox instead of starting one each; the suite runs
    # its destroy rule last.
    @pytest.fixture(scope='class')
    @classmethod
    async def backend(cls) -> AsyncIterator[DaytonaSandboxBackend]:
        backend = DaytonaSandboxBackend(auto_stop_interval=LIVE_AUTO_STOP_INTERVAL)
        try:
            await backend.get_sandbox()
            yield backend
        finally:
            try:
                if backend.ref is not None:
                    await _delete_if_live(backend.ref.id)
            finally:
                await backend.aclose()

    # The suite drops the backend it attaches; each one opened its own SDK client, which is closed
    # here rather than left to the garbage collector with a connection still open.
    @pytest.fixture
    async def attach_backend(self) -> AsyncIterator[Callable[[WorkspaceRef], WorkspaceBackend]]:
        attached: list[DaytonaSandboxBackend] = []

        def attach(ref: WorkspaceRef) -> WorkspaceBackend:
            backend = DaytonaSandboxBackend(ref=ref)
            attached.append(backend)
            return backend

        yield attach
        for backend in attached:
            await backend.aclose()

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete
