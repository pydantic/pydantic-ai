"""Pydantic AI's workspace backend conformance suite, run against `BoxdSandboxBackend`.

`TestFakeBoxdSandboxBackend` runs everywhere, over the fake boxd SDK, where commands act on a
temporary host directory. `TestLiveBoxdSandboxBackend` runs the same rules against a real machine
and is gated like `test_boxd_live.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox, BoxdSandboxBackend

from .conftest import LIVE_AUTO_SUSPEND_TIMEOUT
from .fake_boxd import FakeBoxd


async def _delete(backend: WorkspaceBackend) -> None:
    assert isinstance(backend, BoxdSandboxBackend) and backend.ref is not None
    await BoxdSandbox().destroy(backend.ref)


class TestFakeBoxdSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, fake_boxd: FakeBoxd, tmp_path: Path) -> BoxdSandboxBackend:
        # Resolved so the working directory `pwd -P` reports matches it on hosts whose temp
        # directory sits behind a symlink.
        fake_boxd.host_root = tmp_path.resolve()
        return BoxdSandboxBackend()

    @pytest.fixture
    def fresh_backend(self, fake_boxd: FakeBoxd, tmp_path: Path) -> Callable[[], WorkspaceBackend]:
        fake_boxd.host_root = tmp_path.resolve()
        return BoxdSandboxBackend

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: BoxdSandboxBackend(ref=ref)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete


@pytest.mark.boxd_live
# The live tier runs without coverage; in CI only its gate fixtures run, to skip it.
class TestLiveBoxdSandboxBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    # Destructive rules need their own machine, not the shared class-scoped one. Each is deleted
    # here if a failed rule left it behind, and the client it opened is closed.
    @pytest.fixture
    async def destructive_backend(self) -> AsyncIterator[Callable[[], WorkspaceBackend]]:
        created: list[BoxdSandboxBackend] = []

        def create() -> WorkspaceBackend:
            backend = BoxdSandboxBackend(auto_suspend_timeout=LIVE_AUTO_SUSPEND_TIMEOUT)
            created.append(backend)
            return backend

        yield create
        for backend in created:
            try:
                if backend.ref is not None:
                    await BoxdSandbox().destroy(backend.ref)
            finally:
                await backend.aclose()

    # Class-scoped so the rules share one machine instead of creating one each; the suite runs
    # its destroy rule last.
    @pytest.fixture(scope='class')
    @classmethod
    async def backend(cls) -> AsyncIterator[BoxdSandboxBackend]:
        backend = BoxdSandboxBackend(auto_suspend_timeout=LIVE_AUTO_SUSPEND_TIMEOUT)
        try:
            await backend.get_sandbox()
            yield backend
        finally:
            try:
                if backend.ref is not None:
                    await BoxdSandbox().destroy(backend.ref)
            finally:
                await backend.aclose()

    # The suite never deletes what `fresh_backend` creates, so each machine is deleted here.
    @pytest.fixture
    async def fresh_backend(self) -> AsyncIterator[Callable[[], WorkspaceBackend]]:
        created: list[BoxdSandboxBackend] = []

        def create() -> WorkspaceBackend:
            backend = BoxdSandboxBackend(auto_suspend_timeout=LIVE_AUTO_SUSPEND_TIMEOUT)
            created.append(backend)
            return backend

        yield create
        for backend in created:
            try:
                if backend.ref is not None:
                    await BoxdSandbox().destroy(backend.ref)
            finally:
                await backend.aclose()

    # The suite drops the backend it attaches; each one opened its own SDK client, which is closed here.
    @pytest.fixture
    async def attach_backend(self) -> AsyncIterator[Callable[[WorkspaceRef], WorkspaceBackend]]:
        attached: list[BoxdSandboxBackend] = []

        def attach(ref: WorkspaceRef) -> WorkspaceBackend:
            backend = BoxdSandboxBackend(ref=ref)
            attached.append(backend)
            return backend

        yield attach
        for backend in attached:
            await backend.aclose()

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete
