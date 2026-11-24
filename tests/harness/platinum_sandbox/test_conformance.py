"""Pydantic AI's workspace backend conformance suite, run against `PlatinumSandboxBackend`.

`TestFakePlatinumSandboxBackend` runs on POSIX hosts, over the fake Platinum API, where commands and file operations
act on a temporary host directory. `TestLivePlatinumSandboxBackend` runs the same rules against a real sandbox and is
gated like `test_platinum_live.py`.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.platinum_sandbox import PlatinumSandbox, PlatinumSandboxBackend

from .conftest import LIVE_AUTO_STOP_MINUTES, delete_live_sandbox
from .fake_platinum import FakePlatinum


async def _delete(backend: WorkspaceBackend) -> None:
    assert isinstance(backend, PlatinumSandboxBackend) and backend.ref is not None
    await PlatinumSandbox().destroy(backend.ref)


@pytest.mark.skipif(os.name != 'posix', reason='needs a POSIX host')
class TestFakePlatinumSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, fake_platinum: FakePlatinum, work: Path) -> PlatinumSandboxBackend:
        return PlatinumSandboxBackend(working_dir=str(work))

    @pytest.fixture
    def fresh_backend(self, fake_platinum: FakePlatinum, work: Path) -> Callable[[], WorkspaceBackend]:
        return lambda: PlatinumSandboxBackend(working_dir=str(work))

    @pytest.fixture
    def attach_backend(self, work: Path) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: PlatinumSandboxBackend(ref=ref, working_dir=str(work))

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete


@pytest.mark.platinum_live
# The live tier runs without coverage; in CI only its gate fixtures run, to skip it.
class TestLivePlatinumSandboxBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    # Class-scoped so the rules share one sandbox instead of creating one each; the suite runs its destroy rule last.
    @pytest.fixture(scope='class')
    @classmethod
    async def backend(cls) -> AsyncIterator[PlatinumSandboxBackend]:
        backend = PlatinumSandboxBackend(auto_stop_minutes=LIVE_AUTO_STOP_MINUTES)
        try:
            await backend.get_sandbox()
            yield backend
        finally:
            try:
                if backend.ref is not None:
                    await delete_live_sandbox(backend.ref)
            finally:
                await backend.aclose()

    # The suite never deletes what `fresh_backend` creates, and uses it for its destructive rules too, so each
    # sandbox is deleted here.
    @pytest.fixture
    async def fresh_backend(self) -> AsyncIterator[Callable[[], WorkspaceBackend]]:
        created: list[PlatinumSandboxBackend] = []

        def create() -> WorkspaceBackend:
            backend = PlatinumSandboxBackend(auto_stop_minutes=LIVE_AUTO_STOP_MINUTES)
            created.append(backend)
            return backend

        yield create
        for backend in created:
            try:
                if backend.ref is not None:
                    await delete_live_sandbox(backend.ref)
            finally:
                await backend.aclose()

    # The suite drops the backend it attaches; each one opened its own SDK client, which is closed here.
    @pytest.fixture
    async def attach_backend(self) -> AsyncIterator[Callable[[WorkspaceRef], WorkspaceBackend]]:
        attached: list[PlatinumSandboxBackend] = []

        def attach(ref: WorkspaceRef) -> WorkspaceBackend:
            backend = PlatinumSandboxBackend(ref=ref)
            attached.append(backend)
            return backend

        yield attach
        for backend in attached:
            await backend.aclose()

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete
