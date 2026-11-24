"""Pydantic AI's workspace backend conformance suite, run against `RuntaSandboxBackend`.

`TestFakeRuntaSandboxBackend` runs on POSIX hosts, over the fake Runta API, whose commands run in
local subprocesses under a temporary host directory. `TestLiveRuntaSandboxBackend` runs the same
rules against a real runtime, behind `PYDANTIC_AI_HARNESS_RUNTA_LIVE=1` and the `runta_live` marker.

The backend implements `SupportsCommands`, so the suite's command, reattach, and destroy rules run
here, and its filesystem rules run through the file operations Pydantic AI derives from commands.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.runta_sandbox import RuntaSandbox, RuntaSandboxBackend

from .fake_runta import FakeRunta


def _attach(ref: WorkspaceRef) -> WorkspaceBackend:
    return RuntaSandboxBackend(ref=ref)


async def _delete(backend: WorkspaceBackend) -> None:
    assert backend.ref is not None
    await RuntaSandbox[None]().destroy(backend.ref)


@pytest.mark.skipif(os.name != 'posix', reason='needs a POSIX host')
class TestFakeRuntaSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, runta: FakeRunta) -> RuntaSandboxBackend:
        del runta
        return RuntaSandboxBackend()

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete


@pytest.mark.runta_live
@pytest.mark.usefixtures('runta_token')
# The live tier runs without coverage.
class TestLiveRuntaSandboxBackend(WorkspaceBackendSuite):  # pragma: no cover
    @pytest.fixture(scope='class')
    @classmethod
    def anyio_backend(cls) -> str:
        return 'asyncio'

    # Class-scoped so the rules share one runtime instead of creating one each.
    @pytest.fixture(scope='class')
    @classmethod
    async def backend(cls) -> AsyncIterator[RuntaSandboxBackend]:
        backend = RuntaSandboxBackend()
        yield backend
        if backend.ref is not None:
            await _delete(backend)

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    # Destructive rules need their own runtime, not the shared class-scoped one. A runtime has no
    # lifetime of its own, so one a failed rule left behind is deleted here.
    @pytest.fixture
    async def destructive_backend(self) -> AsyncIterator[Callable[[], WorkspaceBackend]]:
        created: list[RuntaSandboxBackend] = []

        def create() -> WorkspaceBackend:
            backend = RuntaSandboxBackend()
            created.append(backend)
            return backend

        yield create
        for backend in created:
            if backend.ref is not None:
                await _delete(backend)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete
