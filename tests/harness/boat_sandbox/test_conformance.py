"""Pydantic AI's workspace backend conformance suite, run against `BoatSandboxBackend`.

`TestFakeBoatSandboxBackend` runs everywhere, over the fake Boat API, whose sandboxes run real
commands and file operations in temporary host directories. `TestLiveBoatSandboxBackend` runs the
same rules against a real sandbox and is gated by the `boat_live` marker.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.boat_sandbox import BoatSandbox, BoatSandboxBackend

from .conftest import LIVE_AUTO_STOP
from .fake_boat import FakeBoat


def _attach(ref: WorkspaceRef) -> WorkspaceBackend:
    return BoatSandboxBackend(ref=ref)


async def _delete(backend: WorkspaceBackend) -> None:
    assert backend.ref is not None
    await BoatSandbox().destroy(backend.ref)


# The live tier runs without coverage.
async def _delete_if_live(sandbox_id: str) -> None:  # pragma: no cover
    # Quiet when the destroy rule already deleted it, and closes the client it opens.
    await BoatSandbox().destroy(WorkspaceRef(provider='boat', id=sandbox_id))


class TestFakeBoatSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, fake_boat: FakeBoat) -> BoatSandboxBackend:
        return BoatSandboxBackend()

    @pytest.fixture
    def fresh_backend(self, fake_boat: FakeBoat) -> Callable[[], WorkspaceBackend]:
        return BoatSandboxBackend

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete

    @pytest.fixture
    def filesystem_honors_shell_permissions(self) -> bool:
        # Boat's file API reads as root, so a file the command user cannot read is still read;
        # the fake reads as the host user, and opts out to match Boat.
        return False


@pytest.mark.boat_live
# The live tier runs without coverage; in CI only its gate fixtures run, to skip it.
class TestLiveBoatSandboxBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    @pytest.fixture
    def filesystem_honors_shell_permissions(self) -> bool:
        # Boat's file API reads as root, so a file the command user cannot read is still read.
        return False

    # Destructive rules need their own sandbox, not the shared class-scoped one. Each is deleted
    # here if a failed rule left it behind, and the client it opened is closed rather than left to
    # the garbage collector with a connection still open.
    @pytest.fixture
    async def destructive_backend(self) -> AsyncIterator[Callable[[], WorkspaceBackend]]:
        created: list[BoatSandboxBackend] = []

        def create() -> WorkspaceBackend:
            backend = BoatSandboxBackend(machine_type='small', auto_stop=LIVE_AUTO_STOP)
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
    async def backend(cls) -> AsyncIterator[BoatSandboxBackend]:
        backend = BoatSandboxBackend(machine_type='small', auto_stop=LIVE_AUTO_STOP)
        try:
            await backend.working_dir()
            yield backend
        finally:
            try:
                if backend.ref is not None:
                    await _delete_if_live(backend.ref.id)
            finally:
                await backend.aclose()

    # The suite drops the backend it attaches; each one opened its own API client, which is closed
    # here rather than left to the garbage collector with a connection still open.
    @pytest.fixture
    async def attach_backend(self) -> AsyncIterator[Callable[[WorkspaceRef], WorkspaceBackend]]:
        attached: list[BoatSandboxBackend] = []

        def attach(ref: WorkspaceRef) -> WorkspaceBackend:
            backend = BoatSandboxBackend(ref=ref)
            attached.append(backend)
            return backend

        yield attach
        for backend in attached:
            await backend.aclose()

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete
