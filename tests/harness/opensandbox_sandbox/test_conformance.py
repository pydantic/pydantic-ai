"""Pydantic AI's workspace backend conformance suite, run against `OpenSandboxBackend`.

`TestFakeOpenSandboxBackend` runs everywhere, over the fake server, where commands run in a temporary host
directory. `TestLiveOpenSandboxBackend` runs the same rules against a real server and is gated like
`test_opensandbox_live.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox, OpenSandboxBackend

from .fake_opensandbox import FakeOpenSandbox
from .test_opensandbox_live import LIVE_IMAGE, live_capability


class TestFakeOpenSandboxBackend(WorkspaceBackendSuite):
    @pytest.mark.skip(reason='the shell filesystem pages a large file at a command per 64 KiB; the live tier runs it')
    async def test_large_file_round_trip(self, backend: WorkspaceBackend) -> None: ...

    @pytest.fixture
    def fresh_backend(self, server: FakeOpenSandbox, working_dir: str) -> Callable[[], WorkspaceBackend]:
        return lambda: OpenSandboxBackend(image='fake-image', connection=server.connection, working_dir=working_dir)

    @pytest.fixture
    def backend(self, fresh_backend: Callable[[], WorkspaceBackend]) -> WorkspaceBackend:
        return fresh_backend()

    @pytest.fixture
    def attach_backend(self, server: FakeOpenSandbox, working_dir: str) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: OpenSandboxBackend(ref=ref, connection=server.connection, working_dir=working_dir)

    @pytest.fixture
    def destroy_environment(self, server: FakeOpenSandbox) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await OpenSandbox[None](connection=server.connection).destroy(backend.ref)

        return destroy


@pytest.mark.opensandbox_live
# The live tier runs without coverage; in CI only its gate fixtures run, to skip it.
class TestLiveOpenSandboxBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    # The suite never destroys what `fresh_backend` creates and uses it for its destructive rules too, so each
    # sandbox is destroyed here, and every backend's transport closed.
    @pytest.fixture
    async def fresh_backend(self) -> AsyncIterator[Callable[[], WorkspaceBackend]]:
        capability = live_capability()
        created: list[OpenSandboxBackend] = []

        def create() -> WorkspaceBackend:
            backend = OpenSandboxBackend(
                image=LIVE_IMAGE,
                connection=capability.connection,
                resource=capability.resource,
                sandbox_timeout=capability.sandbox_timeout,
            )
            created.append(backend)
            return backend

        yield create
        for backend in created:
            try:
                if backend.ref is not None:
                    await capability.destroy(backend.ref)
            finally:
                await backend.aclose()

    # Class-scoped so the rules share one sandbox instead of creating one each; the destructive rules use
    # `fresh_backend`.
    @pytest.fixture(scope='class')
    @classmethod
    async def backend(cls) -> AsyncIterator[OpenSandboxBackend]:
        capability = live_capability()
        backend = OpenSandboxBackend(
            image=LIVE_IMAGE,
            connection=capability.connection,
            resource=capability.resource,
            sandbox_timeout=capability.sandbox_timeout,
        )
        try:
            await backend.get_sandbox()
            yield backend
        finally:
            try:
                if backend.ref is not None:
                    await capability.destroy(backend.ref)
            finally:
                await backend.aclose()

    @pytest.fixture
    async def attach_backend(self) -> AsyncIterator[Callable[[WorkspaceRef], WorkspaceBackend]]:
        attached: list[OpenSandboxBackend] = []

        def attach(ref: WorkspaceRef) -> WorkspaceBackend:
            backend = live_capability().backend(ref)
            attached.append(backend)
            return backend

        yield attach
        for backend in attached:
            await backend.aclose()

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await live_capability().destroy(backend.ref)

        return destroy
