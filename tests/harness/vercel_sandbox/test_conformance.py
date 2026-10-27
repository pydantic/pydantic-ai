"""Pydantic AI's workspace backend conformance suite, run against `VercelSandboxBackend`.

`TestFakeVercelSandboxBackend` runs everywhere, over the fake Vercel SDK, whose processes run on
the host in a temporary directory. `TestLiveVercelSandboxBackend` runs the same rules against a
real sandbox and is gated like `test_vercel_live.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.vercel_sandbox import VercelSandbox, VercelSandboxBackend

from .conftest import LIVE_SANDBOX_TIMEOUT
from .fake_vercel import FakeVercel


def _attach(ref: WorkspaceRef) -> WorkspaceBackend:
    return VercelSandboxBackend(ref=ref)


async def _destroy(backend: WorkspaceBackend) -> None:
    assert isinstance(backend, VercelSandboxBackend)
    sandbox = await backend.get_sandbox()
    await sandbox.destroy()


class TestFakeVercelSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, fake_vercel: FakeVercel, monkeypatch: pytest.MonkeyPatch) -> VercelSandboxBackend:
        # The background-child rule waits out the drain grace; it needs no real seconds here.
        monkeypatch.setattr('pydantic_ai_harness.vercel_sandbox._backend._OUTPUT_DRAIN_GRACE', 0.2)
        return VercelSandboxBackend()

    @pytest.mark.skip(reason='shell paging at a host process per 64 KiB; the live tier runs it in a sandbox')
    async def test_large_file_round_trip(self, backend: WorkspaceBackend) -> None: ...

    @pytest.fixture
    def fresh_backend(self, backend: VercelSandboxBackend) -> Callable[[], WorkspaceBackend]:
        return VercelSandboxBackend

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _destroy


async def _destroy_if_live(name: str) -> None:  # pragma: no cover - live tier runs without coverage
    # Quiet when the destroy rule already deleted it, and closes the client it opens.
    await VercelSandbox().destroy(WorkspaceRef(provider='vercel', id=name))


@pytest.mark.vercel_live
# The live tier runs without coverage; in CI only its gate fixtures run, to skip it.
class TestLiveVercelSandboxBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    # Destructive rules need their own sandbox, not the shared class-scoped one; each is deleted
    # here if a failed rule left it behind.
    @pytest.fixture
    async def destructive_backend(self) -> AsyncIterator[Callable[[], WorkspaceBackend]]:
        created: list[VercelSandboxBackend] = []

        def create() -> WorkspaceBackend:
            backend = VercelSandboxBackend(sandbox_timeout=LIVE_SANDBOX_TIMEOUT)
            created.append(backend)
            return backend

        yield create
        for backend in created:
            try:
                if backend.ref is not None:
                    await _destroy_if_live(backend.ref.id)
            finally:
                await backend.aclose()

    # Class-scoped so the rules share one sandbox instead of starting one each.
    @pytest.fixture(scope='class')
    @classmethod
    async def backend(cls) -> AsyncIterator[VercelSandboxBackend]:
        backend = VercelSandboxBackend(sandbox_timeout=LIVE_SANDBOX_TIMEOUT)
        try:
            await backend.get_sandbox()
            yield backend
        finally:
            try:
                if backend.ref is not None:
                    await _destroy_if_live(backend.ref.id)
            finally:
                await backend.aclose()

    @pytest.fixture
    def can_detect_exit_with_inherited_output_pipes(self) -> bool:
        # Vercel's `Process.wait()` returns only once a background child holding stdout exits (live
        # probe, 2026-10-27). Documented with a redirect workaround.
        return False

    @pytest.fixture
    def filesystem_honors_shell_permissions(self) -> bool:
        # Commands run as uid 1000 but with every capability, CAP_DAC_OVERRIDE included (live probe,
        # 2026-10-27), so `chmod 000` does not stop a read.
        return False

    # The suite drops the backends it attaches; each opened its own SDK client, closed here.
    @pytest.fixture
    async def attach_backend(self) -> AsyncIterator[Callable[[WorkspaceRef], WorkspaceBackend]]:
        attached: list[VercelSandboxBackend] = []

        def attach(ref: WorkspaceRef) -> WorkspaceBackend:
            backend = VercelSandboxBackend(ref=ref)
            attached.append(backend)
            return backend

        yield attach
        for backend in attached:
            await backend.aclose()

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _destroy
