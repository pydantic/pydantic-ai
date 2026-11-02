"""Pydantic AI's workspace backend conformance suite, run against `VercelSandboxBackend`.

`TestFakeVercelSandboxBackend` runs everywhere, over the fake Vercel SDK, where commands act on a
temporary host directory. `TestLiveVercelSandboxBackend` runs the same rules against a real sandbox
and is gated like `test_vercel_live.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.vercel_sandbox import VercelSandbox, VercelSandboxBackend

from .conftest import LIVE_EXECUTION_TIME_LIMIT
from .fake_vercel import FakeVercel


async def _destroy(backend: WorkspaceBackend) -> None:
    assert isinstance(backend, VercelSandboxBackend) and backend.ref is not None
    await VercelSandbox().destroy(backend.ref)


class TestFakeVercelSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, host: FakeVercel) -> VercelSandboxBackend:
        return VercelSandboxBackend()

    @pytest.fixture
    def fresh_backend(self, host: FakeVercel) -> Callable[[], WorkspaceBackend]:
        return VercelSandboxBackend

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: VercelSandboxBackend(ref=ref)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _destroy


@pytest.mark.vercel_live
# The live tier runs without coverage; in CI only its gate fixtures run, to skip it.
class TestLiveVercelSandboxBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    # Class-scoped so the rules share one sandbox instead of creating one each; the suite runs
    # its destroy rule last.
    @pytest.fixture(scope='class')
    @classmethod
    async def backend(cls) -> AsyncIterator[VercelSandboxBackend]:
        backend = VercelSandboxBackend(execution_time_limit=LIVE_EXECUTION_TIME_LIMIT)
        try:
            await backend.get_sandbox()
            yield backend
        finally:
            try:
                if backend.ref is not None:
                    await VercelSandbox().destroy(backend.ref)
            finally:
                await backend.aclose()

    # The suite never destroys what `fresh_backend` creates, and uses it for its destructive rules too,
    # so each sandbox is destroyed here.
    @pytest.fixture
    async def fresh_backend(self) -> AsyncIterator[Callable[[], WorkspaceBackend]]:
        created: list[VercelSandboxBackend] = []

        def create() -> WorkspaceBackend:
            backend = VercelSandboxBackend(execution_time_limit=LIVE_EXECUTION_TIME_LIMIT)
            created.append(backend)
            return backend

        yield create
        for backend in created:
            try:
                if backend.ref is not None:
                    await VercelSandbox().destroy(backend.ref)
            finally:
                await backend.aclose()

    # The suite drops the backend it attaches; each one opened its own SDK client, which is closed here.
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
