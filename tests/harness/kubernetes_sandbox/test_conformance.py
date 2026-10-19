"""Pydantic AI's workspace backend conformance suite, run against `KubernetesSandboxBackend` over `fake_sandboxd`."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox, KubernetesSandboxBackend

from .fake_sandboxd import FakeCluster

# `grpc.aio` and the SDK run on asyncio only; Trio raises `UserError` (see `test_kubernetes_sandbox.py`).
pytestmark = pytest.mark.parametrize('anyio_backend', ['asyncio'])


class TestKubernetesSandboxConformance(WorkspaceBackendSuite):
    @pytest.fixture
    async def created(self, cluster: FakeCluster) -> AsyncIterator[list[KubernetesSandboxBackend]]:
        backends: list[KubernetesSandboxBackend] = []
        yield backends
        for backend in backends:
            await backend.aclose()

    @pytest.fixture
    def fresh_backend(self, created: list[KubernetesSandboxBackend]) -> Callable[[], WorkspaceBackend]:
        def make() -> WorkspaceBackend:
            backend = KubernetesSandboxBackend(warm_pool='pool', namespace='agents')
            created.append(backend)
            return backend

        return make

    @pytest.fixture
    def backend(self, fresh_backend: Callable[[], WorkspaceBackend]) -> WorkspaceBackend:
        return fresh_backend()

    @pytest.fixture
    def attach_backend(self, created: list[KubernetesSandboxBackend]) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        def attach(ref: WorkspaceRef) -> WorkspaceBackend:
            backend = KubernetesSandboxBackend(ref=ref)
            created.append(backend)
            return backend

        return attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await KubernetesSandbox(warm_pool='pool').destroy(backend.ref)

        return destroy
