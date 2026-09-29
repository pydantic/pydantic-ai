"""Pydantic AI's workspace backend conformance suite, run against real containers.

Opt in with `PYDANTIC_AI_HARNESS_DOCKER_LIVE=1` on a machine with a Docker daemon.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.docker_sandbox import DockerSandbox, DockerSandboxBackend

LIVE_IMAGE = 'python:3.13-slim'


@pytest.mark.skipif(
    os.getenv('PYDANTIC_AI_HARNESS_DOCKER_LIVE') != '1', reason='set PYDANTIC_AI_HARNESS_DOCKER_LIVE=1 to use Docker'
)
class TestLiveDocker(WorkspaceBackendSuite):
    """Runs against real containers; each test's containers are removed afterwards."""

    @pytest.fixture
    async def created(self) -> AsyncIterator[list[WorkspaceBackend]]:
        backends: list[WorkspaceBackend] = []
        yield backends
        for backend in backends:
            if backend.ref is not None:
                await DockerSandbox[None](LIVE_IMAGE).destroy(backend.ref)

    @pytest.fixture
    def fresh_backend(self, created: list[WorkspaceBackend]) -> Callable[[], WorkspaceBackend]:
        def build() -> WorkspaceBackend:
            backend = DockerSandboxBackend(LIVE_IMAGE)
            created.append(backend)
            return backend

        return build

    @pytest.fixture
    def backend(self, fresh_backend: Callable[[], WorkspaceBackend]) -> WorkspaceBackend:
        return fresh_backend()

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: DockerSandboxBackend(ref=ref)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await DockerSandbox[None](LIVE_IMAGE).destroy(backend.ref)

        return destroy
