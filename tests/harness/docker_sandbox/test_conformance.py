"""Pydantic AI's workspace backend conformance suite, run against `DockerSandboxBackend` over a fake `docker`."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.docker_sandbox import DockerSandbox, DockerSandboxBackend

from ._fake_docker import install_fake_docker

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='`DockerSandboxBackend` runs `docker` as a POSIX subprocess')


class TestFakeDocker(WorkspaceBackendSuite):
    """Runs against a fake `docker` whose containers are directories on this machine."""

    @pytest.fixture
    def working_dir(self, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> str:
        install_fake_docker(tmp_path_factory.mktemp('fake-docker'), monkeypatch)
        return str(tmp_path_factory.mktemp('container'))

    @pytest.mark.skip(reason='the same shell paging as `TestRunOnlyWorkspaceBackend`, at an exec per 64 KiB')
    async def test_large_file_round_trip(self, backend: WorkspaceBackend) -> None: ...

    @pytest.fixture
    def can_detect_exit_with_inherited_output_pipes(self) -> bool:
        # A real `docker exec` waits until every copy of the command's output is closed; the fake runs
        # locally, so it would pass where a real container hangs.
        return False

    @pytest.fixture
    def backend(self, working_dir: str) -> WorkspaceBackend:
        return DockerSandboxBackend('fake-image', working_dir=working_dir)

    @pytest.fixture
    def fresh_backend(self, working_dir: str) -> Callable[[], WorkspaceBackend]:
        return lambda: DockerSandboxBackend('fake-image', working_dir=working_dir)

    @pytest.fixture
    def attach_backend(self, working_dir: str) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: DockerSandboxBackend(ref=ref, working_dir=working_dir)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await DockerSandbox[None]('fake-image').destroy(backend.ref)

        return destroy
