"""Pydantic AI's workspace backend conformance suite, run against `SSHWorkspaceBackend` over a fake `ssh`."""

from __future__ import annotations

import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path

import anyio.to_thread
import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.ssh_workspace import SSHWorkspaceBackend

from .._fake_remote_tools import install_fake_remote_tools

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='the fake `ssh` runs the remote command with POSIX sh')


class TestSSHWorkspaceBackend(WorkspaceBackendSuite):
    """Runs against a fake `ssh` that executes the remote command on this machine."""

    @pytest.fixture(autouse=True)
    def fake_ssh(self, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_remote_tools(tmp_path_factory.mktemp('fake-remote'), monkeypatch)

    @pytest.mark.skip(reason='the same shell paging as `TestRunOnlyWorkspaceBackend`, at a connection per 64 KiB')
    async def test_large_file_round_trip(self, backend: WorkspaceBackend) -> None: ...

    @pytest.fixture
    def can_detect_exit_with_inherited_output_pipes(self) -> bool:
        # A real `sshd` keeps the session open until every copy of the command's output is closed;
        # the fake `ssh` runs locally, so it would pass where a real host hangs.
        return False

    @staticmethod
    def attach(ref: WorkspaceRef) -> WorkspaceBackend:
        # Every ref here is on the fake host `box`; a real destination such as `ssh://host:2222` has colons of its own.
        return SSHWorkspaceBackend('box', working_dir=ref.id.removeprefix('box:'))

    @pytest.fixture
    def backend(self, tmp_path: Path) -> WorkspaceBackend:
        return SSHWorkspaceBackend('box', working_dir=str(tmp_path))

    @pytest.fixture
    def destructive_backend(self, tmp_path_factory: pytest.TempPathFactory) -> Callable[[], WorkspaceBackend]:
        path = tmp_path_factory.mktemp('fresh-ssh-ws')
        return lambda: self.attach(WorkspaceRef(provider='ssh', id=f'box:{path}'))

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return self.attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await anyio.to_thread.run_sync(shutil.rmtree, backend.ref.id.removeprefix('box:'))

        return destroy
