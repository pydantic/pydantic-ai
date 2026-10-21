"""Pydantic AI's workspace backend conformance suite, run against `BubblewrapWorkspace`.

`TestBubblewrapAroundSSH` runs everywhere, through a fake `bwrap` on a fake SSH host; `TestRealBubblewrap`
runs the same rules in a real sandbox where `bwrap` works (Linux with user namespaces).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import anyio.to_thread
import pytest

from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace, WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.bubblewrap_sandbox import BubblewrapWorkspace
from pydantic_ai_harness.ssh_workspace import SSHWorkspaceBackend

from .._fake_remote_tools import BWRAP_WORKS

# The module, not the class: a `Test*` name imported here would be collected and run a second time.
from ..ssh_workspace import test_conformance as ssh_conformance

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='the fake `ssh` and `bwrap` run commands with POSIX sh')


class TestBubblewrapAroundSSH(ssh_conformance.TestSSHWorkspaceBackend):
    """The same rules hold with every command going through (a fake) `bwrap` on the SSH host."""

    @staticmethod
    def attach(ref: WorkspaceRef) -> WorkspaceBackend:
        return BubblewrapWorkspace(Workspace(ssh_conformance.TestSSHWorkspaceBackend.attach(ref)))

    @pytest.fixture
    def backend(self, tmp_path: Path) -> WorkspaceBackend:
        return BubblewrapWorkspace(Workspace(SSHWorkspaceBackend('box', working_dir=str(tmp_path))))


# The same rules hold with every command in a real bubblewrap sandbox.
@pytest.mark.skipif(not BWRAP_WORKS, reason='needs a working `bwrap` (Linux with user namespaces)')
class TestRealBubblewrap(WorkspaceBackendSuite):  # pragma: no cover - CI hosts may not have bubblewrap
    @pytest.fixture
    def backend(self, tmp_path: Path) -> WorkspaceBackend:
        return BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))

    @pytest.fixture
    def destructive_backend(self) -> Iterator[Callable[[], WorkspaceBackend]]:
        # Not under `/tmp`: the sandbox mounts a private `/tmp`, and the directory `bwrap` makes there for the
        # bind mount outlives the host deleting the real one, so a command inside never sees it go.
        path = Path(tempfile.mkdtemp(prefix='fresh-bwrap-ws-', dir='/var/tmp'))
        yield lambda: BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(path)))
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(ref.id)))

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await anyio.to_thread.run_sync(shutil.rmtree, backend.ref.id)

        return destroy
