"""Pydantic AI's workspace backend conformance suite, run against `SmolSandboxBackend`.

`TestFakeSmolSandboxBackend` runs everywhere smolmachines installs, over the fake SDK, where
commands run as host processes in a temporary directory. `TestLiveSmolSandboxBackend` runs the
same rules against real VMs and is gated by `PYDANTIC_AI_HARNESS_SMOL_LIVE=1`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.smol_sandbox import SmolSandbox, SmolSandboxBackend

from .fake_smol import FakeSmol


async def _destroy(backend: WorkspaceBackend) -> None:
    assert backend.ref is not None
    await SmolSandbox[None]().destroy(backend.ref)


class TestFakeSmolSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def working_dir(self, fake_smol: FakeSmol, tmp_path: Path) -> str:
        # Resolved so the directory `pwd -P` reports matches it on hosts whose temp directory is a symlink.
        return str(tmp_path.resolve() / 'workspace')

    @pytest.fixture
    def backend(self, working_dir: str) -> SmolSandboxBackend:
        return SmolSandboxBackend(working_dir=working_dir)

    @pytest.mark.skip(reason='shell paging at a host process per 64 KiB; the live tier runs it in a VM')
    async def test_large_file_round_trip(self, backend: WorkspaceBackend) -> None: ...

    @pytest.fixture
    def can_detect_exit_with_inherited_output_pipes(self) -> bool:
        # The fake's host processes report exit only once their output pipes close; smolvm reports
        # it on its own, which the live tier checks.
        return False

    @pytest.fixture
    def fresh_backend(self, working_dir: str) -> Callable[[], WorkspaceBackend]:
        return lambda: SmolSandboxBackend(working_dir=working_dir)

    @pytest.fixture
    def attach_backend(self, working_dir: str) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: SmolSandboxBackend(ref=ref, working_dir=working_dir)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _destroy


def _delete_now(ref: WorkspaceRef | None) -> None:  # pragma: no cover - live tier
    # Blocking, because a class-scoped fixture outlives each test's event loop.
    import smol

    if ref is not None:
        try:
            smol.Machine.connect(ref.id, smol.ConnectOptions(target='local')).delete()
        except smol.SmolError as error:
            if error.code != 'NOT_FOUND':
                raise


# The live tier runs without coverage; in CI only its gate runs, to skip it.
@pytest.mark.smol_live
class TestLiveSmolSandboxBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    # Class-scoped so the rules share one VM instead of booting one each.
    @pytest.fixture(scope='class')
    @classmethod
    def backend(cls) -> Iterator[SmolSandboxBackend]:
        backend = SmolSandboxBackend()
        yield backend
        _delete_now(backend.ref)

    @pytest.fixture
    def fresh_backend(self) -> Iterator[Callable[[], WorkspaceBackend]]:
        created: list[SmolSandboxBackend] = []

        def factory() -> SmolSandboxBackend:
            backend = SmolSandboxBackend()
            created.append(backend)
            return backend

        yield factory
        for backend in created:
            _delete_now(backend.ref)

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: SmolSandboxBackend(ref=ref)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _destroy
