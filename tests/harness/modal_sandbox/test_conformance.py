"""Pydantic AI's workspace backend conformance suite, run against `ModalSandboxBackend`.

`TestFakeModalSandboxBackend` runs everywhere, over the fake Modal SDK in host mode, where
commands and file operations act on a temporary host directory. `TestLiveModalSandboxBackend`
runs the same rules against a real sandbox and is gated like `test_modal_live.py` (see `conftest.py`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.modal_sandbox import ModalSandboxBackend

from .conftest import LIVE_IDLE_TIMEOUT, LIVE_SANDBOX_TIMEOUT, skip_or_fail_live_tier
from .fake_modal import FakeModal


def _attach(ref: WorkspaceRef) -> WorkspaceBackend:
    return ModalSandboxBackend(ref=ref)


async def _terminate(backend: WorkspaceBackend) -> None:
    assert isinstance(backend, ModalSandboxBackend)
    sandbox = await backend.get_sandbox()
    await sandbox.terminate.aio()


class TestFakeModalSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, fake_modal: FakeModal, tmp_path: Path) -> ModalSandboxBackend:
        # Resolved so the working directory `pwd -P` reports matches it on hosts whose temp
        # directory sits behind a symlink.
        fake_modal.host_root = tmp_path.resolve()
        return ModalSandboxBackend()

    @pytest.fixture
    def fresh_backend(self, backend: ModalSandboxBackend) -> Callable[[], WorkspaceBackend]:
        # `backend` points the fake Modal at the host directory; each call starts another sandbox there.
        return ModalSandboxBackend

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _terminate


# The live tier runs without coverage; in CI only its gate fixtures run, to skip it.
@pytest.mark.modal_live
class TestLiveModalSandboxBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    # One event loop for the class: Modal's client keeps a gRPC channel bound to the loop it was
    # first used on. A class-scoped async fixture is what holds that loop open between tests.
    @pytest.fixture(scope='class')
    @classmethod
    def anyio_backend(cls) -> str:
        return 'asyncio'

    # Class-scoped so the rules share one sandbox instead of starting one each.
    @pytest.fixture(scope='class')
    @classmethod
    async def backend(cls) -> AsyncIterator[ModalSandboxBackend]:
        # Class fixtures run before the function-scoped live gate in conftest.
        skip_or_fail_live_tier()
        backend = ModalSandboxBackend(
            image='python:3.12-slim', sandbox_timeout=LIVE_SANDBOX_TIMEOUT, idle_timeout=LIVE_IDLE_TIMEOUT
        )
        await backend.get_sandbox()
        try:
            yield backend
        finally:
            if backend.ref is not None:
                import modal

                sandbox = await modal.Sandbox.from_id.aio(backend.ref.id)
                if await sandbox.poll.aio() is None:
                    await sandbox.terminate.aio()

    # Not `fresh_backend`: that would also enable the concurrent first-use rule, whose sandbox
    # nothing terminates. The destruction rules terminate their own.
    @pytest.fixture
    def destructive_backend(self) -> Callable[[], WorkspaceBackend]:
        return lambda: ModalSandboxBackend(
            image='python:3.12-slim', sandbox_timeout=LIVE_SANDBOX_TIMEOUT, idle_timeout=LIVE_IDLE_TIMEOUT
        )

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _terminate
