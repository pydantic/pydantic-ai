"""Pydantic AI's workspace backend conformance suite, run against `RenderSandboxBackend`.

`TestFakeRenderSandboxBackend` runs everywhere: the fake Render SDK in `workspace_helpers.py` runs every
command under the shipped supervisor in a child interpreter, in a temporary host directory per sandbox,
and the shipped file helper in-process.
`TestLiveRenderSandboxBackend` runs the same rules against a real sandbox, gated in `conftest.py`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
    from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
    from pydantic_ai_harness.render_sandbox import RenderSandbox, RenderSandboxBackend

    from .workspace_helpers import WorkspaceHarness

pytestmark = pytest.mark.skipif(not imports_successful(), reason='render not installed')


@pytest.mark.subprocess(reason='Commands run under the shipped supervisor in a child interpreter, as on Render')
class TestFakeRenderSandboxBackend(WorkspaceBackendSuite):
    # Isolated roots give each fake sandbox its own directory, which termination deletes.
    @pytest.fixture
    def backend(self, isolated_workspace_harness: WorkspaceHarness) -> RenderSandboxBackend:
        return isolated_workspace_harness.backend()

    @pytest.fixture
    def fresh_backend(self, isolated_workspace_harness: WorkspaceHarness) -> Callable[[], WorkspaceBackend]:
        return isolated_workspace_harness.backend

    @pytest.fixture
    def attach_backend(
        self, isolated_workspace_harness: WorkspaceHarness
    ) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return isolated_workspace_harness.backend

    @pytest.fixture
    def destroy_environment(
        self, isolated_workspace_harness: WorkspaceHarness
    ) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await RenderSandbox(client=isolated_workspace_harness.client).destroy(backend.ref)

        return destroy


async def _destroy(backend: WorkspaceBackend) -> None:  # pragma: no cover - live tier only
    assert backend.ref is not None
    await RenderSandbox().destroy(backend.ref)


def _attach(ref: WorkspaceRef) -> WorkspaceBackend:  # pragma: no cover - live tier only
    return RenderSandboxBackend(ref=ref)


# The live tier runs without coverage; in CI only its gate runs, to skip it.
@pytest.mark.render_live
class TestLiveRenderSandboxBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    # Class-scoped so the rules share one sandbox. Each operation opens its own client, so the backend
    # outlives each test's event loop, and the teardown runs `destroy` on a loop of its own.
    @pytest.fixture(scope='class')
    @classmethod
    def backend(cls) -> Iterator[RenderSandboxBackend]:
        backend = RenderSandboxBackend(sandbox_timeout=600)
        yield backend
        if backend.ref is not None:
            asyncio.run(_destroy(backend))

    @pytest.fixture
    def destructive_backend(self) -> Callable[[], WorkspaceBackend]:
        # Destructive rules need their own sandbox, not the shared class-scoped one.
        return lambda: RenderSandboxBackend(sandbox_timeout=600)

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _destroy
