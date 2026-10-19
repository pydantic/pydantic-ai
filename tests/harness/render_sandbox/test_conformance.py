"""Pydantic AI's workspace backend conformance suite, run against `RenderSandboxBackend`.

The fake Render SDK in `workspace_helpers.py` runs commands and the shipped file helper as local
subprocesses under a temporary host directory. The same rules run against a real sandbox in
`test_workspace_live.py`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
    from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
    from pydantic_ai_harness.render_sandbox import RenderSandbox, RenderSandboxBackend

    from .workspace_helpers import WorkspaceHarness

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='render not installed'),
    pytest.mark.subprocess(reason='Exercise real POSIX workspace commands and file helper processes', conditional=True),
]


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

    @pytest.fixture
    def can_detect_exit_with_inherited_output_pipes(self) -> bool:
        # The fake runs commands without the remote supervisor, so it waits for pipe EOF;
        # `test_supervisor.py` covers the supervisor returning on the direct child's exit.
        return False
