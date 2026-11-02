"""Pydantic AI's workspace backend conformance suite, run against `AgentCoreSandboxBackend`.

`TestFakeAgentCoreSandboxBackend` runs everywhere, over a fake `bedrock-agentcore` client whose
sessions run commands and code on the host. `TestLiveAgentCoreSandboxBackend` runs the same rules
against a real Code Interpreter session, gated in `conftest.py`. Run it locally with
`PYDANTIC_AI_HARNESS_AGENTCORE_LIVE=1 uv run pytest -m agentcore_live tests/harness/agentcore_sandbox/`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.agentcore_sandbox import AgentCoreSandbox, AgentCoreSandboxBackend

from .conftest import LIVE_REGION
from .fake_agentcore import FakeAgentCore


async def _stop(backend: WorkspaceBackend) -> None:
    assert isinstance(backend, AgentCoreSandboxBackend) and backend.ref is not None
    await AgentCoreSandbox[None](region_name=LIVE_REGION).destroy(backend.ref)


class TestFakeAgentCoreSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, fake_agentcore: FakeAgentCore) -> AgentCoreSandboxBackend:
        return AgentCoreSandboxBackend()

    @pytest.fixture
    def fresh_backend(self, fake_agentcore: FakeAgentCore) -> Callable[[], WorkspaceBackend]:
        return AgentCoreSandboxBackend

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: AgentCoreSandboxBackend(ref=ref)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _stop


@pytest.mark.agentcore_live
# The live tier runs without coverage; in CI only its gate fixtures run, to skip it.
class TestLiveAgentCoreSandboxBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    # Class-scoped so the rules share one session instead of starting one each.
    @pytest.fixture(scope='class')
    @classmethod
    async def backend(cls) -> AsyncIterator[AgentCoreSandboxBackend]:
        backend = AgentCoreSandboxBackend(region_name=LIVE_REGION)
        try:
            await backend.working_dir()
            yield backend
        finally:
            if backend.ref is not None:
                await _stop(backend)

    # The suite never stops what `fresh_backend` starts, so each session is stopped here.
    @pytest.fixture
    async def fresh_backend(self) -> AsyncIterator[Callable[[], WorkspaceBackend]]:
        created: list[AgentCoreSandboxBackend] = []

        def create() -> WorkspaceBackend:
            backend = AgentCoreSandboxBackend(region_name=LIVE_REGION)
            created.append(backend)
            return backend

        yield create
        for backend in created:
            if backend.ref is not None:
                await _stop(backend)

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: AgentCoreSandboxBackend(ref=ref, region_name=LIVE_REGION)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _stop
