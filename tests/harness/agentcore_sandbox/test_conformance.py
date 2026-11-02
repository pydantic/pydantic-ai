"""Pydantic AI's workspace backend conformance suite, run live against `AgentCoreSandboxBackend`.

Needs AWS credentials with `bedrock-agentcore:*CodeInterpreter*` access. Opt in with
`PYDANTIC_AI_HARNESS_AGENTCORE_LIVE=1`; the region comes from `AWS_REGION` (default `us-east-1`).
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Iterator

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.agentcore_sandbox import AgentCoreSandboxBackend, destroy_session

pytestmark = pytest.mark.skipif(
    os.environ.get('PYDANTIC_AI_HARNESS_AGENTCORE_LIVE') != '1',
    reason='set PYDANTIC_AI_HARNESS_AGENTCORE_LIVE=1 to run against a real Code Interpreter session',
)

REGION = os.environ.get('AWS_REGION', 'us-east-1')


def _new() -> AgentCoreSandboxBackend:
    return AgentCoreSandboxBackend(region_name=REGION, session_timeout_seconds=900)


def _attach(ref: WorkspaceRef) -> WorkspaceBackend:
    return AgentCoreSandboxBackend(region_name=REGION, ref=ref)


async def _destroy(backend: WorkspaceBackend) -> None:
    assert backend.ref is not None
    await destroy_session(backend.ref, region_name=REGION)


class TestLiveAgentCoreSandboxBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    _created: list[AgentCoreSandboxBackend] = []

    @pytest.fixture(scope='class')
    @classmethod
    def backend(cls) -> Iterator[AgentCoreSandboxBackend]:
        backend = _new()
        cls._created.append(backend)
        yield backend
        import boto3

        client = boto3.client('bedrock-agentcore', region_name=REGION)
        for created in cls._created:
            if created.ref is not None:
                identifier, _, session_id = created.ref.id.rpartition('/')
                try:
                    client.stop_code_interpreter_session(codeInterpreterIdentifier=identifier, sessionId=session_id)
                except Exception:
                    pass

    @pytest.fixture
    def fresh_backend(self) -> Callable[[], WorkspaceBackend]:
        def build() -> WorkspaceBackend:
            backend = _new()
            self._created.append(backend)
            return backend

        return build

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _destroy
