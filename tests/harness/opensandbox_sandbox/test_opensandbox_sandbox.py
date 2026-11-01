"""Tests for the `OpenSandbox` capability, through agent runs against the fake server."""

from __future__ import annotations

from typing import Any

import pytest
from opensandbox.config import ConnectionConfig

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox, OpenSandboxBackend

from .fake_opensandbox import FakeOpenSandbox

pytestmark = pytest.mark.anyio(backends=['asyncio'])


def _call_pwd_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returned = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    if returned:
        return ModelResponse(parts=[TextPart(str(returned[-1].content))])
    return ModelResponse(parts=[ToolCallPart('pwd', {})])


def _agent(capability: OpenSandbox[None]) -> Agent[None, str]:
    agent = Agent(FunctionModel(_call_pwd_then_answer), deps_type=type(None), capabilities=[capability])

    @agent.tool
    async def pwd(ctx: RunContext[None]) -> str:
        return (await ctx.workspace.run(['pwd'])).stdout

    return agent


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def _backend(workspace: Any) -> OpenSandboxBackend:
    backend = workspace.backend
    assert isinstance(backend, OpenSandboxBackend)
    return backend


class TestConfiguration:
    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'defer_loading': True}, 'does not support `defer_loading=True`'),
            ({'working_dir': 'relative'}, 'working_dir must be an absolute POSIX path'),
            ({'sandbox_timeout': 0}, 'sandbox_timeout must be an integer of at least 1'),
            ({'env': {'A=B': 'x'}}, 'env keys must be non-empty without `=`'),
        ],
    )
    def test_bad_values_fail_where_they_are_written(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises((UserError, ValueError), match=message):
            OpenSandbox[None](image='fake-image', **kwargs)

    async def test_a_ref_from_another_provider_is_left_to_other_capabilities(self, server: FakeOpenSandbox) -> None:
        capability = OpenSandbox[None](image='fake-image', connection=server.connection)
        assert capability.get_workspace(_ctx(), ref=WorkspaceRef(provider='docker', id='x')) is None
        with pytest.raises(ValueError, match="expected 'opensandbox'"):
            await capability.destroy(WorkspaceRef(provider='docker', id='x'))

    def test_backend_attaches_lazily_with_the_capabilitys_settings(self, server: FakeOpenSandbox) -> None:
        ref = WorkspaceRef(provider='opensandbox', id='x')
        backend = OpenSandbox[None](connection=server.connection, sandbox_timeout=60).backend(ref)
        assert (backend.ref, server.requests) == (ref, [])

    def test_without_an_image_only_refs_are_supplied(self, server: FakeOpenSandbox) -> None:
        capability = OpenSandbox[None](connection=server.connection)
        assert capability.get_workspace(_ctx(), ref=None) is None
        ref = WorkspaceRef(provider='opensandbox', id='x')
        backend = capability.get_workspace(_ctx(), ref=ref)
        assert isinstance(backend, OpenSandboxBackend) and backend.ref == ref


class TestRuns:
    async def test_a_run_creates_a_sandbox_closes_its_transport_and_keeps_the_sandbox(
        self, server: FakeOpenSandbox, owned_connection: ConnectionConfig, working_dir: str
    ) -> None:
        capability = OpenSandbox[None](image='fake-image', connection=owned_connection, working_dir=working_dir)
        result = await _agent(capability).run('where are you?')
        assert result.output == f'{working_dir}\n'
        assert server.closed_transports == server.opened_transports > 0
        backend = _backend(result.workspace)
        assert backend.ref is not None and list(server.sandboxes) == [backend.ref.id]

    async def test_a_handed_on_workspace_reaches_the_same_sandbox_and_is_closed_again(
        self, server: FakeOpenSandbox, owned_connection: ConnectionConfig, working_dir: str
    ) -> None:
        capability = OpenSandbox[None](image='fake-image', connection=owned_connection, working_dir=working_dir)
        agent = _agent(capability)
        first = await agent.run('where are you?')
        second = await agent.run('and now?', workspace=first.workspace)
        assert second.output == first.output
        assert len(server.sandboxes) == 1
        # The second run does not own it, so the backend closed its transport after each operation.
        assert server.closed_transports == server.opened_transports

    async def test_a_run_by_ref_attaches_instead_of_creating(self, server: FakeOpenSandbox, working_dir: str) -> None:
        capability = OpenSandbox[None](image='fake-image', connection=server.connection, working_dir=working_dir)
        agent = _agent(capability)
        first = await agent.run('where are you?')
        ref = _backend(first.workspace).ref
        assert ref is not None
        await agent.run('and now?', workspace=ref)
        assert list(server.sandboxes) == [ref.id]

    async def test_destroy_removes_the_sandbox(self, server: FakeOpenSandbox, working_dir: str) -> None:
        capability = OpenSandbox[None](image='fake-image', connection=server.connection, working_dir=working_dir)
        result = await _agent(capability).run('where are you?')
        ref = _backend(result.workspace).ref
        assert ref is not None
        await capability.destroy(ref)
        assert server.sandboxes == {}
