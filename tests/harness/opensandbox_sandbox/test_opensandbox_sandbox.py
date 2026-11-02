"""Tests for the `OpenSandbox` capability, through agent runs against the fake server."""

from __future__ import annotations

import importlib.util
import inspect
import os
from collections.abc import AsyncIterator
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from opensandbox.config import ConnectionConfig

import pydantic_ai_harness.opensandbox_sandbox as opensandbox_package
from pydantic_ai import Agent, RunContext, models
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import Workspace, WorkspaceRef
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox, OpenSandboxBackend

from .._docs_examples import python_blocks
from .fake_opensandbox import FakeOpenSandbox


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


def _backend(workspace: Workspace) -> OpenSandboxBackend:
    backend = workspace.backend
    assert isinstance(backend, OpenSandboxBackend)
    return backend


class TestConfiguration:
    def test_agent_spec_schema_publishes_the_spec_expressible_fields(self) -> None:
        definitions = AgentSpec.model_json_schema_with_capabilities([OpenSandbox])['$defs']
        assert sorted(definitions['spec_params_OpenSandbox']['properties']) == [
            'allow_internet_access',
            'description',
            'env',
            'id',
            'image',
            'resource',
            'sandbox_timeout',
            'working_dir',
        ]

    def test_from_spec_forwards_the_spec_fields(self) -> None:
        fields: dict[str, Any] = {
            'id': 'sandbox',
            'description': 'self-hosted workspace',
            'image': 'python:3.13-slim',
            'resource': {'cpu': '500m', 'memory': '1Gi'},
            'allow_internet_access': False,
            'sandbox_timeout': 60,
            'working_dir': '/project',
            'env': {'CONFIGURED': 'value'},
        }
        assert OpenSandbox[None].from_spec(**fields) == OpenSandbox(**fields)

    def test_repr_does_not_expose_the_api_key(self) -> None:
        capability = OpenSandbox[None](connection=ConnectionConfig(api_key='secret-server-key'))
        assert 'secret-server-key' not in repr(capability)

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

    async def test_backend_attaches_lazily_with_the_capabilitys_settings(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        created = OpenSandboxBackend(image='fake-image', connection=server.connection, working_dir=working_dir)
        await created.run(['true'])
        assert created.ref is not None
        server.requests.clear()
        backend = OpenSandbox[None](
            connection=server.connection, sandbox_timeout=60, working_dir=working_dir, env={'CONFIGURED': 'value'}
        ).backend(created.ref)
        assert server.requests == []
        before = datetime.now(UTC) + timedelta(seconds=60)
        result = await backend.run(['sh', '-c', 'pwd; printf %s "$CONFIGURED"'])
        assert result.stdout == f'{working_dir}\nvalue'
        [(sandbox_id, expires_at)] = server.renewals
        assert sandbox_id == created.ref.id
        assert before <= datetime.fromisoformat(expires_at) <= datetime.now(UTC) + timedelta(seconds=60)

    def test_without_an_image_only_refs_are_supplied(self, server: FakeOpenSandbox) -> None:
        capability = OpenSandbox[None](connection=server.connection)
        assert capability.get_workspace(_ctx(), ref=None) is None
        ref = WorkspaceRef(provider='opensandbox', id='x')
        backend = capability.get_workspace(_ctx(), ref=ref)
        assert isinstance(backend, OpenSandboxBackend) and backend.ref == ref


class TestDocs:
    @pytest.mark.skipif(importlib.util.find_spec('json_repair') is None, reason='coder extra not installed')
    @pytest.mark.parametrize('failure', [None, 'setup', 'agent'], ids=['success', 'setup-failure', 'agent-failure'])
    async def test_preparing_a_sandbox_cleans_up_on_success_and_failure(
        self,
        server: FakeOpenSandbox,
        owned_connection: ConnectionConfig,
        working_dir: str,
        monkeypatch: pytest.MonkeyPatch,
        failure: str | None,
    ) -> None:
        # A local pip stub keeps the unchanged example's installation step offline.
        bindir = Path(working_dir, 'bin')
        bindir.mkdir()
        pip = bindir / 'pip'
        pip.write_text(f'#!/bin/sh\nexit {1 if failure == "setup" else 0}\n', encoding='utf-8')
        pip.chmod(0o755)

        def backend(*, image: str) -> OpenSandboxBackend:
            return OpenSandboxBackend(
                image=image,
                connection=owned_connection,
                working_dir=working_dir,
                env={'PATH': f'{bindir}:{os.environ["PATH"]}'},
            )

        def capability(*, image: str | None = None) -> OpenSandbox[None]:
            return OpenSandbox(image=image, connection=owned_connection)

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            if failure == 'agent':
                raise RuntimeError('agent failed')
            yield 'done'

        def infer_model(*args: object, **kwargs: object) -> FunctionModel:
            return FunctionModel(stream_function=stream)

        monkeypatch.setattr(opensandbox_package, 'OpenSandboxBackend', backend)
        monkeypatch.setattr(opensandbox_package, 'OpenSandbox', capability)
        monkeypatch.setattr(models, 'infer_model', infer_model)
        [example] = [
            block for block in python_blocks('docs/harness/opensandbox-sandbox.md') if 'async def main(' in block.source
        ]
        namespace: dict[str, object] = {'__name__': 'example'}
        exec(compile(example.source, str(example.path), 'exec'), namespace)
        main = namespace['main']
        assert callable(main)
        expected = pytest.raises(RuntimeError, match='pip install failed' if failure == 'setup' else 'agent failed')
        with expected if failure else nullcontext():
            result = main()
            assert inspect.isawaitable(result)
            await result
        assert server.closed_transports == server.opened_transports > 0
        assert server.sandboxes == {}


class TestRuns:
    async def test_disabling_internet_requests_deny_all_on_creation(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        capability = OpenSandbox[None](
            image='fake-image', connection=server.connection, working_dir=working_dir, allow_internet_access=False
        )
        await _agent(capability).run('where are you?')
        [sandbox] = server.sandboxes.values()
        assert sandbox.network_policy == {'defaultAction': 'deny'}

    @pytest.mark.parametrize('use_agent', [False, True], ids=['backend', 'agent'])
    async def test_disabling_internet_does_not_change_an_existing_sandbox(
        self, server: FakeOpenSandbox, working_dir: str, use_agent: bool
    ) -> None:
        created = OpenSandboxBackend(image='fake-image', connection=server.connection, working_dir=working_dir)
        try:
            sandbox = await created.get_sandbox()
        finally:
            await created.aclose()
        capability = OpenSandbox[None](
            connection=server.connection, working_dir=working_dir, allow_internet_access=False
        )
        if use_agent:
            await _agent(capability).run('where are you?', workspace=created.ref)
        else:
            assert created.ref is not None
            backend = capability.backend(created.ref)
            try:
                await backend.working_dir()
            finally:
                await backend.aclose()
        assert [(s.id, s.network_policy) for s in server.sandboxes.values()] == [(sandbox.id, None)]

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
