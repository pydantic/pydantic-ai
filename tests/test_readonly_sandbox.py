"""Tests for `ReadOnlySandbox`, the policy wrapper that blocks execution and file mutation."""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.sandboxes import (
    ReadOnlySandbox,
    Sandbox,
    SandboxBackend,
    SandboxRef,
    SupportsFilesystem,
)

from .sandbox_fakes import FakeSandbox, RecordingSandboxBackend

pytestmark = pytest.mark.anyio


async def test_reads_forward_and_mutations_raise():
    """Reads pass through to the wrapped backend; every mutating or executing operation raises."""
    backend = FakeSandbox('rw', {'/workspace/data.csv': b'a,b\n1,2\n'})
    sandbox = Sandbox(ReadOnlySandbox(backend))

    assert await sandbox.read_text('data.csv') == 'a,b\n1,2\n'
    assert (await sandbox.fs.stat('/workspace/data.csv')).size == 8
    assert [entry.name for entry in await sandbox.fs.list_dir('/workspace')] == ['data.csv']
    assert await sandbox.fs.exists('/workspace/data.csv')

    with pytest.raises(UserError, match='read-only'):
        await sandbox.write_text('data.csv', 'overwritten')
    with pytest.raises(UserError, match='read-only'):
        await sandbox.fs.make_dir('/workspace/new')
    with pytest.raises(UserError, match='read-only'):
        await sandbox.fs.remove('/workspace/data.csv')
    with pytest.raises(UserError, match='read-only'):
        await sandbox.run(['rm', '-rf', '/workspace'])

    assert backend.fs.files == {'/workspace/data.csv': b'a,b\n1,2\n'}
    assert backend.commands == []


async def test_caller_owned_backend_stays_fully_usable_outside_the_wrapper():
    """The caller's backend stays read-write while its model-facing wrapper is read-only."""
    backend = FakeSandbox('rw')
    sandbox = Sandbox(ReadOnlySandbox(backend))

    await backend.fs.write_bytes('/workspace/data.csv', b'a,b\n')
    await backend.run(['touch', 'marker'])

    assert await sandbox.read_text('data.csv') == 'a,b\n'  # the application's write is visible
    assert backend.commands == [['touch', 'marker']]


async def test_identity_and_working_dir_forward():
    """The wrapper keeps the wrapped backend's identity: policy is not part of it."""
    read_only = ReadOnlySandbox(FakeSandbox('rw'))

    assert read_only.sandbox_id == 'fake-rw'
    assert await read_only.working_dir() == '/workspace'


async def _run_with_connected(backend: SandboxBackend) -> None:
    """Run an agent whose tool uses a capability-connected `ReadOnlySandbox` over `backend`."""

    class Connector(AbstractCapability[Any]):
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            return SandboxRef(sandbox_id=backend.sandbox_id)

        def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
            return ReadOnlySandbox(backend)

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('probe', {})])
        return ModelResponse(parts=[TextPart('done')])

    agent: Agent = Agent(FunctionModel(model_func), capabilities=[Connector()])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        return await ctx.sandbox.working_dir()

    await agent.run('go')


async def test_connection_close_forwards_detach_but_blocks_termination():
    class ClosableBackend(FakeSandbox):
        def __init__(self) -> None:
            super().__init__('closable')
            self.close_calls: list[bool] = []

        async def close(self, *, terminate: bool) -> None:
            self.close_calls.append(terminate)

    backend = ClosableBackend()
    await _run_with_connected(backend)
    assert backend.close_calls == [False]

    with pytest.raises(UserError, match='cannot terminate'):
        await ReadOnlySandbox(backend).close(terminate=True)


async def test_connection_close_ignores_unrelated_incompatible_close_method():
    class BackendWithUnrelatedClose(FakeSandbox):
        async def close(self) -> None:
            raise AssertionError('incompatible close must not be called')  # pragma: no cover

    await _run_with_connected(BackendWithUnrelatedClose('unrelated-close'))


async def test_filesystem_support_mirrors_wrapped_backend():
    """The wrapper claims `SupportsFilesystem` only when the wrapped backend does."""
    assert isinstance(ReadOnlySandbox(FakeSandbox('fs')), SupportsFilesystem)

    without_filesystem = ReadOnlySandbox(RecordingSandboxBackend('no-fs'))
    assert not isinstance(without_filesystem, SupportsFilesystem)
    with pytest.raises(NotImplementedError, match='SupportsFilesystem'):
        await Sandbox(without_filesystem).read_text('data.csv')


async def test_windowed_read_through_read_only_never_executes_the_wrapped_backend():
    backend = FakeSandbox('read-only', {'/workspace/lines.txt': b'one\ntwo\nthree\nfour\n'})
    window = await Sandbox(ReadOnlySandbox(backend)).read_file('lines.txt', offset=2, limit=2)

    assert window.lines == ('two', 'three')
    assert window.start_line == 2
    assert window.has_more
    assert window.total_lines == 4
    assert backend.commands == []


async def test_agent_run_surfaces_read_only_reason():
    """A tool writing through `ctx.sandbox` fails the run with the policy reason."""

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('rewrite', {})])

    agent: Agent = Agent(FunctionModel(model_func))

    @agent.tool
    async def rewrite(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.write_text('data.csv', 'overwritten')
        return 'wrote'  # pragma: no cover

    with pytest.raises(UserError, match='read-only'):
        await agent.run('Rewrite the data file.', sandbox=ReadOnlySandbox(FakeSandbox('rw')))
