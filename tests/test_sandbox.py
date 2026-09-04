"""Tests for the sandbox facade and its lazy backend contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec._sandbox import guard_workflow_sandbox
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.sandboxes import Sandbox, SandboxBackend, SandboxRef, UnavailableSandbox

from .sandbox_fakes import (
    ConnectOnlySandboxCapability,
    DecliningSandboxCapability,
    FakeSandbox,
    FakeSandboxResult,
    SandboxCapability,
)

pytestmark = pytest.mark.anyio


def _tool_call_model(tool_name: str = 'probe') -> FunctionModel:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, {})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(model)


async def test_resolve_normalizes_paths_against_the_working_directory() -> None:
    sandbox = Sandbox(FakeSandbox('resolve'))

    assert await sandbox.resolve('sub/../notes.txt') == '/workspace/notes.txt'
    assert await sandbox.resolve('/abs/./x') == '/abs/x'
    assert await sandbox.resolve('x', base='/elsewhere') == '/elsewhere/x'


async def test_resolve_rejects_a_relative_base() -> None:
    with pytest.raises(ValueError, match="base must be an absolute path, got 'relative'"):
        await Sandbox(FakeSandbox('resolve')).resolve('file.txt', base='relative')


async def test_flat_file_operations_use_the_backend_filesystem() -> None:
    backend = FakeSandbox('files', {'/workspace/data.txt': b'hello'})
    sandbox = Sandbox(backend)

    assert await sandbox.read_bytes('data.txt') == b'hello'
    await sandbox.write_text('data.txt', 'updated')

    assert backend.fs.files['/workspace/data.txt'] == b'updated'


async def test_run_only_backend_supports_bounded_reads_through_shell() -> None:
    inner = FakeSandbox('run-only', {'/workspace/data.txt': b'one\ntwo\nthree\n'})

    class RunOnlyBackend:
        @property
        def ref(self) -> SandboxRef | None:
            return inner.ref

        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeSandboxResult:
            return await inner.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

        async def working_dir(self) -> str:
            return await inner.working_dir()

    sandbox = Sandbox(RunOnlyBackend())
    window = await sandbox.read_file('data.txt', limit=2)

    assert window.lines == ('one', 'two')
    with pytest.raises(NotImplementedError, match='SupportsFilesystem'):
        await sandbox.read_file('data.txt')


def test_backend_construction_does_no_io() -> None:
    backend = FakeSandbox('lazy')

    assert backend.ref is None
    assert backend.create_calls == 0
    assert backend.attach_calls == 0


async def test_first_operation_creates_the_environment_once() -> None:
    backend = FakeSandbox('fresh')

    await backend.run(['true'])
    await backend.working_dir()

    assert backend.create_calls == 1
    assert backend.attach_calls == 0
    assert backend.ref is not None


async def test_concurrent_first_operations_create_one_environment() -> None:
    backend = FakeSandbox('concurrent')

    async with anyio.create_task_group() as tg:
        tg.start_soon(backend.run, ['true'])
        tg.start_soon(backend.working_dir)

    assert backend.create_calls == 1
    assert backend.ref is not None


async def test_create_backend_ref_is_set_after_its_first_operation() -> None:
    backend = FakeSandbox('identity')
    assert backend.ref is None

    await backend.run(['true'])

    assert backend.ref == SandboxRef(sandbox_id='fake-identity')


async def test_two_capabilities_cannot_supply_the_sandbox() -> None:
    class FirstSandboxCapability(AbstractCapability[Any]):
        def get_sandbox(self, ctx: RunContext[Any], *, ref: SandboxRef | None) -> SandboxBackend:
            return FakeSandbox('first')

    class SecondSandboxCapability(AbstractCapability[Any]):
        def get_sandbox(self, ctx: RunContext[Any], *, ref: SandboxRef | None) -> SandboxBackend:
            return FakeSandbox('second')

    agent = Agent(_tool_call_model(), capabilities=[FirstSandboxCapability(), SecondSandboxCapability()])

    with pytest.raises(UserError, match=r'FirstSandboxCapability.*SecondSandboxCapability'):
        await agent.run('go')


async def test_declining_capability_leaves_the_run_sandbox_unavailable() -> None:
    capability = DecliningSandboxCapability()
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        assert isinstance(ctx.sandbox.backend, UnavailableSandbox)
        await ctx.sandbox.run(['true'])
        return 'unreachable'  # pragma: no cover

    with pytest.raises(UserError, match='No sandbox is attached'):
        await agent.run('go')
    assert capability.calls == 1


async def test_unrecognized_sandbox_ref_is_rejected() -> None:
    agent = Agent(_tool_call_model(), capabilities=[DecliningSandboxCapability()])

    with pytest.raises(UserError, match="No capability can supply sandbox 'missing'"):
        await agent.run('go', sandbox=SandboxRef(sandbox_id='missing'))


async def test_capability_backend_is_available_without_connecting_during_run_setup() -> None:
    capability = SandboxCapability()
    seen: list[Sandbox] = []
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        seen.append(ctx.sandbox)
        return 'ok'

    await agent.run('go')

    assert seen[0].backend is capability.backend
    assert capability.backend.create_calls == 0


async def test_run_never_cleans_up_the_sandbox() -> None:
    backend = FakeSandbox('persistent')
    agent = Agent(_tool_call_model(), capabilities=[])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.run(['true'])
        return 'ok'

    await agent.run('go', sandbox=backend)

    assert backend.cleanup_calls == []


async def test_failed_run_never_cleans_up_the_sandbox() -> None:
    backend = FakeSandbox('failed')
    agent = Agent(_tool_call_model('explode'))

    @agent.tool
    async def explode(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.run(['true'])
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('go', sandbox=backend)

    assert backend.cleanup_calls == []


async def test_cancelled_run_never_cleans_up_the_sandbox() -> None:
    backend = FakeSandbox('cancelled')
    agent = Agent(_tool_call_model())
    entered = anyio.Event()

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.run(['true'])
        entered.set()
        await anyio.sleep(60)
        return 'unreachable'  # pragma: no cover

    async with anyio.create_task_group() as tg:

        async def run_agent() -> None:
            await agent.run('go', sandbox=backend)

        tg.start_soon(run_agent)
        await entered.wait()
        tg.cancel_scope.cancel()

    assert backend.cleanup_calls == []


async def test_guard_workflow_sandbox_only_rejects_a_live_handle() -> None:
    ref = SandboxRef(sandbox_id='existing')

    assert guard_workflow_sandbox(ref, live_error='live sandbox') is ref
    assert guard_workflow_sandbox(None, live_error='live sandbox') is None
    with pytest.raises(UserError, match='live sandbox'):
        guard_workflow_sandbox(FakeSandbox('live'), live_error='live sandbox')


async def test_capability_can_supply_a_backend_for_an_explicit_ref() -> None:
    capability = ConnectOnlySandboxCapability()
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        return (await ctx.sandbox.run(['true'])).stdout

    result: AgentRunResult[Any] = await agent.run('go', sandbox=SandboxRef(sandbox_id='existing'))

    assert result.output == 'done'
    assert capability.sandbox_ids == ['existing']
