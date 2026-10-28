"""Public boxd workspace capability tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import anyio
import boxd
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox, BoxdSandboxBackend
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.shell import Shell

from .._tool_calls import call_tools
from .fake_boxd import FakeBoxd, wait_until_gone

pytestmark = pytest.mark.anyio(backends=['asyncio'])


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


async def _touch(ctx: RunContext[None]) -> str:
    await ctx.workspace.write_bytes('note', b'kept')
    return 'ok'


def test_capability_repr_hides_command_environment() -> None:
    assert 'secret-sentinel' not in repr(BoxdSandbox(env={'TOKEN': 'secret-sentinel'}))


def test_root_package_exports_the_capability_and_backend() -> None:
    import pydantic_ai_harness

    assert pydantic_ai_harness.BoxdSandbox is BoxdSandbox
    assert pydantic_ai_harness.BoxdSandboxBackend is BoxdSandboxBackend
    assert {'BoxdSandbox', 'BoxdSandboxBackend'} <= set(pydantic_ai_harness.__all__)


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ({'working_dir': 'repo'}, 'working_dir must be an absolute POSIX path'),
        ({'auto_suspend_timeout': -1}, 'auto_suspend_timeout must be an integer of at least 0'),
        ({'auto_suspend_timeout': True}, 'auto_suspend_timeout must be an integer of at least 0'),
        ({'vcpu': 3}, r'^vcpu must be 1, 2, 4, 8 or None, got 3\.$'),
        ({'vcpu': True}, r'^vcpu must be 1, 2, 4, 8 or None, got True\.$'),
        ({'vcpu': '2'}, r"^vcpu must be 1, 2, 4, 8 or None, got '2'\.$"),
        ({'snapshot': 'golden', 'vcpu': 2}, r'^`snapshot` keeps the size it was saved at, so it cannot be combined'),
        ({'defer_loading': True, 'id': 'machine'}, 'does not support `defer_loading=True`'),
    ],
)
def test_invalid_settings_fail_at_construction(settings: dict[str, Any], message: str) -> None:
    with pytest.raises(UserError, match=message):
        BoxdSandbox(**settings)


@pytest.mark.parametrize(
    ('env', 'error'), [({'PORT': 3000}, TypeError), ({'': 'empty'}, ValueError), ({'A': 'nul\0'}, ValueError)]
)
def test_invalid_env_fails_at_construction(env: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error, match='env'):
        BoxdSandbox(env=env)  # pyright: ignore[reportArgumentType]


async def test_capability_is_lazy_and_forwards_options(host: FakeBoxd, tmp_path: Path) -> None:
    directory = str(tmp_path.resolve() / 'work')
    capability = BoxdSandbox[None](org='acme', vcpu=2, auto_suspend_timeout=300, working_dir=directory, env={'A': 'b'})
    backend = capability.get_workspace(_ctx(), ref=None)
    assert isinstance(backend, BoxdSandboxBackend)
    assert not host.create_names
    assert (await backend.run(['sh', '-c', 'printf %s "$A"'])).stdout == 'b'
    assert host.only.create_kwargs == {
        'org': 'acme',
        'from_snapshot': None,
        'isolated': True,
        'vcpu': 2,
        'auto_suspend_timeout': 300,
    }
    assert await backend.working_dir() == directory


async def test_a_foreign_ref_is_declined() -> None:
    assert BoxdSandbox[None]().get_workspace(_ctx(), ref=WorkspaceRef(provider='daytona', id='x')) is None


async def test_backend_for_a_ref_attaches_with_the_capability_settings(host: FakeBoxd, tmp_path: Path) -> None:
    machine = host.add()
    capability = BoxdSandbox[None](working_dir=str(tmp_path.resolve()), env={'A': 'b'})
    backend = capability.backend(WorkspaceRef(provider='boxd', id=machine.id))
    assert (await backend.run(['sh', '-c', 'printf %s "$A"'])).stdout == 'b'
    assert not host.create_names
    await backend.aclose()


async def test_an_agent_that_never_uses_the_workspace_creates_nothing(fake_boxd: FakeBoxd) -> None:
    result = await Agent(TestModel(custom_output_text='done'), capabilities=[BoxdSandbox()]).run('hello')
    assert result.output == 'done'
    assert not fake_boxd.create_names and not fake_boxd.clients


async def test_the_run_closes_the_client_it_opened_and_the_result_workspace_reopens(host: FakeBoxd) -> None:
    agent = Agent(TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[BoxdSandbox()])
    result = await agent.run('go')
    assert result.workspace.ref == WorkspaceRef(provider='boxd', id=host.only.id)
    # The run keeps one client across its operations and closes it at the end.
    assert [client.closed for client in host.clients] == [True]

    # After the run nothing would close the backend, so each operation closes its own client.
    assert await result.workspace.read_bytes('note') == b'kept'
    assert len(host.clients) > 1 and all(client.closed for client in host.clients)
    assert len(host.create_names) == 1


async def test_a_backend_used_outside_a_run_closes_its_client_after_each_operation(host: FakeBoxd) -> None:
    # What a Temporal activity does: rebuild the backend from the ref, use it, and drop it.
    backend = BoxdSandbox[None]().get_workspace(_ctx(), ref=None)
    assert isinstance(backend, BoxdSandboxBackend)
    await backend.run(['touch', 'note'])
    assert [client.closed for client in host.clients] == [True]

    attached = BoxdSandbox[None]().get_workspace(_ctx(), ref=backend.ref)
    assert isinstance(attached, BoxdSandboxBackend)
    # Concurrent operations share one client, closed once the last of them ends.
    first, second = await asyncio.gather(attached.run(['test', '-f', 'note']), attached.working_dir())
    assert (first.exit_code, second) == (0, str(host.host_root))
    assert [client.closed for client in host.clients] == [True, True]


@pytest.mark.parametrize('supplied', ['client', 'workspace'])
async def test_the_run_leaves_a_supplied_client_or_workspace_open(host: FakeBoxd, supplied: str) -> None:
    capability = BoxdSandbox[None](client=boxd.AsyncBoxd()) if supplied == 'client' else BoxdSandbox[None]()
    workspace = BoxdSandboxBackend() if supplied == 'workspace' else None
    agent = Agent(TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[capability])
    await agent.run('go', workspace=workspace)
    assert [client.closed for client in host.clients] == [False]


async def test_a_child_run_leaves_the_parent_runs_client_open(host: FakeBoxd) -> None:
    child = Agent(TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[BoxdSandbox()])
    closed_after_child: list[list[bool]] = []

    async def delegate(ctx: RunContext[None]) -> str:
        await child.run('go', workspace=ctx.workspace)
        closed_after_child.append([client.closed for client in host.clients])
        return 'ok'

    parent = Agent(
        TestModel(call_tools=['_touch', 'delegate']),
        deps_type=type(None),
        tools=[_touch, delegate],
        capabilities=[BoxdSandbox()],
    )
    await parent.run('go')
    assert closed_after_child == [[False]]
    assert [client.closed for client in host.clients] == [True]


async def test_a_cancelled_run_stops_its_command_and_closes_its_client(host: FakeBoxd, tmp_path: Path) -> None:
    async def long_command(ctx: RunContext[None]) -> str:
        return (await ctx.workspace.run(['sh', '-c', 'echo $$ > pid; exec sleep 30'])).stdout

    agent = Agent(
        TestModel(call_tools=['long_command']), deps_type=type(None), tools=[long_command], capabilities=[BoxdSandbox()]
    )
    task = asyncio.create_task(agent.run('go'))
    with anyio.fail_after(10):
        while not (tmp_path / 'pid').exists() or not (tmp_path / 'pid').read_text():
            await asyncio.sleep(0.01)
    task.cancel()
    with anyio.fail_after(10), pytest.raises(asyncio.CancelledError):
        await task
    assert [client.closed for client in host.clients] == [True]
    await wait_until_gone(tmp_path / 'pid')


async def test_continuing_the_history_attaches_to_the_same_machine(host: FakeBoxd) -> None:
    agent = Agent(TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[BoxdSandbox()])
    first = await agent.run('touch')
    second = await agent.run('continue', message_history=first.all_messages())
    assert len(host.create_names) == 1
    assert second.workspace.ref == first.workspace.ref


async def test_history_in_a_boxd_machine_without_the_capability_names_it() -> None:
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('hi')]),
        ModelResponse(parts=[TextPart('ok')], workspace_ref=WorkspaceRef(provider='boxd', id='machine-1')),
    ]
    agent = Agent(TestModel(), capabilities=[Shell()])
    with pytest.raises(UserError, match=r'workspace `boxd:machine-1`.*\(such as `BoxdSandbox\(\)`\)'):
        await agent.run('again', message_history=history)


async def test_coder_works_in_the_machine(host: FakeBoxd, tmp_path: Path) -> None:
    results = await call_tools(
        [BoxdSandbox(), Coder()],
        [
            ('write_file', {'path': 'hello.sh', 'content': 'echo hi from the machine\n'}),
            ('shell', {'command': 'sh hello.sh'}),
        ],
    )
    assert 'hi from the machine' in results[1]
    assert (tmp_path / 'hello.sh').read_text() == 'echo hi from the machine\n'


async def test_fork_copies_a_machine_and_closes_the_client_it_opened(host: FakeBoxd) -> None:
    source = host.add()
    await BoxdSandbox[None]().fork(WorkspaceRef(provider='boxd', id=source.id))
    assert [client.closed for client in host.clients] == [True]
