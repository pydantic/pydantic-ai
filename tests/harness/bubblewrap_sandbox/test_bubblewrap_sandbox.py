"""Tests for `BubblewrapWorkspace` and the `BubblewrapSandbox` capability.

Most use a fake `bwrap` that records its arguments, so they run anywhere; the last class needs a working `bwrap`.
"""

from __future__ import annotations

import os
from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import (
    LocalWorkspaceBackend,
    ReadOnlyWorkspace,
    Workspace,
    WorkspaceReadOnlyError,
    WorkspaceRef,
    WorkspaceUnavailableError,
)
from pydantic_ai.workspaces.workspace import workspace_layers
from pydantic_ai_harness.bubblewrap_sandbox import BubblewrapSandbox, BubblewrapWorkspace
from pydantic_ai_harness.ssh_workspace import SSHWorkspace, SSHWorkspaceBackend

from .._fake_remote_tools import BWRAP_WORKS, FakeRemoteTools, install_fake_remote_tools

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(os.name != 'posix', reason='the wrapped workspaces run POSIX subprocesses'),
]


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeRemoteTools:
    return install_fake_remote_tools(tmp_path, monkeypatch)


def _sandbox_args(working_dir: str, *, network: bool = False) -> str:
    return ' '.join(
        [
            '--die-with-parent --new-session --unshare-user-try --unshare-ipc --unshare-uts --unshare-cgroup-try',
            *([] if network else ['--unshare-net']),
            '--ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --tmpfs /run',
            *(
                [
                    '--ro-bind-try /run/systemd/resolve /run/systemd/resolve',
                    '--ro-bind-try /run/NetworkManager /run/NetworkManager',
                    '--ro-bind-try /run/resolvconf /run/resolvconf',
                ]
                if network
                else []
            ),
            f'--bind {working_dir} {working_dir}',
        ]
    )


async def test_commands_run_in_bwrap_on_the_wrapped_host(tools: FakeRemoteTools, tmp_path: Path) -> None:
    workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)), bwrap_args=['--tmpfs', '/secrets'])
    working_dir = await workspace.working_dir()
    assert tools.bwrap_calls == []

    argv = await workspace.run(['printf', '%s', 'a b'], env={'GREETING': 'hi'})
    shell = await workspace.run('printf "%s" "$GREETING"', shell=True, env={'GREETING': 'hi'})

    assert (argv.exit_code, argv.stdout, shell.stdout) == (0, 'a b', 'hi')
    assert tools.bwrap_calls == [
        f'{_sandbox_args(working_dir)} --tmpfs /secrets --chdir {working_dir} --setenv GREETING hi -- '
        'sh -c exec "$@" sh printf %s a b',
        f'{_sandbox_args(working_dir)} --tmpfs /secrets --chdir {working_dir} --setenv GREETING hi -- '
        'sh -c printf "%s" "$GREETING"',
    ]


async def test_the_call_env_reaches_only_the_sandboxed_command(tools: FakeRemoteTools, tmp_path: Path) -> None:
    """A model-controlled `PATH` must not pick which `bwrap` runs."""
    impostor = tmp_path / 'impostor'
    impostor.mkdir()
    (impostor / 'bwrap').write_text(f'#!/bin/sh\ntouch {impostor}/escaped\n')
    (impostor / 'bwrap').chmod(0o755)
    workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))

    result = await workspace.run(['sh', '-c', 'printf %s "$PATH"'], env={'PATH': f'{impostor}:{os.environ["PATH"]}'})

    assert result.stdout.startswith(str(impostor))
    assert not (impostor / 'escaped').exists()
    assert len(tools.bwrap_calls) == 1


async def test_network_is_shared_only_when_asked(tools: FakeRemoteTools, tmp_path: Path) -> None:
    workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)), network=True)
    working_dir = await workspace.working_dir()

    await workspace.run(['true'])

    assert tools.bwrap_calls[0].startswith(_sandbox_args(working_dir, network=True))


async def test_file_methods_go_to_the_wrapped_workspace(tools: FakeRemoteTools, tmp_path: Path) -> None:
    workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))

    await workspace.write_text('notes.txt', 'hello')

    assert await workspace.read_text('notes.txt') == 'hello'
    assert tools.bwrap_calls == []


async def test_a_failing_command_checks_the_sandbox_once(tools: FakeRemoteTools, tmp_path: Path) -> None:
    workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))

    assert (await workspace.run(['pydantic-ai-missing-program'])).exit_code == 127
    assert (await workspace.run(['sh', '-c', 'exit 3'])).exit_code == 3

    calls = tools.bwrap_calls
    assert [call.rsplit(' -- ', 1)[1] for call in calls] == [
        'sh -c exec "$@" sh pydantic-ai-missing-program',
        'true',
        'sh -c exec "$@" sh sh -c exit 3',
    ]


async def test_a_sandbox_that_cannot_start_is_unavailable(tools: FakeRemoteTools, tmp_path: Path) -> None:
    workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)), bwrap_args=['--fake-fail'])

    with pytest.raises(WorkspaceUnavailableError, match=r'bubblewrap could not start a sandbox.*uid map'):
        await workspace.run(['true'])


async def test_invalid_commands_and_arguments_are_rejected(tmp_path: Path) -> None:
    wrapped = Workspace(LocalWorkspaceBackend(tmp_path))
    workspace = BubblewrapWorkspace(wrapped)

    with pytest.raises(TypeError, match='bwrap_args must be a sequence'):
        BubblewrapWorkspace(wrapped, bwrap_args='--share-net')
    with pytest.raises(TypeError, match='requires shell=True'):
        await workspace.run('true')
    with pytest.raises(TypeError, match='cannot be combined with shell=True'):
        await workspace.run(['true'], shell=True)
    with pytest.raises(ValueError, match='must not be empty'):
        await workspace.run([])
    with pytest.raises(ValueError, match='timeout'):
        await workspace.run(['true'], timeout=0)


async def test_read_only_inside_the_sandbox_still_refuses_commands(tmp_path: Path) -> None:
    workspace = BubblewrapWorkspace(ReadOnlyWorkspace(Workspace(LocalWorkspaceBackend(tmp_path))))

    assert workspace.read_only is True
    with pytest.raises(WorkspaceReadOnlyError):
        await workspace.run(['true'])


async def test_bubblewrap_around_ssh_sandboxes_commands_on_the_remote_host(
    tools: FakeRemoteTools, tmp_path: Path
) -> None:
    backend = SSHWorkspaceBackend('box', working_dir=str(tmp_path))
    workspace = BubblewrapWorkspace(Workspace(backend))

    result = await workspace.run(['sh', '-c', 'printf %s "$PWD"'])

    working_dir = await backend.working_dir()
    assert result.stdout == working_dir
    # The fake `ssh` ran `bwrap` "remotely", and `bwrap` bound the remote working directory.
    assert tools.bwrap_calls[0].startswith(_sandbox_args(working_dir))
    assert workspace.backend is backend
    assert workspace.ref == WorkspaceRef(provider='ssh', id=f'box:{tmp_path}')
    assert workspace_layers(workspace) == [BubblewrapWorkspace, SSHWorkspaceBackend]


async def test_capability_wraps_the_ssh_workspace(tools: FakeRemoteTools, tmp_path: Path) -> None:
    agent = Agent(
        TestModel(call_tools=['probe']),
        capabilities=[BubblewrapSandbox(SSHWorkspace('box', working_dir=str(tmp_path)), network=True)],
    )

    @agent.tool
    async def probe(ctx: RunContext[object]) -> str:
        return (await ctx.workspace.run(['printf', 'sandboxed'])).stdout

    result = await agent.run('go')

    assert result.output == '{"probe":"sandboxed"}'
    assert workspace_layers(result.workspace) == [BubblewrapWorkspace, SSHWorkspaceBackend]
    assert '--unshare-net' not in tools.bwrap_calls[0]
    continued = await agent.run('again', message_history=result.all_messages())
    assert workspace_layers(continued.workspace) == [BubblewrapWorkspace, SSHWorkspaceBackend]


async def test_capability_keeps_the_wrapped_policy_and_declines_foreign_refs(tmp_path: Path) -> None:
    agent = Agent(TestModel(), capabilities=[BubblewrapSandbox(LocalWorkspace(tmp_path, read_only=True))])

    result = await agent.run('go')

    assert workspace_layers(result.workspace) == [BubblewrapWorkspace, ReadOnlyWorkspace, LocalWorkspaceBackend]
    with pytest.raises(UserError, match="none of the agent's workspace capabilities recognized it"):
        await agent.run('go', workspace=WorkspaceRef(provider='local', id=str(tmp_path / 'elsewhere')))


@pytest.mark.skipif(not BWRAP_WORKS, reason='needs a working `bwrap` (Linux with user namespaces)')
class TestRealBubblewrap:  # pragma: no cover - CI hosts may not have bubblewrap
    async def test_commands_write_only_to_the_working_dir(self, tmp_path: Path) -> None:
        workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path / 'work')))
        (tmp_path / 'work').mkdir()

        inside = await workspace.run(['sh', '-c', 'printf ok > inside.txt'])
        await workspace.run(['sh', '-c', f'printf no > {tmp_path}/outside.txt'])

        assert inside.exit_code == 0
        assert await workspace.read_text('inside.txt') == 'ok'
        # Outside the working directory, a write fails or lands in the sandbox's private `/tmp`.
        assert not (tmp_path / 'outside.txt').exists()

    async def test_tmp_is_private(self, tmp_path: Path) -> None:
        marker = Path('/tmp') / f'pydantic-ai-bwrap-{os.getpid()}'
        marker.write_text('host')
        try:
            workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))
            assert (await workspace.run(['test', '-e', str(marker)])).exit_code == 1
        finally:
            marker.unlink()

    async def test_host_daemon_sockets_are_hidden(self, tmp_path: Path) -> None:
        workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))

        assert (await workspace.run(['sh', '-c', 'ls -A /run'])).stdout == ''

    async def test_a_detached_command_outlives_the_call(self, tmp_path: Path) -> None:
        """The harness `Shell` detaches its jobs and checks on them in later calls."""
        workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))

        await workspace.run('setsid sh -c "sleep 1; echo alive > out" < /dev/null > /dev/null 2>&1 &', shell=True)
        await anyio.sleep(3)

        assert await workspace.read_text('out') == 'alive\n'
