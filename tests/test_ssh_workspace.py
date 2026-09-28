"""Tests for `SSHWorkspaceBackend` and the `SSHWorkspace` capability, against a fake `ssh` that runs commands locally."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import SSHWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import (
    ReadOnlyWorkspace,
    SSHWorkspaceBackend,
    Workspace,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from .fake_remote_tools import FakeRemoteTools, install_fake_remote_tools

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(os.name != 'posix', reason='`SSHWorkspaceBackend` runs `ssh` as a POSIX subprocess'),
]


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeRemoteTools:
    return install_fake_remote_tools(tmp_path, monkeypatch)


async def test_commands_and_files_reach_the_remote_working_dir(tools: FakeRemoteTools, tmp_path: Path) -> None:
    backend = SSHWorkspaceBackend('dev@box', working_dir=str(tmp_path), ssh_args=['-p', '2222'])
    workspace = Workspace(backend)

    await workspace.write_text('notes.txt', 'hello')
    result = await workspace.run('cat notes.txt; printf oops >&2; exit 3', shell=True)

    assert (result.exit_code, result.stdout, result.stderr) == (3, 'hello', 'oops')
    assert (tmp_path / 'notes.txt').read_text() == 'hello'
    assert backend.ref == WorkspaceRef(provider='ssh', id=f'dev@box:{tmp_path}')
    assert tools.ssh_options[:5] == ['-T', '-o', 'BatchMode=yes', '-p', '2222']


async def test_working_dir_defaults_to_the_login_directory(tools: FakeRemoteTools) -> None:
    (tools.home / '-project').mkdir()
    default = SSHWorkspaceBackend('box')
    relative = SSHWorkspaceBackend('box', working_dir='-project/')

    assert default.ref == WorkspaceRef(provider='ssh', id='box')
    assert await default.working_dir() == str(tools.home.resolve())
    assert relative.ref == WorkspaceRef(provider='ssh', id='box:-project')
    assert await relative.working_dir() == str((tools.home / '-project').resolve())


async def test_env_layers_the_call_over_the_backend(tools: FakeRemoteTools) -> None:
    backend = SSHWorkspaceBackend('box', env={'A': 'backend', 'B': "it's $HOME"})

    result = await backend.run(['sh', '-c', 'printf "%s|%s" "$A" "$B"'], env={'A': 'call'})

    assert result.stdout == "call|it's $HOME"
    with pytest.raises(ValueError, match='invalid environment variable name'):
        await backend.run(['true'], env={'NOT-A-NAME': 'x'})


async def test_a_remote_exit_255_is_a_result_not_a_connection_failure(tools: FakeRemoteTools) -> None:
    assert (await SSHWorkspaceBackend('box').run(['sh', '-c', 'exit 255'])).exit_code == 255


async def test_an_unreachable_host_is_unavailable(tools: FakeRemoteTools) -> None:
    with pytest.raises(WorkspaceUnavailableError, match='is unavailable: ssh: connect to host unreachable'):
        await SSHWorkspaceBackend('unreachable').run(['true'])


async def test_a_dropped_connection_is_unavailable(tools: FakeRemoteTools, tmp_path: Path) -> None:
    backend = SSHWorkspaceBackend('dropped', working_dir=str(tmp_path))

    with pytest.raises(WorkspaceUnavailableError, match='the connection was lost during the command'):
        await backend.working_dir()


async def test_a_missing_working_dir_is_unavailable(tools: FakeRemoteTools, tmp_path: Path) -> None:
    with pytest.raises(WorkspaceUnavailableError, match=r'is unavailable: .*missing'):
        await SSHWorkspaceBackend('box', working_dir=str(tmp_path / 'missing')).working_dir()


async def test_timeouts_and_output_limits_keep_only_the_commands_stderr(tools: FakeRemoteTools) -> None:
    backend = SSHWorkspaceBackend('box')

    with pytest.raises(WorkspaceTimeoutError) as timeout:
        await backend.run('printf partial >&2; sleep 30', shell=True, timeout=1)
    with pytest.raises(WorkspaceOutputLimitError, match='SSH workspace output exceeded') as limit:
        await backend.run('printf big >&2; head -c 11000000 /dev/zero', shell=True)

    assert timeout.value.stderr == 'partial'
    assert limit.value.stderr == 'big'


async def test_stopping_a_command_kills_its_process_group_on_the_host(tools: FakeRemoteTools) -> None:
    """Killing the local `ssh` leaves the remote command running, so a second connection stops it."""
    tag = '__pydantic_ai_ssh_job_0123456789abcdef'
    # Stands in for the remote command: its own session, with the tag on its command line and a child without it.
    remote = subprocess.Popen(['sh', '-c', f': {tag}; sleep 60; :'], start_new_session=True)
    bystander = subprocess.Popen(['sh', '-c', ': __pydantic_ai_ssh_job_other; sleep 60; :'], start_new_session=True)
    try:
        with anyio.fail_after(10):
            await SSHWorkspaceBackend('box')._stop(tag)  # pyright: ignore[reportPrivateUsage]
            await anyio.to_thread.run_sync(remote.wait)

        assert bystander.poll() is None
    finally:
        remote.kill()
        bystander.kill()
        remote.wait()
        bystander.wait()


async def test_a_login_banner_stays_out_of_the_output(tools: FakeRemoteTools) -> None:
    backend = SSHWorkspaceBackend('chatty')

    result = await backend.run(['sh', '-c', 'printf out; printf err >&2'])

    assert await backend.working_dir() == str(tools.home.resolve())
    assert (result.stdout, result.stderr) == ('out', 'err')


async def test_resolving_the_working_dir_counts_against_the_first_timeout(tools: FakeRemoteTools) -> None:
    with anyio.fail_after(10), pytest.raises(WorkspaceTimeoutError):
        await SSHWorkspaceBackend('slow').run(['true'], timeout=1)


async def test_stderr_from_a_background_child_after_the_command_is_kept(tools: FakeRemoteTools) -> None:
    # The child keeps stderr open, so `ssh` waits for it and its output lands after the wrapper's marker.
    result = await SSHWorkspaceBackend('box').run('(sleep 1; printf late >&2) & printf done', shell=True)

    assert (result.exit_code, result.stdout, result.stderr) == (0, 'done', 'late')


async def test_an_empty_argv_is_rejected(tools: FakeRemoteTools) -> None:
    with pytest.raises(ValueError, match='command must not be empty'):
        await SSHWorkspaceBackend('box').run([])


async def test_a_timeout_is_raised_about_on_time(tools: FakeRemoteTools) -> None:
    """Stopping the remote command costs one round trip, not the `SIGKILL` grace period."""
    backend = SSHWorkspaceBackend('box')
    await backend.working_dir()
    started = anyio.current_time()

    with pytest.raises(WorkspaceTimeoutError):
        await backend.run(['sleep', '30'], timeout=0.5)

    # About 0.6s here; waiting out the grace period would make it at least 1.5s.
    assert anyio.current_time() - started < 1.45


async def test_invalid_configuration_fails_at_construction(tools: FakeRemoteTools) -> None:
    with pytest.raises(ValueError, match='destination must be a host'):
        SSHWorkspaceBackend('-oProxyCommand=evil')
    with pytest.raises(ValueError, match='invalid environment variable name'):
        SSHWorkspaceBackend('box', env={'A=B': 'x'})
    with pytest.raises(TypeError, match='ssh_args must be a sequence'):
        SSHWorkspaceBackend('box', ssh_args='-p 2222')
    with pytest.raises(ValueError, match='destination must be a host'):
        SSHWorkspace('')


async def test_capability_gives_tools_the_remote_workspace(tools: FakeRemoteTools, tmp_path: Path) -> None:
    agent = Agent(TestModel(call_tools=['probe']), capabilities=[SSHWorkspace('box', working_dir=str(tmp_path))])

    @agent.tool
    async def probe(ctx: RunContext[object]) -> str:
        await ctx.workspace.write_text('probe.txt', 'remote')
        return (await ctx.workspace.run(['cat', 'probe.txt'])).stdout

    result = await agent.run('go')

    assert result.output == '{"probe":"remote"}'
    assert isinstance(result.workspace.backend, SSHWorkspaceBackend)
    assert result.workspace.ref == WorkspaceRef(provider='ssh', id=f'box:{tmp_path}')


async def test_capability_declines_a_ref_for_another_host(tools: FakeRemoteTools) -> None:
    agent = Agent(TestModel(), capabilities=[SSHWorkspace('box')])

    with pytest.raises(UserError, match="none of the agent's workspace capabilities recognized it"):
        await agent.run('go', workspace=WorkspaceRef(provider='ssh', id='elsewhere'))
    assert (await agent.run('go', workspace=WorkspaceRef(provider='ssh', id='box'))).workspace.ref == WorkspaceRef(
        provider='ssh', id='box'
    )


async def test_a_repeated_ssh_workspace_resolves_to_the_later_one(tools: FakeRemoteTools) -> None:
    agent = Agent(TestModel(), capabilities=[SSHWorkspace('first'), SSHWorkspace('second')])

    assert (await agent.run('go')).workspace.ref == WorkspaceRef(provider='ssh', id='second')


async def test_agent_spec_builds_a_read_only_ssh_workspace(tools: FakeRemoteTools) -> None:
    agent = Agent.from_spec(
        {'model': 'test', 'capabilities': [{'SSHWorkspace': {'destination': 'box', 'read_only': True}}]}
    )

    result = await agent.run('go')

    assert isinstance(result.workspace, ReadOnlyWorkspace)
    assert await result.workspace.working_dir() == str(tools.home.resolve())
