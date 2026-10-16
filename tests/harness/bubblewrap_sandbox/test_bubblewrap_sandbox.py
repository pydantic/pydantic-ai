"""Tests for `BubblewrapWorkspace` and the `BubblewrapSandbox` capability.

Most use a fake `bwrap` that records its arguments, so they run anywhere; the last class needs a working `bwrap`.
"""

from __future__ import annotations

import base64
import contextlib
import os
import shutil
import socket
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import (
    CommandResult,
    LocalWorkspaceBackend,
    ReadOnlyWorkspace,
    Workspace,
    WorkspaceError,
    WorkspaceReadOnlyError,
    WorkspaceRef,
    WorkspaceUnavailableError,
)
from pydantic_ai.workspaces.workspace import workspace_layers
from pydantic_ai_harness.bubblewrap_sandbox import BubblewrapSandbox, BubblewrapWorkspace
from pydantic_ai_harness.bubblewrap_sandbox._seccomp import NETWORK_FILTER_BASE64
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
            '--die-with-parent --new-session --unshare-user --unshare-ipc --unshare-uts --unshare-cgroup-try',
            *([] if network else ['--unshare-net --seccomp 3']),
            '--cap-drop ALL',
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


class _FailedHome:
    """A backend whose home probe fails, so the sandbox refuses to start."""

    @property
    def ref(self) -> None:
        return None

    async def working_dir(self) -> str:
        return '/work'

    async def run(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        del command, shell, env, timeout
        return CommandResult(exit_code=1, stdout='', stderr='no home')


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


async def test_an_inherited_path_entry_inside_the_working_dir_is_skipped(
    tools: FakeRemoteTools, tmp_path: Path
) -> None:
    """The backend's own `PATH` is the host launcher's, so a directory inside the workspace cannot supply `bwrap`."""
    impostor = tmp_path / 'bin'
    impostor.mkdir()
    ran = impostor / 'ran'
    (impostor / 'bwrap').write_text(f'#!/bin/sh\ntouch {ran}\n')
    (impostor / 'bwrap').chmod(0o755)
    backend = LocalWorkspaceBackend(tmp_path, env={'PATH': f'{impostor}{os.pathsep}{os.environ["PATH"]}'})

    result = await BubblewrapWorkspace(Workspace(backend)).run(['true'])

    assert result.exit_code == 0
    assert not ran.exists()
    assert len(tools.bwrap_calls) == 1


async def test_a_symlinked_path_entry_inside_the_working_dir_is_skipped(tools: FakeRemoteTools, tmp_path: Path) -> None:
    """A `PATH` entry outside the workspace can still be the workspace directory through a symlink."""
    real_bin = tmp_path / 'bin'
    real_bin.mkdir()
    ran = real_bin / 'ran'
    (real_bin / 'bwrap').write_text(f'#!/bin/sh\ntouch {ran}\n')
    (real_bin / 'bwrap').chmod(0o755)
    alias = tmp_path.parent / f'{tmp_path.name}-alias'
    alias.mkdir()
    (alias / 'bin').symlink_to(real_bin, target_is_directory=True)
    backend = LocalWorkspaceBackend(tmp_path, env={'PATH': f'{alias / "bin"}{os.pathsep}{os.environ["PATH"]}'})

    result = await BubblewrapWorkspace(Workspace(backend)).run(['true'])

    assert result.exit_code == 0
    assert not ran.exists()
    assert tools.bwrap_calls


async def test_bwrap_inside_the_working_dir_is_not_executed(tmp_path: Path) -> None:
    impostor = tmp_path / 'bin'
    impostor.mkdir()
    ran = impostor / 'ran'
    (impostor / 'bwrap').write_text(f'#!/bin/sh\ntouch {ran}\n')
    (impostor / 'bwrap').chmod(0o755)
    backend = LocalWorkspaceBackend(tmp_path, env={'PATH': str(impostor)})

    with pytest.raises(WorkspaceUnavailableError, match='could not start a sandbox'):
        await BubblewrapWorkspace(Workspace(backend)).run(['true'])
    assert not ran.exists()


async def test_ssh_directory_inside_the_working_dir_is_mounted_read_only(tools: FakeRemoteTools) -> None:
    workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tools.home)))

    await workspace.run(['true'])
    await workspace.run(['true'])

    ssh_dir = tools.home.resolve() / '.ssh'
    assert ssh_dir.is_dir()
    assert ssh_dir.stat().st_mode & 0o777 == 0o700
    mount = f'--ro-bind {ssh_dir} {ssh_dir}'
    assert tools.bwrap_calls[0].count(mount) == 1
    assert tools.bwrap_calls[1].count(mount) == 1


async def test_shell_startup_files_inside_the_working_dir_are_mounted_read_only(tools: FakeRemoteTools) -> None:
    """The next SSH login runs `~/.bashrc` and friends on the host, so a command must not be able to write them."""
    (tools.home / '.bashrc').write_text('export KEEP=1\n')
    (tools.home / '.cshrc').write_text('setenv KEEP 1\n')
    workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tools.home)))

    await workspace.run(['true'])

    home = tools.home.resolve()
    # Existing files are kept, missing ones are created empty, and tcsh keeps reading `~/.cshrc`.
    assert (home / '.bashrc').read_text() == 'export KEEP=1\n'
    assert (home / '.cshrc').read_text() == 'setenv KEEP 1\n'
    assert (home / '.tcshrc').read_text() == 'source ~/.cshrc\n'
    assert (home / '.zshenv').read_text() == ''
    assert (home / '.pam_environment').read_text() == ''
    assert (home / '.config' / 'fish').is_dir()
    mounts = ' '.join(
        [
            f'--ro-bind {home}/.bashrc {home}/.bashrc',
            # A mount point can't be renamed, so a command can't move `~/.config` aside to plant its own.
            f'--bind {home}/.config {home}/.config',
            f'--ro-bind {home}/.config/fish {home}/.config/fish',
            f'--ro-bind {home}/.cshrc {home}/.cshrc',
            f'--ro-bind {home}/.pam_environment {home}/.pam_environment',
            f'--ro-bind {home}/.ssh {home}/.ssh',
            f'--ro-bind {home}/.tcshrc {home}/.tcshrc',
            f'--ro-bind {home}/.zshenv {home}/.zshenv',
        ]
    )
    assert f'--bind {home} {home} {mounts} --chdir {home} --' in tools.bwrap_calls[0]


async def test_a_home_directory_inside_the_working_dir_is_pinned(tools: FakeRemoteTools, tmp_path: Path) -> None:
    """Renaming `~` aside would leave a command free to make a new one, so it is a mount point too."""
    home = tmp_path / 'home'
    home.mkdir()
    workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path, env={'HOME': str(home)})))

    await workspace.run(['true'])

    resolved = home.resolve()
    assert f'--bind {resolved} {resolved} --ro-bind {resolved}/.bashrc ' in tools.bwrap_calls[0]


@pytest.mark.parametrize(
    ('login_path', 'link', 'reason'),
    [
        ('.bashrc', Path.symlink_to, r'\.bashrc is a symbolic link'),
        ('.config/fish/config.fish', Path.symlink_to, r'config\.fish is a symbolic or hard link'),
        ('.bashrc', Path.hardlink_to, r'\.bashrc is a symbolic or hard link'),
    ],
)
async def test_a_linked_startup_file_inside_the_working_dir_is_unavailable(
    tmp_path: Path, login_path: str, link: Callable[[Path, Path], None], reason: str
) -> None:
    """A read-only mount over a link doesn't stop a command from replacing it or writing its target's other name."""
    home = tmp_path / 'home'
    target = home / 'dotfiles' / 'startup'
    target.parent.mkdir(parents=True)
    target.write_text('')
    (home / login_path).parent.mkdir(parents=True, exist_ok=True)
    link(home / login_path, target)
    backend = LocalWorkspaceBackend(home, env={'HOME': str(home)})

    with pytest.raises(WorkspaceUnavailableError, match=reason):
        await BubblewrapWorkspace(Workspace(backend)).run(['true'])


async def test_the_ssh_directory_itself_is_mounted_read_only(tools: FakeRemoteTools, tmp_path: Path) -> None:
    """The writable directory can be `~/.ssh`, not merely a parent of it."""
    home = tmp_path / 'home'
    ssh_dir = home / '.ssh'
    ssh_dir.mkdir(parents=True)
    workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(ssh_dir, env={'HOME': str(home)})))

    await workspace.run(['true'])

    resolved = str(ssh_dir.resolve())
    assert tools.bwrap_calls[0].count(f'--ro-bind {resolved} {resolved}') == 1


async def test_the_filesystem_root_keeps_ssh_read_only_and_refuses_to_launch(tools: FakeRemoteTools) -> None:
    """`/` contains both `~/.ssh` and `/bin/sh`, so the launcher must not start."""
    workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend('/', env={'HOME': str(tools.home)})))

    with pytest.raises(WorkspaceUnavailableError, match='could not start a sandbox'):
        await workspace.run(['true'])

    ssh_dir = tools.home.resolve() / '.ssh'
    assert ssh_dir.is_dir()
    assert ssh_dir.stat().st_mode & 0o777 == 0o700
    assert tools.bwrap_calls == []


async def test_a_home_directory_probe_that_fails_is_unavailable() -> None:
    backend = _FailedHome()
    assert backend.ref is None
    workspace = BubblewrapWorkspace(Workspace(backend))

    with pytest.raises(WorkspaceUnavailableError, match='could not read the host account'):
        await workspace.run(['true'])


async def test_a_relative_home_directory_is_unavailable(tmp_path: Path) -> None:
    backend = LocalWorkspaceBackend(tmp_path, env={'HOME': 'relative'})

    with pytest.raises(WorkspaceUnavailableError, match='could not read the host account'):
        await BubblewrapWorkspace(Workspace(backend)).run(['true'])


async def test_zdotdir_inside_the_working_dir_is_mounted_read_only(tools: FakeRemoteTools, tmp_path: Path) -> None:
    """ZDOTDIR overrides HOME for zsh startup, so sshd_config SetEnv could escape the sandbox through it."""
    work, home = tmp_path / 'work', tmp_path / 'home'
    zdotdir = work / 'zsh'
    zdotdir.mkdir(parents=True)
    home.mkdir()
    backend = LocalWorkspaceBackend(work, env={'HOME': str(home), 'ZDOTDIR': str(zdotdir)})
    workspace = BubblewrapWorkspace(Workspace(backend))

    await workspace.run(['true'])

    resolved = str(zdotdir.resolve())
    # The ZDOTDIR directory is a parent mount so it cannot be renamed aside.
    assert tools.bwrap_calls[0].count(f'--bind {resolved} {resolved}') == 1
    # The .zshenv inside it is read-only.
    assert tools.bwrap_calls[0].count(f'--ro-bind {resolved}/.zshenv {resolved}/.zshenv') == 1


async def test_zdotdir_same_as_home_uses_standard_protection(tools: FakeRemoteTools, tmp_path: Path) -> None:
    """When ZDOTDIR equals HOME, no extra mounts are needed since HOME protection already covers it."""
    home = tmp_path / 'home'
    home.mkdir()
    backend = LocalWorkspaceBackend(tmp_path, env={'HOME': str(home), 'ZDOTDIR': str(home)})
    workspace = BubblewrapWorkspace(Workspace(backend))

    await workspace.run(['true'])

    resolved = str(home.resolve())
    # Only one .zshenv mount (from HOME protection), not two.
    assert tools.bwrap_calls[0].count(f'--ro-bind {resolved}/.zshenv {resolved}/.zshenv') == 1


async def test_zdotdir_outside_working_dir_needs_no_extra_protection(tools: FakeRemoteTools, tmp_path: Path) -> None:
    """A ZDOTDIR outside the writable directory cannot be planted by a sandboxed command."""
    work, home, zdotdir = tmp_path / 'work', tmp_path / 'home', tmp_path / 'zsh'
    for directory in (work, home, zdotdir):
        directory.mkdir()
    backend = LocalWorkspaceBackend(work, env={'HOME': str(home), 'ZDOTDIR': str(zdotdir)})
    workspace = BubblewrapWorkspace(Workspace(backend))

    await workspace.run(['true'])

    # No mounts for ZDOTDIR since it's outside the working directory.
    assert str(zdotdir.resolve()) not in tools.bwrap_calls[0]


async def test_a_relative_zdotdir_is_unavailable(tmp_path: Path) -> None:
    """A relative ZDOTDIR would resolve against the working directory, allowing a sandbox escape."""
    # The home directory must exist, or the sandbox fails on resolving it before it ever reads `ZDOTDIR`.
    home = tmp_path / 'home'
    home.mkdir()
    backend = LocalWorkspaceBackend(tmp_path, env={'HOME': str(home), 'ZDOTDIR': 'relative'})

    with pytest.raises(WorkspaceUnavailableError, match='relative ZDOTDIR'):
        await BubblewrapWorkspace(Workspace(backend)).run(['true'])


async def test_a_planted_mkdir_on_path_is_not_used(tools: FakeRemoteTools, tmp_path: Path) -> None:
    """Creating `~/.ssh` is `/bin/mkdir`, not whatever `PATH` finds inside the workspace."""
    home = tmp_path / 'home'
    impostor = home / 'bin'
    impostor.mkdir(parents=True)
    ran = impostor / 'ran'
    planted = impostor / 'mkdir'
    planted.write_text(f'#!/bin/sh\ntouch {ran}\n')
    planted.chmod(0o755)
    backend = LocalWorkspaceBackend(
        home, env={'HOME': str(home), 'PATH': f'{impostor}{os.pathsep}{os.environ["PATH"]}'}
    )

    await BubblewrapWorkspace(Workspace(backend)).run(['true'])

    assert not ran.exists()
    assert (home.resolve() / '.ssh').is_dir()


async def test_ssh_home_resolution_does_not_run_planted_helpers(tools: FakeRemoteTools, tmp_path: Path) -> None:
    """Resolving `$HOME` over SSH stays inside `/bin/sh`. It does not run `readlink`, `wc`, or `base64`."""
    impostor = tmp_path / 'bin'
    impostor.mkdir()
    ran = impostor / 'ran'
    for name in ('readlink', 'wc', 'base64'):
        planted = impostor / name
        planted.write_text(f'#!/bin/sh\ntouch {ran}\n')
        planted.chmod(0o755)
    backend = SSHWorkspaceBackend(
        'box',
        working_dir=str(tmp_path),
        env={'HOME': str(tools.home), 'PATH': f'{impostor}{os.pathsep}{os.environ["PATH"]}'},
    )

    result = await BubblewrapWorkspace(Workspace(backend)).run(['true'])

    assert result.exit_code == 0
    assert not ran.exists()


async def test_an_ssh_path_that_is_not_a_directory_is_unavailable(tmp_path: Path) -> None:
    home = tmp_path / 'home'
    home.mkdir()
    (home / '.ssh').write_text('nope')
    backend = LocalWorkspaceBackend(home, env={'HOME': str(home)})

    with pytest.raises(WorkspaceUnavailableError, match=r'could not make .*\.ssh is not a directory'):
        await BubblewrapWorkspace(Workspace(backend)).run(['true'])


def test_durable_policy_is_the_network_and_bwrap_args(tmp_path: Path) -> None:
    workspace = BubblewrapWorkspace(
        Workspace(LocalWorkspaceBackend(tmp_path)), network=True, bwrap_args=('--bind', '/srv', '/srv')
    )

    assert workspace.durable_policy() == (True, ('--bind', '/srv', '/srv'))


async def test_network_is_shared_only_when_asked(tools: FakeRemoteTools, tmp_path: Path) -> None:
    workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)), network=True)
    working_dir = await workspace.working_dir()

    await workspace.run(['true'])

    assert tools.bwrap_calls[0].startswith(_sandbox_args(working_dir, network=True))
    assert not (tools.bin_dir / 'seccomp-filter').exists()


async def test_without_network_bwrap_gets_the_socket_filter(tools: FakeRemoteTools, tmp_path: Path) -> None:
    workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))

    result = await workspace.run(['sh', '-c', 'cat; printf done'])

    # The filter arrived on descriptor 3, and the command still reads an empty stdin.
    assert tools.seccomp_filter == base64.b64decode(NETWORK_FILTER_BASE64)
    assert (result.exit_code, result.stdout) == (0, 'done')


async def test_file_methods_run_in_the_sandbox(tools: FakeRemoteTools, tmp_path: Path) -> None:
    """So a symlink swapped in after a path check can't lead a write out of the sandbox."""
    workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))
    working_dir = await workspace.working_dir()

    await workspace.make_dir('docs')
    await workspace.write_text('docs/notes.txt', 'hello')

    assert await workspace.read_text('docs/notes.txt') == 'hello'
    assert await workspace.exists('docs/notes.txt')
    assert (await workspace.stat('docs/notes.txt')).size == 5
    assert [entry.name for entry in await workspace.list_dir('docs')] == ['notes.txt']
    assert await workspace.realpath('docs/../docs/notes.txt') == f'{working_dir}/docs/notes.txt'
    await workspace.remove('docs')
    assert not (tmp_path / 'docs').exists()
    assert tools.bwrap_calls
    assert all(call.startswith(_sandbox_args(working_dir)) for call in tools.bwrap_calls)


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
    # A read-only workspace runs no commands, so its files are read from the host.
    (tmp_path / 'notes.txt').write_text('hello')
    assert await workspace.read_text('notes.txt') == 'hello'
    with pytest.raises(WorkspaceReadOnlyError):
        await workspace.write_text('notes.txt', 'changed')


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

    async def test_file_methods_cannot_write_through_a_symlink_out_of_the_working_dir(self, tmp_path: Path) -> None:
        outside = tmp_path / 'outside.txt'
        outside.write_text('host')
        (tmp_path / 'work').mkdir()
        (tmp_path / 'work' / 'link').symlink_to(outside)
        workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path / 'work')))

        # The write fails, or lands in the sandbox's private `/tmp`: either way the host file is untouched.
        with contextlib.suppress(OSError, WorkspaceError):
            await workspace.write_text('link', 'escaped')

        assert outside.read_text() == 'host'

    async def test_without_network_host_unix_sockets_are_unreachable(self, tmp_path: Path) -> None:
        # Not under `/tmp`, which the sandbox replaces anyway: the filter must stop it, not a mount.
        directory = Path(tempfile.mkdtemp(prefix='pydantic-ai-bwrap-sock-', dir='/var/tmp'))
        server = socket.socket(socket.AF_UNIX)
        try:
            server.bind(str(directory / 'host.sock'))
            server.listen()
            connect = f'import socket; socket.socket(socket.AF_UNIX).connect({str(directory / "host.sock")!r})'
            sandboxed = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))
            networked = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)), network=True)

            blocked = await sandboxed.run(['python3', '-c', connect])
            allowed = await networked.run(['python3', '-c', connect])

            assert blocked.exit_code != 0 and 'PermissionError' in blocked.stderr
            assert allowed.exit_code == 0
        finally:
            server.close()
            shutil.rmtree(directory)

    async def test_commands_cannot_change_what_the_next_login_runs(self, tmp_path: Path) -> None:
        home = tmp_path / 'home'
        home.mkdir()
        workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(home, env={'HOME': str(home)})))

        for attack in (
            'echo touch pwned >> .bashrc',
            'rm -f .zshenv && echo touch pwned > .zshenv',
            'mv .config .moved && mkdir -p .config/fish && echo touch pwned > .config/fish/config.fish',
            'echo touch pwned > .ssh/rc',
        ):
            assert (await workspace.run(attack, shell=True)).exit_code != 0
        assert (await workspace.run('printf ok > notes.txt', shell=True)).exit_code == 0

        assert (home / '.bashrc').read_text() == ''
        assert (home / '.zshenv').read_text() == ''
        assert not (home / '.config' / 'fish' / 'config.fish').exists()
        assert not (home / '.ssh' / 'rc').exists()

    async def test_commands_cannot_write_to_zdotdir_zshenv(self, tmp_path: Path) -> None:
        """ZDOTDIR override from sshd_config SetEnv cannot escape the sandbox through zsh startup."""
        home = tmp_path / 'home'
        home.mkdir()
        zdotdir = tmp_path / 'zsh'
        zdotdir.mkdir()
        workspace = BubblewrapWorkspace(
            Workspace(LocalWorkspaceBackend(tmp_path, env={'HOME': str(home), 'ZDOTDIR': str(zdotdir)}))
        )

        attack = await workspace.run(f'echo touch pwned >> {zdotdir}/.zshenv', shell=True)
        normal = await workspace.run('printf ok > notes.txt', shell=True)

        assert attack.exit_code != 0
        assert normal.exit_code == 0
        assert (zdotdir / '.zshenv').read_text() == ''

    async def test_host_daemon_sockets_are_hidden(self, tmp_path: Path) -> None:
        workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))

        assert (await workspace.run(['sh', '-c', 'ls -A /run'])).stdout == ''

    async def test_a_detached_command_outlives_the_call(self, tmp_path: Path) -> None:
        """The harness `Shell` detaches its jobs and checks on them in later calls."""
        workspace = BubblewrapWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))

        await workspace.run('setsid sh -c "sleep 1; echo alive > out" < /dev/null > /dev/null 2>&1 &', shell=True)
        await anyio.sleep(3)

        assert await workspace.read_text('out') == 'alive\n'
