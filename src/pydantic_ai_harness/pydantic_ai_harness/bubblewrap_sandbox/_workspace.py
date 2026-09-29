"""A wrapper that runs another workspace's commands in a [bubblewrap](https://github.com/containers/bubblewrap) sandbox."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic_ai.workspaces import (
    CommandResult,
    Workspace,
    WorkspaceCommand,
    WorkspaceUnavailableError,
    WrapperWorkspace,
)
from pydantic_ai_harness._workspace_provider import check_timeout

__all__ = ('BubblewrapWorkspace',)

_DNS_DIRS = ('/run/systemd/resolve', '/run/NetworkManager', '/run/resolvconf')
"""Where `/etc/resolv.conf` usually points; the ones that exist are mounted back over the empty `/run`."""


class BubblewrapWorkspace(WrapperWorkspace):
    """A [`Workspace`][pydantic_ai.workspaces.Workspace] that runs commands in a bubblewrap (`bwrap`) sandbox.

    The wrapped workspace runs `bwrap`, so the sandbox is on its host: wrap an
    [`SSHWorkspaceBackend`][pydantic_ai_harness.ssh_workspace.SSHWorkspaceBackend] to sandbox commands on the
    remote host. `bwrap` must be installed there (Linux only).

    Commands see the host read-only, with an empty `/run` (so no host daemon sockets), a private
    `/tmp`, and no network, and can only write to the working directory. They share the host's processes, so a detached command keeps running after
    the call that started it, and they can signal the host user's other processes. File methods are
    not sandboxed: they go to the wrapped workspace.

    Args:
        wrapped: The workspace whose host runs the sandbox.
        network: Whether commands can reach the network.
        bwrap_args: Extra `bwrap` arguments, placed after the defaults so they can override them,
            such as `['--bind', path, path]` for another writable directory or `['--tmpfs', secrets_dir]`
            to hide one.
    """

    def __init__(self, wrapped: Workspace, *, network: bool = False, bwrap_args: Sequence[str] = ()):
        if isinstance(bwrap_args, str):
            raise TypeError('bwrap_args must be a sequence of arguments, not a string')
        super().__init__(wrapped)
        self._network = network
        self._bwrap_args = tuple(bwrap_args)
        self._sandbox_works = False

    async def _sandbox(self) -> list[str]:
        working_dir = await self.wrapped.working_dir()
        # Later mounts win, so the working directory and `bwrap_args` override the read-only root.
        return [
            'bwrap',
            '--die-with-parent',
            '--new-session',
            # `--unshare-all` without its PID namespace: `bwrap` ends that namespace, killing every
            # process in it, when the command exits, so detached commands (the harness `Shell`'s jobs)
            # would die with the call that started them, and a later call could not see or signal them.
            *('--unshare-user-try', '--unshare-ipc', '--unshare-uts', '--unshare-cgroup-try'),
            *([] if self._network else ['--unshare-net']),
            *('--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp'),
            # Host daemons listen on sockets under `/run` (Docker, podman, D-Bus), and a read-only mount
            # doesn't stop a connection, so hide it; with the network on, bring back the DNS configuration.
            *('--tmpfs', '/run'),
            *(
                arg
                for directory in (_DNS_DIRS if self._network else ())
                for arg in ('--ro-bind-try', directory, directory)
            ),
            *('--bind', working_dir, working_dir),
            *self._bwrap_args,
            *('--chdir', working_dir),
        ]

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        check_timeout(timeout)
        if isinstance(command, str):
            if not shell:
                raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
            argv = ['sh', '-c', command]
        elif shell:
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')
        elif not command:
            raise ValueError('command must not be empty')
        else:
            # Through `sh`, a missing program exits 127 as the contract says, not with `bwrap`'s own 1.
            argv = ['sh', '-c', 'exec "$@"', 'sh', *command]
        sandbox = await self._sandbox()
        # Set inside the sandbox, not on `bwrap` itself, so a `PATH` or `LD_PRELOAD` can't swap out `bwrap`.
        env_args = [arg for name, value in (env or {}).items() for arg in ('--setenv', name, value)]
        result = await self.wrapped.run([*sandbox, *env_args, '--', *argv], timeout=timeout)
        if result.exit_code != 0 and not self._sandbox_works:
            # `bwrap` exits like the command when it can't start one, so tell the two apart once.
            probe = await self.wrapped.run([*sandbox, '--', 'true'])
            if probe.exit_code != 0:
                raise WorkspaceUnavailableError(
                    'bubblewrap could not start a sandbox; install `bwrap` on the host that runs the commands '
                    f'and allow it to create user namespaces: {probe.stderr.strip()}'
                )
        self._sandbox_works = True
        return result
