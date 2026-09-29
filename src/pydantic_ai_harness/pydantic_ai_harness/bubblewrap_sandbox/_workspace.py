"""A wrapper that runs another workspace's commands in a [bubblewrap](https://github.com/containers/bubblewrap) sandbox."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    Workspace,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceRef,
    WorkspaceUnavailableError,
    WrapperWorkspace,
)
from pydantic_ai_harness._workspace_provider import check_timeout
from pydantic_ai_harness.bubblewrap_sandbox._seccomp import NETWORK_FILTER_BASE64

__all__ = ('BubblewrapWorkspace',)

_DNS_DIRS = ('/run/systemd/resolve', '/run/NetworkManager', '/run/resolvconf')
"""Where `/etc/resolv.conf` usually points; the ones that exist are mounted back over the empty `/run`."""

_LOAD_FILTER = 'printf %s "$1" | base64 -d | { shift; exec 3<&0 </dev/null; exec "$@"; }'
"""Decodes the seccomp filter (the first argument) onto descriptor 3 for `bwrap --seccomp 3`, then runs `bwrap`.

The filter goes in the command line because `bwrap` may run on another host, and the wrapped workspace
passes it a command, not open files.
"""

_PROBE_TIMEOUT = 30.0
"""Seconds to wait for the check that `bwrap` can start a sandbox at all."""


class _SandboxedCommands(WorkspaceBackend, SupportsCommands):
    """A backend that runs every command in the sandbox, for core's shell-based file methods to build on."""

    def __init__(self, sandbox: BubblewrapWorkspace):
        self._sandbox = sandbox

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._sandbox.ref

    async def working_dir(self) -> str:
        return await self._sandbox.working_dir()

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        return await self._sandbox.run(command, shell=shell, env=env, timeout=timeout)


class BubblewrapWorkspace(WrapperWorkspace):
    """A [`Workspace`][pydantic_ai.workspaces.Workspace] that runs commands in a bubblewrap (`bwrap`) sandbox.

    The wrapped workspace runs `bwrap`, so the sandbox is on its host: wrap an
    [`SSHWorkspaceBackend`][pydantic_ai_harness.ssh_workspace.SSHWorkspaceBackend] to sandbox commands on the
    remote host. `bwrap` must be installed there (Linux only).

    Commands see the host read-only, with an empty `/run`, a private `/tmp`, and no network, and can only
    write to the working directory. Without the network, a seccomp filter also stops them from connecting to
    or serving any socket, including the host's Unix sockets. They share the host's processes, so a detached
    command keeps running after the call that started it, and they can signal the host user's other processes.

    File methods run in the sandbox too, as shell commands, so they see what commands see and can't be
    tricked into writing outside it through a symlink. Only when the wrapped workspace is read-only (and so
    runs no commands) do its file methods read the host directly.

    Args:
        wrapped: The workspace whose host runs the sandbox.
        network: Whether commands share the host's network; without it, a seccomp filter also blocks every
            socket connection.
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
        self._files = wrapped if wrapped.read_only else Workspace(_SandboxedCommands(self))

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
            # The user namespace is required, not tried: with the host's processes visible, it is what keeps
            # a command from reaching the host's files through another process's `/proc/<pid>/root`.
            *('--unshare-user', '--unshare-ipc', '--unshare-uts', '--unshare-cgroup-try'),
            *([] if self._network else ['--unshare-net', '--seccomp', '3']),
            *('--cap-drop', 'ALL'),
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
        result = await self.wrapped.run(self._launch([*sandbox, *env_args, '--', *argv]), timeout=timeout)
        if result.exit_code != 0 and not self._sandbox_works:
            # `bwrap` exits like the command when it can't start one, so tell the two apart once.
            probe = await self.wrapped.run(self._launch([*sandbox, '--', 'true']), timeout=_PROBE_TIMEOUT)
            if probe.exit_code != 0:
                raise WorkspaceUnavailableError(
                    'bubblewrap could not start a sandbox; install `bwrap` on the host that runs the commands '
                    f'and allow it to create user namespaces: {probe.stderr.strip()}'
                )
        self._sandbox_works = True
        return result

    def _launch(self, bwrap: list[str]) -> list[str]:
        # A private network namespace doesn't cover Unix sockets, so without the network a filter blocks those too.
        return bwrap if self._network else ['sh', '-c', _LOAD_FILTER, 'sh', NETWORK_FILTER_BASE64, *bwrap]

    async def read_bytes(self, path: str) -> bytes:
        return await self._files.read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self._files.write_bytes(path, data)

    async def stat(self, path: str) -> FileEntry:
        return await self._files.stat(path)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        return await self._files.list_dir(path)

    async def make_dir(self, path: str) -> None:
        await self._files.make_dir(path)

    async def remove(self, path: str) -> None:
        await self._files.remove(path)

    async def exists(self, path: str) -> bool:
        return await self._files.exists(path)

    async def realpath(self, path: str) -> str:
        return await self._files.realpath(path)
