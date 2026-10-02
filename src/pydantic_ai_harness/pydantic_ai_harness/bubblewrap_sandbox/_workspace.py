"""A wrapper that runs another workspace's commands in a [bubblewrap](https://github.com/containers/bubblewrap) sandbox."""

from __future__ import annotations

import posixpath
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

_TRUSTED_LAUNCH = r"""wd=$1
filter=$2
shift 2
# `working_dir()` is already canonical. A `PATH` entry may be the same directory through a symlink,
# which a textual prefix would miss (`/var` and `/private/var` on macOS).
if [ -d "$wd" ]; then
  wd=$(cd "$wd" && pwd -P)
fi
inside() {
  if [ "$2" = / ]; then
    return 0
  fi
  case $1 in
    "$2"|"$2"/*) return 0 ;;
  esac
  return 1
}
sh_dir=$(cd /bin && pwd -P) || sh_dir=/bin
if inside "$sh_dir/sh" "$wd"; then
  printf '%s\n' 'bubblewrap cannot start: /bin/sh is inside the working directory' >&2
  exit 127
fi
pick() {
  name=$1
  saved=$IFS
  IFS=:
  for dir in $PATH; do
    [ -n "$dir" ] || continue
    case $dir in
      /*) ;;
      *) continue ;;
    esac
    canon=$(cd "$dir" 2>/dev/null && pwd -P) || continue
    if inside "$canon" "$wd"; then
      continue
    fi
    cand=$canon/$name
    if [ -x "$cand" ] && [ ! -d "$cand" ]; then
      printf '%s\n' "$cand"
      IFS=$saved
      return 0
    fi
  done
  IFS=$saved
  return 1
}
shift
resolved=$(pick bwrap) || { printf '%s\n' 'bwrap: command not found' >&2; exit 127; }
if [ -n "$filter" ]; then
  decoder=$(pick base64) || { printf '%s\n' 'base64: command not found' >&2; exit 127; }
  printf %s "$filter" | "$decoder" -d | { exec 3<&0 </dev/null; exec "$resolved" "$@"; }
else
  exec "$resolved" "$@"
fi
"""
"""Runs `bwrap` from a `PATH` directory outside the working directory.

`/bin/sh` is absolute, so the host does not resolve the launcher through `PATH`. The script then
skips relative entries and anything inside the writable working directory, compared as canonical
paths: a command can plant
`bwrap`, `sh` or `base64` there, and the next call would otherwise run that plant on the host,
before the sandbox exists. It also refuses to start when `/bin/sh` itself is inside that directory,
because the absolute launcher would then be writable. With the network off, `base64` is chosen the
same way and decodes the seccomp filter onto descriptor 3. The filter travels on the command line
because `bwrap` may run on another host, which receives a command, not open files.
"""

_ENSURE_SSH_DIR = r"""d=$1
if [ -d "$d" ]; then
  exit 0
fi
if [ -e "$d" ] || [ -L "$d" ]; then
  exit 2
fi
/bin/mkdir -m 700 "$d"
"""
"""Create `~/.ssh` on the host that runs commands when the writable directory would contain it.

OpenSSH runs `~/.ssh/rc` before the command. The directory is then bind-mounted read-only, so a
sandboxed command cannot plant that file for the next connection. An existing directory is left
alone, including a symlink to one; anything else at the path is a failure.
"""

_CANONICAL_HOME = r"""case "$HOME" in
  /*) ;;
  *) exit 1 ;;
esac
cd -P -- "$HOME" || exit 1
pwd -P
"""
"""`$HOME` as a canonical absolute path, using only shell builtins."""

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
    `bwrap` and, without the network, `base64` are taken from a `PATH` directory outside the working
    directory, so a command cannot replace the next launch by writing there. When that directory contains
    the account's `~/.ssh`, it is bind-mounted read-only (and created if missing): OpenSSH runs
    `~/.ssh/rc` before the next connection.

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
        self._ssh_guard: tuple[str, ...] | None = None

    def durable_policy(self) -> tuple[object, ...]:
        """`(network, bwrap_args)`, so a durable unit cannot rebuild this sandbox with another."""
        return (self._network, self._bwrap_args)

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
            *await self._ssh_directory_mount(working_dir),
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
        # Set inside the sandbox, not on the host launcher, so a call `env` cannot swap out `bwrap`.
        env_args = [arg for name, value in (env or {}).items() for arg in ('--setenv', name, value)]
        working_dir = await self.wrapped.working_dir()
        result = await self.wrapped.run(
            self._launch([*sandbox, *env_args, '--', *argv], working_dir=working_dir), timeout=timeout
        )
        if result.exit_code != 0 and not self._sandbox_works:
            # `bwrap` exits like the command when it can't start one, so tell the two apart once.
            probe = await self.wrapped.run(
                self._launch([*sandbox, '--', 'true'], working_dir=await self.wrapped.working_dir()),
                timeout=_PROBE_TIMEOUT,
            )
            if probe.exit_code != 0:
                raise WorkspaceUnavailableError(
                    'bubblewrap could not start a sandbox; install `bwrap` on the host that runs the commands '
                    f'and allow it to create user namespaces: {probe.stderr.strip()}'
                )
        self._sandbox_works = True
        return result

    def _launch(self, bwrap: list[str], *, working_dir: str) -> list[str]:
        # A private network namespace doesn't cover Unix sockets, so without the network a filter blocks those too.
        filter_arg = '' if self._network else NETWORK_FILTER_BASE64
        return ['/bin/sh', '-c', _TRUSTED_LAUNCH, 'sh', working_dir, filter_arg, *bwrap]

    async def _ssh_directory_mount(self, working_dir: str) -> tuple[str, ...]:
        """Bind `~/.ssh` read-only when the writable directory contains it.

        OpenSSH runs `~/.ssh/rc` before the requested command when `PermitUserRC` is enabled, which
        is the default. The next workspace call opens a new connection, so a command that can write
        that file runs on the host, outside `bwrap`. `bwrap_args` come after this and can override it.
        """
        if self._ssh_guard is not None:
            return self._ssh_guard
        # `cd -P` and `pwd -P` are builtins. `Workspace.realpath` over SSH runs `readlink`, `wc` and
        # `base64` from `PATH`, which a writable directory on that `PATH` could supply.
        reported = await self.wrapped.run(['/bin/sh', '-c', _CANONICAL_HOME], timeout=_PROBE_TIMEOUT)
        home = reported.stdout.removesuffix('\n')
        if reported.exit_code != 0 or not posixpath.isabs(home):
            raise WorkspaceUnavailableError(
                "bubblewrap could not read the host account's home directory, so it cannot keep `~/.ssh` "
                'out of the writable sandbox'
            )
        ssh_dir = posixpath.join(home, '.ssh')
        root = posixpath.normpath(working_dir)
        inside = root == '/' or ssh_dir == root or ssh_dir.startswith(root + '/')
        if not inside:
            self._ssh_guard = ()
            return ()
        created = await self.wrapped.run(['/bin/sh', '-c', _ENSURE_SSH_DIR, 'sh', ssh_dir], timeout=_PROBE_TIMEOUT)
        if created.exit_code != 0:
            raise WorkspaceUnavailableError(
                f'bubblewrap could not make {ssh_dir} read-only inside the sandbox: {created.stderr.strip()}'
            )
        self._ssh_guard = ('--ro-bind', ssh_dir, ssh_dir)
        return self._ssh_guard

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
