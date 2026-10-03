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

_LOGIN_PATHS = ('.ssh/', '.config/fish/', '.bashrc', '.zshenv', '.cshrc', '.tcshrc', '.pam_environment')
"""Paths under `$HOME` that the host reads or runs when the next SSH connection logs in, before the command.

`sshd` runs `~/.ssh/rc` and reads `~/.ssh/environment` and `~/.pam_environment`; the login shell runs the
command with `-c`, which makes bash run `~/.bashrc`, zsh `~/.zshenv`, csh and tcsh `~/.cshrc` or `~/.tcshrc`,
and fish its `~/.config/fish`. A trailing `/` marks a directory.
"""

_ENSURE_LOGIN_PATHS = r"""umask 077
set -C
for entry do
  path=${entry#?:}
  if [ -L "$path" ]; then
    printf '%s is a symbolic link\n' "$path" >&2
    exit 2
  fi
  case $entry in
    [pd]:*)
      if [ ! -d "$path" ]; then
        if [ -e "$path" ]; then
          printf '%s is not a directory\n' "$path" >&2
          exit 2
        fi
        /bin/mkdir -- "$path" || exit 1
      fi ;;
    f:*.tcshrc)
      [ -e "$path" ] || printf 'source ~/.cshrc\n' > "$path" || exit 1 ;;
    f:*)
      [ -e "$path" ] || : > "$path" || exit 1 ;;
  esac
  case $entry in
    [df]:*)
      linked=$(/usr/bin/find "$path" '(' -type l -o '(' ! -type d -links +1 ')' ')' -print) || exit 1
      if [ -n "$linked" ]; then
        printf '%s is a symbolic or hard link\n' "$linked" >&2
        exit 2
      fi ;;
  esac
done
"""
"""Create the missing login paths in the writable directory, parents first, and check what they hold.

Each argument is `p:` for a parent directory, `d:` for a protected directory, or `f:` for a protected file.
They are then mounted (parents writable, the rest read-only), so a sandboxed command cannot plant one for
the next connection, and `bwrap` cannot mount over a path that doesn't exist. A new file is empty, except
`~/.tcshrc`: tcsh reads `~/.cshrc` only without one, so the new one sources it. A symlink at any of these
paths, or a symlink or hard-linked file inside a protected one, is a failure: a read-only mount doesn't stop
a command from replacing the link or writing through the target's other name. So is anything but a
directory where one is expected. `set -C` keeps a symlink planted after the check from redirecting a new
file, and `/usr/bin/find` is absolute so a writable `PATH` entry can't supply it.
"""


_CANONICAL_HOME = r"""case "$HOME" in
  /*) ;;
  *) exit 1 ;;
esac
cd -P -- "$HOME" || exit 1
pwd -P
"""
"""`$HOME` as a canonical absolute path, using only shell builtins."""

_CANONICAL_ZDOTDIR = r"""case "$ZDOTDIR" in
  '') exit 0 ;;
  /*) ;;
  *) exit 1 ;;
esac
cd -P -- "$ZDOTDIR" || exit 1
pwd -P
"""
"""Canonical `$ZDOTDIR` if set and absolute, empty if unset, or exit 1 if relative.

`ZDOTDIR` overrides `$HOME` for zsh startup files, so `sshd_config SetEnv` can direct zsh to read
`<ZDOTDIR>/.zshenv` from the writable working directory before `bwrap` starts. This resolves it the
same way `_CANONICAL_HOME` resolves `$HOME`, so `_login_path_mounts` can protect it.
"""

_PROBE_TIMEOUT = 30.0
"""Seconds to wait for the check that `bwrap` can start a sandbox at all."""


def _contains(root: str, path: str) -> bool:
    """Whether the normalized absolute `path` is `root` or inside it."""
    return root == '/' or path == root or path.startswith(root + '/')


class _SandboxedCommands(WorkspaceBackend, SupportsCommands):
    """A backend that runs every command in the sandbox, for core's shell-based file methods to build on."""

    def __init__(self, sandbox: BubblewrapWorkspace):
        self._sandbox = sandbox

    @property
    def ref(self) -> WorkspaceRef | None:
        # `WorkspaceBackend` requires it, but the sandbox routes only file operations through this backend.
        return self._sandbox.ref  # pragma: no cover

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
    files the next SSH login runs on the host, such as `~/.ssh/rc` or `~/.bashrc`, they are bind-mounted
    read-only (and created if missing), so a command cannot run code outside the sandbox through them.

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
        self._login_guard: tuple[str, ...] | None = None

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
            *await self._login_path_mounts(working_dir),
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

    async def _login_path_mounts(self, working_dir: str) -> tuple[str, ...]:
        """Bind the `_LOGIN_PATHS` that the writable directory contains read-only.

        The next workspace call opens a new connection, so a command that can write one of them runs on the
        host, outside `bwrap`. Each directory between the working directory and a protected path is bound
        onto itself, so it is a mount point that a command cannot rename to swap in its own copy.
        `bwrap_args` come after this and can override it.
        """
        if self._login_guard is not None:
            return self._login_guard
        # `cd -P` and `pwd -P` are builtins. `Workspace.realpath` over SSH runs `readlink`, `wc` and
        # `base64` from `PATH`, which a writable directory on that `PATH` could supply.
        reported = await self.wrapped.run(['/bin/sh', '-c', _CANONICAL_HOME], timeout=_PROBE_TIMEOUT)
        home = reported.stdout.removesuffix('\n')
        if reported.exit_code != 0 or not posixpath.isabs(home):
            raise WorkspaceUnavailableError(
                "bubblewrap could not read the host account's home directory, so it cannot keep its "
                'login files out of the writable sandbox'
            )
        # ZDOTDIR overrides $HOME for zsh startup files. If set via sshd_config SetEnv to a directory inside
        # the writable working directory, zsh would execute <ZDOTDIR>/.zshenv before bwrap starts.
        zdotdir_reported = await self.wrapped.run(['/bin/sh', '-c', _CANONICAL_ZDOTDIR], timeout=_PROBE_TIMEOUT)
        zdotdir = zdotdir_reported.stdout.removesuffix('\n')
        if zdotdir_reported.exit_code != 0:
            raise WorkspaceUnavailableError(
                'bubblewrap found a relative ZDOTDIR, which zsh would resolve against the working directory; '
                'this cannot be made safe inside the sandbox'
            )
        root = posixpath.normpath(working_dir)
        # Path to kind, as `_ENSURE_LOGIN_PATHS` takes them: `p` for a parent pinned in place by binding it onto
        # itself, `d` for a protected directory, `f` for a protected file.
        kinds: dict[str, str] = {}
        protected = [
            (posixpath.join(home, name.rstrip('/')), 'd' if name.endswith('/') else 'f') for name in _LOGIN_PATHS
        ]
        if zdotdir and zdotdir != home:
            protected.append((posixpath.join(zdotdir, '.zshenv'), 'f'))
        for path, kind in protected:
            if not _contains(root, path):
                continue
            kinds[path] = kind
            parent = posixpath.dirname(path)
            while parent != root and _contains(root, parent):
                kinds.setdefault(parent, 'p')
                parent = posixpath.dirname(parent)
        # Sorted, a parent comes before what's inside it, both to be created and to be mounted.
        paths = sorted(kinds)
        if paths:
            entries = [f'{kinds[path]}:{path}' for path in paths]
            created = await self.wrapped.run(
                ['/bin/sh', '-c', _ENSURE_LOGIN_PATHS, 'sh', *entries], timeout=_PROBE_TIMEOUT
            )
            if created.exit_code != 0:
                raise WorkspaceUnavailableError(
                    f'bubblewrap could not make the login files in {home} read-only inside the sandbox: '
                    f'{created.stderr.strip()}'
                )
        self._login_guard = tuple(
            arg for path in paths for arg in ('--bind' if kinds[path] == 'p' else '--ro-bind', path, path)
        )
        return self._login_guard

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
