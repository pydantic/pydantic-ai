"""A [workspace backend][pydantic_ai.workspaces.WorkspaceBackend] on a remote host, reached with the `ssh` client."""

from __future__ import annotations as _annotations

import os
import posixpath
import re
import secrets
import shlex
from collections.abc import Mapping, Sequence

import anyio

from pydantic_ai.workspaces import (
    CommandResult,
    LocalWorkspaceBackend,
    SupportsCommands,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._workspace_provider import check_timeout

__all__ = ('SSHWorkspaceBackend',)

# The remote wrapper script reports on stderr that it reached the working directory, and then whether
# the directory outlived the command. Without them, ssh's own failures (exit 255) look like results.
_READY = '__pydantic_ai_ssh_ready__\n'
_GONE = '\n__pydantic_ai_ssh_gone__\n'

_JOB_TAG = '__pydantic_ai_ssh_job_'
"""Starts every remote script, followed by a random suffix, so a later connection can find the command's processes."""

_STOP = r"""me=$(ps -o pgid= -p $$ | tr -d ' ')
groups=$(ps -A -o pgid=,args= | grep -F -e "$1" | awk -v me="$me" '$1 != me { print $1 }' | sort -u)
[ -n "$groups" ] || exit 0
for group in $groups; do kill -s TERM -- "-$group" 2> /dev/null; done
(sleep 1; for group in $groups; do kill -s KILL -- "-$group" 2> /dev/null; done) < /dev/null > /dev/null 2>&1 &
exit 0"""
"""Stop the process groups whose command line holds the tag `$1`, leaving out this script's own group.

`sshd` starts each command in a session of its own, so its group is the command's; a detached command
that started a session of its own (the harness `Shell`'s jobs) is in another group and keeps running.
`kill -s SIG --` is the form every POSIX shell's `kill` accepts; dash rejects `kill -TERM -- -<group>`.
The `SIGKILL` for groups that outlast `SIGTERM` comes a second later in the background, with its output
closed so `sshd` doesn't wait for it, so a stop costs one round trip.
"""

_STOP_TIMEOUT = 2.0
"""Bounds how late a stopped command's timeout or cancellation is raised when the host stops answering.

The same grace period `LocalWorkspaceBackend` gives reaping a killed process: past it, the stop gives up
and the remote command may keep running.
"""


_ENV_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_CLIENT_ENV = ('SSH_AUTH_SOCK',)
"""Passed to the local `ssh` process, on top of `LocalWorkspaceBackend`'s, so it can use the SSH agent."""


def _after_ready(stderr: str) -> str:
    return stderr.partition(_READY)[2]


class SSHWorkspaceBackend(WorkspaceBackend, SupportsCommands):
    """Run commands on a remote host with the system's OpenSSH `ssh` client.

    Authentication, host keys, ports and jump hosts come from your SSH configuration (`~/.ssh/config`)
    and agent; `ssh` never prompts, so a missing key fails instead of waiting for a password. File
    operations run as shell commands on the host (see [Writing a backend](https://pydantic.dev/docs/ai/core-concepts/workspace/#writing-a-backend)),
    which needs a POSIX `sh` there.
    The remote directory is the environment: the first operation raises
    [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] if it is missing or
    the host can't be reached. On a timeout or cancellation, a second connection stops the command's
    process group on the host. A background process that keeps the command's output open holds the
    call until it exits, as `sshd` waits for the output to close.

    Args:
        destination: The host, as you'd pass it to `ssh`: `'user@host'`, a `Host` alias from your SSH
            configuration, or `'ssh://user@host:port'`.
        working_dir: Where commands start and relative paths resolve, on the remote host; a relative
            path starts in the login directory, which is the default.
        env: Environment variables for every command, on top of the remote login environment; the
            per-call `env` goes on top.
        ssh_args: Extra `ssh` arguments, such as `['-i', key_path]`, placed before the destination.
    """

    def __init__(
        self,
        destination: str,
        *,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        ssh_args: Sequence[str] = (),
    ):
        if not destination or destination.startswith('-'):
            raise ValueError(f'destination must be a host, got {destination!r}')
        if isinstance(ssh_args, str):
            raise TypeError('ssh_args must be a sequence of arguments, not a string')
        self._env = self._checked_env(env or {})
        self._working_dir = None if working_dir is None else posixpath.normpath(working_dir)
        self._resolved_working_dir: str | None = None
        # No password auth, on purpose: a prompt would hang the run until it timed out, a `password`
        # option would put secrets in agent specs, and ssh can only take one through `sshpass` or an
        # askpass helper. `BatchMode=yes` comes first so it wins over `ssh_args`, and keys or
        # `ssh-agent` (via `SSH_AUTH_SOCK`) authenticate instead.
        self._ssh = ['ssh', '-T', '-o', 'BatchMode=yes', *ssh_args, '--', destination]
        # A local subprocess runner: it owns timeouts, output limits and killing `ssh` on cancellation.
        self._runner = LocalWorkspaceBackend(
            '/', env={name: os.environ[name] for name in _CLIENT_ENV if name in os.environ}
        )
        self._ref = WorkspaceRef(
            provider='ssh', id=destination if self._working_dir is None else f'{destination}:{self._working_dir}'
        )

    @property
    def ref(self) -> WorkspaceRef:
        """`WorkspaceRef(provider='ssh', id='<destination>[:<working_dir>]')`, available from construction."""
        return self._ref

    @staticmethod
    def _checked_env(env: Mapping[str, str]) -> dict[str, str]:
        for name in env:
            if not _ENV_NAME.fullmatch(name):
                raise ValueError(f'invalid environment variable name: {name!r}')
        return dict(env)

    async def working_dir(self) -> str:
        return await self._resolve_working_dir(timeout=None)

    async def _resolve_working_dir(self, *, timeout: float | None) -> str:
        if self._resolved_working_dir is None:
            # A relative directory starts in the login directory; `./` keeps a leading `-` from reading as an option.
            directory = '.' if self._working_dir is None else posixpath.join('.', self._working_dir)
            result = await self._remote(directory, 'pwd -P', env={}, timeout=timeout)
            self._resolved_working_dir = result.stdout.removesuffix('\n')
        return self._resolved_working_dir

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
            line = f'/bin/sh -c {shlex.quote(command)}'
        elif shell:
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')
        elif not command:
            raise ValueError('command must not be empty')
        else:
            # A subshell `exec` runs the program itself, never a builtin, and exits 127 when it's missing.
            line = f'(exec {shlex.join(command)})'
        merged_env = {**self._env, **self._checked_env(env or {})}
        # The first command also resolves the working directory, within the same timeout.
        started = anyio.current_time()
        directory = await self._resolve_working_dir(timeout=timeout)
        if timeout is not None:
            remaining = timeout - (anyio.current_time() - started)
            if remaining <= 0:
                raise WorkspaceTimeoutError(
                    f'command timed out after {timeout:g}s while connecting and was not started', stdout='', stderr=''
                )
            timeout = remaining
        return await self._remote(directory, line, env=merged_env, timeout=timeout)

    async def _remote(
        self, directory: str, line: str, *, env: Mapping[str, str], timeout: float | None
    ) -> CommandResult:
        exports = ''.join(f'export {name}={shlex.quote(value)}\n' for name, value in env.items())
        tag = f'{_JOB_TAG}{secrets.token_hex(8)}'
        # A fresh nonce per command: the marker text is otherwise a constant a child of the command can print.
        nonce = secrets.token_hex(16)
        marker = f'__pydantic_ai_ssh_done_{nonce}__'
        script = (
            f': {tag}\n'
            f'cd {shlex.quote(directory)} || exit 1\n'
            # On both streams, so output from `~/.ssh/rc` or a login banner is cut off.
            f"printf '%s' {shlex.quote(_READY)}; printf '%s' {shlex.quote(_READY)} >&2\n"
            f'{exports}__pydantic_ai_dir=$PWD\n'
            f'{line}\n'
            '__pydantic_ai_status=$?\n'
            f'if [ -d "$__pydantic_ai_dir" ]; then printf \'\\n{marker}%d\\n\' "$__pydantic_ai_status" >&2; '
            f"else printf '%s' {shlex.quote(_GONE)} >&2; fi\n"
            'exit "$__pydantic_ai_status"'
        )
        # `ssh` hands the command to the remote login shell. `/bin/sh` is absolute so a writable `PATH` entry
        # cannot supply the shell that interprets this script.
        argv = [*self._ssh, f'/bin/sh -c {shlex.quote(script)}']
        try:
            result = await self._runner.run(argv, timeout=timeout)
        except anyio.get_cancelled_exc_class():
            await self._stop(tag)
            raise
        except WorkspaceTimeoutError as error:
            await self._stop(tag)
            raise WorkspaceTimeoutError(
                str(error), stdout=_after_ready(error.stdout), stderr=_after_ready(error.stderr)
            ) from error
        except WorkspaceOutputLimitError as error:
            await self._stop(tag)
            raise WorkspaceOutputLimitError(
                "SSH workspace output exceeded its 10 MiB safety limit; redirect the command's output to a file "
                'and read part of it instead',
                limit=error.limit,
                stdout=_after_ready(error.stdout),
                stderr=_after_ready(error.stderr),
            ) from error
        before, ready, stderr = result.stderr.partition(_READY)
        if not ready:
            reason = before.strip() or f'`ssh` exited with code {result.exit_code}'
            raise WorkspaceUnavailableError(f'SSH workspace {self._ref.id} is unavailable: {reason}')
        # The marker carries the status because `ssh` uses exit 255 for a lost connection, and a command can too.
        # Any other process exit is the command's: a child can print a marker after the wrapper has written its own.
        pattern = re.compile(rf'\n{re.escape(marker)}(\d+)\n')
        markers = list(pattern.finditer(stderr))
        if not markers:
            if _GONE in stderr:
                raise WorkspaceUnavailableError(f'SSH workspace {self._ref.id}: the working directory was removed')
            raise WorkspaceUnavailableError(f'SSH workspace {self._ref.id}: the connection was lost during the command')
        status = result.exit_code if result.exit_code != 255 else int(markers[0][1])
        return CommandResult(
            exit_code=status,
            stdout=_after_ready(result.stdout),
            stderr=pattern.sub('', stderr),
        )

    async def _stop(self, tag: str) -> None:
        # Killing the local `ssh` leaves the remote command running, so a second connection stops it.
        # Best effort: a host that stalls or can't be reached now has nothing to report.
        with anyio.CancelScope(shield=True):
            try:
                await self._runner.run(
                    [*self._ssh, f'/bin/sh -c {shlex.quote(_STOP)} /bin/sh {tag}'], timeout=_STOP_TIMEOUT
                )
            except WorkspaceError:
                pass
