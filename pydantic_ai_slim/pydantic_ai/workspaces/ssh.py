"""A [workspace backend][pydantic_ai.workspaces.WorkspaceBackend] on a remote host, reached with the `ssh` client."""

from __future__ import annotations as _annotations

import os
import posixpath
import re
import shlex
from collections.abc import Mapping, Sequence

from .local import LocalWorkspaceBackend
from .protocol import (
    CommandResult,
    SupportsCommands,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
    validate_timeout,
)

__all__ = ('SSHWorkspaceBackend',)

# The remote wrapper script reports on stderr that it reached the working directory, and then whether
# the directory outlived the command. Without them, ssh's own failures (exit 255) look like results.
_READY = '__pydantic_ai_ssh_ready__\n'
_DONE = '\n__pydantic_ai_ssh_done__\n'
_GONE = '\n__pydantic_ai_ssh_gone__\n'

_ENV_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_CLIENT_ENV = ('SSH_AUTH_SOCK',)
"""Passed to the local `ssh` process, on top of `LocalWorkspaceBackend`'s, so it can use the SSH agent."""


def _after_ready(stderr: str) -> str:
    return stderr.partition(_READY)[2]


class SSHWorkspaceBackend(WorkspaceBackend, SupportsCommands):
    """Run commands on a remote host with the system's OpenSSH `ssh` client.

    Authentication, host keys, ports and jump hosts come from your SSH configuration (`~/.ssh/config`)
    and agent; `ssh` never prompts, so a missing key fails instead of waiting for a password. File
    operations run as shell commands on the host (see [Writing a backend](../workspace.md#writing-a-backend)),
    which needs a POSIX `sh` there.
    The remote directory is the environment: the first operation raises
    [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] if it is missing or
    the host can't be reached.

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
        self._client = LocalWorkspaceBackend(
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
        if self._resolved_working_dir is None:
            # A relative directory starts in the login directory; `./` keeps a leading `-` from reading as an option.
            directory = '.' if self._working_dir is None else posixpath.join('.', self._working_dir)
            result = await self._remote(directory, 'pwd -P', env={}, timeout=None)
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
        validate_timeout(timeout)
        if isinstance(command, str):
            if not shell:
                raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
            line = f'sh -c {shlex.quote(command)}'
        elif shell:
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')
        else:
            # A subshell `exec` runs the program itself, never a builtin, and exits 127 when it's missing.
            line = f'(exec {shlex.join(command)})'
        merged_env = {**self._env, **self._checked_env(env or {})}
        return await self._remote(await self.working_dir(), line, env=merged_env, timeout=timeout)

    async def _remote(
        self, directory: str, line: str, *, env: Mapping[str, str], timeout: float | None
    ) -> CommandResult:
        exports = ''.join(f'export {name}={shlex.quote(value)}\n' for name, value in env.items())
        script = (
            f'cd {shlex.quote(directory)} || exit 1\n'
f"printf '%s' {shlex.quote(_READY)} >&2\n"
            f'{exports}__pydantic_ai_dir=$PWD\n'
            f'{line}\n'
            '__pydantic_ai_status=$?\n'
            f'if [ -d "$__pydantic_ai_dir" ]; then printf \'%s\' {shlex.quote(_DONE)} >&2; '
            f"else printf '%s' {shlex.quote(_GONE)} >&2; fi\n"
            'exit "$__pydantic_ai_status"'
        )
        # `ssh` hands the command to the remote login shell, so run `sh` there for POSIX semantics.
        argv = [*self._ssh, f'sh -c {shlex.quote(script)}']
        try:
            result = await self._client.run(argv, timeout=timeout)
        except WorkspaceTimeoutError as error:
            raise WorkspaceTimeoutError(str(error), stdout=error.stdout, stderr=_after_ready(error.stderr)) from error
        except WorkspaceOutputLimitError as error:
            raise WorkspaceOutputLimitError(
                "SSH workspace output exceeded its 10 MiB safety limit; redirect the command's output to a file "
                'and read part of it instead',
                limit=error.limit,
                stdout=error.stdout,
                stderr=_after_ready(error.stderr),
            ) from error
        before, ready, stderr = result.stderr.partition(_READY)
        if not ready:
            reason = before.strip() or f'`ssh` exited with code {result.exit_code}'
            raise WorkspaceUnavailableError(f'SSH workspace {self._ref.id} is unavailable: {reason}')
        if stderr.endswith(_GONE):
            raise WorkspaceUnavailableError(f'SSH workspace {self._ref.id}: the working directory was removed')
        if not stderr.endswith(_DONE):
            raise WorkspaceUnavailableError(f'SSH workspace {self._ref.id}: the connection was lost during the command')
        return CommandResult(exit_code=result.exit_code, stdout=result.stdout, stderr=stderr.removesuffix(_DONE))
