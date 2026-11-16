"""A local implementation of the [workspace backend protocol][pydantic_ai.workspaces.WorkspaceBackend].

[`LocalWorkspaceBackend`][pydantic_ai.workspaces.LocalWorkspaceBackend] runs commands as plain host
subprocesses — it **isolates nothing**.
"""

from __future__ import annotations as _annotations

import os
import signal
from collections.abc import Mapping
from pathlib import Path
from subprocess import DEVNULL, PIPE

import anyio
import anyio.abc

from ._host import HostWorkspaceBackend
from .protocol import (
    SupportsCommands,
    SupportsFilesystem,
    SupportsRealpath,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceRef,
)

__all__ = ('LocalWorkspaceBackend',)

# Not secrets, and without them commands miss the host's tools and the user's configuration.
_INHERITED_ENV = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'LC_CTYPE')


class LocalWorkspaceBackend(
    HostWorkspaceBackend, WorkspaceBackend, SupportsCommands, SupportsFilesystem, SupportsRealpath
):
    """Run commands as subprocesses on this machine and use its filesystem (POSIX only).

    This isolates nothing: commands and absolute paths reach anywhere this process can. Commands
    inherit only `PATH`, `HOME` and locale (`LANG`, `LC_ALL`, `LC_CTYPE`), so they find the host's
    tools and use its text encoding without inheriting arbitrary secrets.
    Background jobs outlive `run()`; redirect their output to avoid waiting up to two seconds
    for inherited output pipes. The caller manages those jobs when the host exits.
    The directory is the environment: its [`ref`][pydantic_ai.workspaces.LocalWorkspaceBackend.ref]
    exists from construction, and the first operation raises
    [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] if it is missing.
    On Windows, use [`WindowsWorkspaceBackend`][pydantic_ai.workspaces.WindowsWorkspaceBackend].

    Args:
        working_dir: Where commands start and relative paths resolve; `~` is expanded and a relative
            path is taken from the current directory. The caller creates and removes it.
        env: Environment variables for every command, on top of the inherited `PATH`, `HOME`, `LANG`,
            `LC_ALL` and `LC_CTYPE`; the per-call `env` goes on top.
    """

    _provider = 'local'
    _inherited_env = _INHERITED_ENV

    def __init__(self, working_dir: str | Path, *, env: Mapping[str, str] | None = None):
        if os.name != 'posix':
            raise NotImplementedError(
                '`LocalWorkspaceBackend` only supports POSIX platforms: its timeout contract kills the whole '
                'process group. On Windows, use `WindowsWorkspaceBackend`; elsewhere, attach a container- or '
                'VM-based workspace instead.'
            )
        super().__init__(working_dir, env=env)

    @property
    def ref(self) -> WorkspaceRef:
        """`WorkspaceRef(provider='local', id=<absolute working_dir>)`, available from construction."""
        return self._ref

    async def _spawn(self, command: WorkspaceCommand, working_dir: Path, env: Mapping[str, str]) -> anyio.abc.Process:
        return await anyio.open_process(
            command, cwd=working_dir, env=env, stdin=DEVNULL, stdout=PIPE, stderr=PIPE, start_new_session=True
        )

    def _kill_tree(self, process: anyio.abc.Process) -> tuple[PermissionError | None, PermissionError | None]:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError as error:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except PermissionError as child_denial:
                return error, child_denial
            return error, None
        return None, None

    def _open_for_read(self, target: Path) -> int:
        # O_NONBLOCK lets us inspect FIFOs and devices without opening a blocking stream.
        return os.open(target, os.O_RDONLY | os.O_NONBLOCK)
