"""A [smolvm](https://github.com/smol-machines/smolvm) VM on this machine, driven through the `smolvm` CLI.

`SmolSandboxBackend` runs smolvm in this process through smolmachines, which has wheels for macOS on Apple
silicon and Linux only. smolvm itself also runs on Windows x86_64, with the Windows Hypervisor Platform, so this
backend drives its CLI instead, as the smol Rust SDK's local transport does:
https://github.com/smol-machines/smol/blob/main/sdk/rust/src/transport/local.rs

CLI behavior, from smolvm 1.36's `smolvm machine --help` and that transport:

- `machine create` records a machine and `machine start` boots it; it keeps running after the CLI exits.
- `machine exec` exits with the command's exit status and, with `--interactive`, passes its stdin to the command.
  Its `--timeout` kills the command in the VM.
- `machine ls --json` lists machines as objects with `name` and `state`, bare or under `machines`. A `stopped` or
  `created` machine is started, and a `paused` one resumed, as that transport does.
- With `--error-format json`, smolvm reports its own failures as a last stderr line
  `{"error": {"code": ..., "message": ...}}`.

Each command reaches the VM as a script on `machine exec`'s stdin, so neither its arguments nor its environment
are on a command line of this machine, which other processes can read and Windows caps at 32,767 characters.
"""

from __future__ import annotations

import logging
import math
import re
import shlex
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING

import anyio
from anyio import run_process
from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from pydantic_ai.workspaces import (
    CommandResult,
    SupportsCommands,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._workspace_provider import (
    EXEC_SCRIPT,
    absolute_path,
    check_env,
    check_provider,
    check_timeout,
    command_argv,
)

if TYPE_CHECKING:
    from subprocess import CompletedProcess

    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('SmolvmCliBackend', 'delete_machine')

logger = logging.getLogger(__name__)

_NAME_PREFIX = 'pydantic-ai-'
_NAME = re.compile(rf'{_NAME_PREFIX}[0-9a-f]{{32}}')
"""The names `SmolSandboxBackend` gives its VMs too, and so the only machines this backend attaches to or deletes.

`machine exec` reaches any machine of this OS user by name, such as one another tool made with this machine's
folders mounted, so a name from message history must be one this backend could have made.
"""

_SETUP_TIMEOUT = 60.0
"""Seconds the backend's own commands in the VM (`mkdir -p` and `pwd -P`) get."""

_ENGINE_TIMEOUT_GRACE = 5.0
"""Seconds past a command's timeout before smolvm kills it itself, in case killing the CLI here does not."""

_HOST_UNAVAILABLE = frozenset(
    {'UNSUPPORTED_PLATFORM', 'RUNTIME_NOT_INSTALLED', 'KVM_UNAVAILABLE', 'HYPERVISOR_UNAVAILABLE'}
)
"""Codes for a host that can't run a local VM at all, so no retry can succeed."""


class _Failure(TypedDict):
    code: str
    message: str


class _Report(TypedDict):
    error: _Failure


class _Machine(TypedDict):
    name: str
    state: str


class _Listing(TypedDict):
    machines: list[_Machine]


_REPORT = TypeAdapter(_Report)
_LISTING = TypeAdapter(list[_Machine] | _Listing)


def check_name(name: str) -> None:
    """Raise `ValueError` unless `name` is one `SmolvmCliBackend` or `SmolSandboxBackend` gives its VMs."""
    if _NAME.fullmatch(name) is None:
        raise ValueError(
            f'SmolvmCliBackend only uses VMs it created, named {_NAME_PREFIX!r} and 32 hex digits, got {name!r}.'
        )


class SmolvmCliBackend(WorkspaceBackend, SupportsCommands):
    """A smolvm VM on this machine, run through the `smolvm` CLI, as a Pydantic AI `WorkspaceBackend`.

    Building one does no I/O. The first operation boots a new persistent VM, or starts the one `ref` names if it
    is stopped. The backend never deletes the VM; `delete_machine` does.

    Args:
        ref: Identity of an existing VM of this backend's to attach to on first use: its name.
        image: OCI image a new VM runs; `None` boots smolvm's own Alpine Linux root filesystem.
        allow_internet_access: Whether a new VM gets outbound network access, made from this machine's network.
        working_dir: Absolute directory commands start in, created if missing when the backend first reaches the
            VM.
        env: Environment variables every command gets; per-command `env` is layered on top.
        executable: The `smolvm` CLI to run, found on `PATH` when not a path.
    """

    def __init__(
        self,
        *,
        ref: WorkspaceRef | None = None,
        image: str | None = None,
        allow_internet_access: bool = False,
        working_dir: str = '/workspace',
        env: Mapping[str, str] | None = None,
        executable: str = 'smolvm',
    ) -> None:
        if ref is not None:
            check_provider(ref, 'smol')
            check_name(ref.id)
        check_env(env)
        self._ref = ref
        self._image = image
        self._network = allow_internet_access
        absolute_path('working_dir', working_dir)
        self._working_dir = working_dir
        self._env = dict(env or {})
        self._executable = executable
        self._lock = anyio.Lock()
        self._ready = False
        self._resolved_working_dir: str | None = None

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the VM, or `None` before one is created; set before the VM boots, so cleanup can delete it."""
        return self._ref

    async def _vm(self) -> str:
        # Concurrent first uses wait here, so they boot one VM.
        async with self._lock:
            if not self._ready:
                await self._boot()
                self._ready = True
        assert self._ref is not None
        return self._ref.id

    async def _boot(self) -> None:
        if (ref := self._ref) is None:
            ref = self._ref = WorkspaceRef(provider='smol', id=f'{_NAME_PREFIX}{uuid.uuid4().hex}')
            logger.info('Creating smol VM %s', ref.id)
            options = [*(['--net'] if self._network else []), *(['--image', self._image] if self._image else [])]
            try:
                await _check(self._executable, 'create', '--name', ref.id, *options)
            except WorkspaceError:
                # smolvm reported it made no machine, so a retry creates one under a new name.
                self._ref = None
                raise
            await _check(self._executable, 'start', '--name', ref.id)
        else:
            state = await _state(self._executable, ref.id)
            if state is None:
                raise WorkspaceUnavailableError(_gone_message(ref.id))
            if state in ('stopped', 'created'):
                await _check(self._executable, 'start', '--name', ref.id)
            elif state == 'paused':
                await _check(self._executable, 'resume', '--name', ref.id)
            elif state != 'running':
                raise WorkspaceUnavailableError(f'The smol VM {ref.id!r} is {state}.')
        result = await self._exec(
            ref.id, ['mkdir', '-p', self._working_dir], env={}, working_dir=None, timeout=_SETUP_TIMEOUT
        )
        if result.exit_code != 0:
            raise WorkspaceError(
                f'Could not create working_dir {self._working_dir!r} in smol VM {ref.id}: {result.stderr}'
            )

    async def working_dir(self) -> str:
        """The canonical absolute directory commands start in."""
        if self._resolved_working_dir is None:
            result = await self.run(['pwd', '-P'], timeout=_SETUP_TIMEOUT)
            printed = result.stdout.removesuffix('\n')
            if result.exit_code != 0 or not printed.startswith('/'):  # pragma: no cover - a VM without `pwd`
                raise WorkspaceError(
                    'Could not determine the working directory of the smol VM: '
                    f'`pwd -P` exited {result.exit_code} and printed {result.stdout!r}.'
                )
            self._resolved_working_dir = printed
        return self._resolved_working_dir

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run a command in the VM; on timeout or cancellation the CLI is killed, and smolvm kills the command."""
        argv = command_argv(command, shell)
        check_timeout(timeout)
        check_env(env)
        name = await self._vm()
        return await self._exec(
            name, argv, env={**self._env, **(env or {})}, working_dir=self._working_dir, timeout=timeout
        )

    async def _exec(
        self, name: str, argv: list[str], *, env: dict[str, str], working_dir: str | None, timeout: float | None
    ) -> CommandResult:
        # `env` sets the variables, as a name need not be one the shell can assign.
        assignments = [f'{key}={value}' for key, value in env.items()]
        script = shlex.join(['exec', 'env', '--', *assignments, '/bin/sh', '-c', EXEC_SCRIPT, 'sh', *argv]) + '\n'
        options = ['--name', name, '--interactive']
        if working_dir is not None:
            options += ['--workdir', working_dir]
        if timeout is not None:
            options += ['--timeout', f'{math.ceil((timeout + _ENGINE_TIMEOUT_GRACE) * 1000)}ms']
        result: CompletedProcess[bytes] | None = None
        with anyio.move_on_after(timeout):
            result = await _cli(self._executable, 'exec', *options, '--', '/bin/sh', '-s', stdin=script.encode())
        if result is None:
            raise WorkspaceTimeoutError(f'Command timed out after {timeout:g} seconds')
        if result.returncode != 0 and _failure(result.stderr) is not None:
            # smolvm's own failure, or the command's output looking like one: only a VM that is gone tells them apart.
            if (state := await _state(self._executable, name)) != 'running':
                raise WorkspaceUnavailableError(
                    _gone_message(name) if state is None else f'The smol VM {name!r} is {state}.'
                )
        return CommandResult(
            exit_code=result.returncode,
            stdout=result.stdout.decode(errors='replace'),
            stderr=result.stderr.decode(errors='replace'),
        )


async def delete_machine(name: str, *, executable: str = 'smolvm') -> None:
    """Delete a VM `SmolvmCliBackend` created, and its disk; one already gone returns quietly."""
    check_name(name)
    result = await _cli(executable, 'delete', '--name', name, '--force')
    if result.returncode != 0 and ((failure := _failure(result.stderr)) is None or failure['code'] != 'NOT_FOUND'):
        raise _error('delete', result.stderr)


async def _cli(executable: str, action: str, *args: str, stdin: bytes | None = None) -> CompletedProcess[bytes]:
    command = [executable, 'machine', action, '--error-format', 'json', *args]
    try:
        return await run_process(command, input=stdin, check=False)
    except FileNotFoundError as error:
        raise WorkspaceUnavailableError(
            f'{executable} was not found: install smolvm from https://github.com/smol-machines/smolvm/releases.'
        ) from error


async def _check(executable: str, action: str, *args: str) -> bytes:
    result = await _cli(executable, action, *args)
    if result.returncode != 0:
        raise _error(action, result.stderr)
    return result.stdout


async def _state(executable: str, name: str) -> str | None:
    """The VM's state, such as `running` or `stopped`; `None` when there is no VM of that name."""
    listing = _LISTING.validate_json(await _check(executable, 'ls', '--json'))
    machines = listing if isinstance(listing, list) else listing['machines']
    return next((machine['state'] for machine in machines if machine['name'] == name), None)


def _failure(stderr: bytes) -> _Failure | None:
    """The failure smolvm reported, from the last line of its stderr."""
    lines = stderr.strip().splitlines()
    try:
        return _REPORT.validate_json(lines[-1])['error'] if lines else None
    except ValidationError:
        return None


def _error(action: str, stderr: bytes) -> WorkspaceError:
    failure = _failure(stderr) or _Failure(code='', message=stderr.decode(errors='replace').strip())
    if failure['code'] in _HOST_UNAVAILABLE:
        return WorkspaceUnavailableError(
            f'This host cannot run a local smol VM ({failure["code"]}): {failure["message"]}'
        )
    return WorkspaceError(f'smolvm machine {action} failed: {failure["message"]}')


def _gone_message(name: str) -> str:
    return f"The smol VM {name!r} no longer exists. Pass `workspace='new'` to start a fresh VM."
