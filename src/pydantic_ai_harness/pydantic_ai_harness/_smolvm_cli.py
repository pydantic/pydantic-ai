"""A [smolvm](https://github.com/smol-machines/smolvm) VM on this machine, driven through the `smolvm` CLI.

`SmolSandboxBackend` runs smolvm in this process through smolmachines, which has wheels for macOS on Apple
silicon and Linux only. smolvm itself also runs on Windows x86_64, with the Windows Hypervisor Platform, so this
backend drives its CLI instead, as the smol Rust SDK's local transport does:
https://github.com/smol-machines/smol/blob/main/sdk/rust/src/transport/local.rs

CLI behavior, from smolvm 1.36's `smolvm machine --help` and that transport, and verified 2026-11-22 by the live
tier (`TestLiveSmolvmCliBackend`) against smolvm 1.36.1 on GitHub's `ubuntu-latest` with KVM; not yet on Windows:

- `machine create` records a machine and `machine start` boots it; it keeps running after the CLI exits.
  `machine stop` of a stopped machine exits 0, and smolvm's own failures name a missing machine `NOT_FOUND`.
- `machine exec` exits with the command's exit status and, with `--interactive`, passes its stdin to the command.
  Its `--timeout` kills the command in the VM.
- `machine ls --json` lists machines as objects with `name` and `state` (bare in 1.36; under `machines` in some
  versions the Rust transport reads). A `stopped` or
  `created` machine is started, and a `paused` one resumed, as that transport does.
- With `--error-format json`, smolvm reports its own failures as a last stderr line
  `{"error": {"code": ..., "message": ...}}`.

Each command reaches the VM as a script on `machine exec`'s stdin, so neither its arguments nor its environment
are on a command line of this machine, which other processes can read and Windows caps at 32,767 characters.
A command whose output passes 10 MiB is killed, as `SmolSandboxBackend` stops one. The CLI is found only in
absolute `PATH` directories: Windows would otherwise run a `smolvm.exe` in the current
directory, such as one a checkout holds.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shlex
import subprocess
import sys
import uuid
from collections.abc import Mapping
from contextlib import suppress
from typing import TYPE_CHECKING

import anyio
from anyio import open_process
from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from pydantic_ai.workspaces import (
    CommandResult,
    SupportsCommands,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceOutputLimitError,
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
    from anyio.abc import ByteReceiveStream, ByteSendStream

    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('NAME', 'NAME_PREFIX', 'SmolvmCliBackend', 'delete_machine', 'find_cli')

logger = logging.getLogger(__name__)

NAME_PREFIX = 'pydantic-ai-'
NAME = re.compile(rf'{NAME_PREFIX}[0-9a-f]{{32}}')
"""The names this backend and `SmolSandboxBackend` give their VMs, and so the only refs either attaches to or deletes.

A local VM's id is its name, and smolvm opens any VM of this OS user by name, so without this check a ref
from message history could reach a VM made by another tool, with host mounts or network. The name is
a namespace, not proof of origin: any process of this OS user could use it, and smolvm can't report
an existing VM's mounts or network, so a ref is still trusted input. Defined here, as `SmolSandboxBackend`'s
module needs smolmachines, which Windows lacks.
"""

_SETUP_TIMEOUT = 60.0
"""Seconds the backend's own commands in the VM (`mkdir -p` and `pwd -P`) get."""

_ENGINE_TIMEOUT_GRACE = 5.0
"""Seconds past a command's timeout before smolvm kills it itself, in case killing the CLI here does not."""

_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
"""What a command's stdout and stderr may hold together before it is killed, as `SmolSandboxBackend` caps them."""
_OUTPUT_PREVIEW_BYTES = 64 * 1024

_HOST_UNAVAILABLE = frozenset(
    {'UNSUPPORTED_PLATFORM', 'RUNTIME_NOT_INSTALLED', 'KVM_UNAVAILABLE', 'HYPERVISOR_UNAVAILABLE'}
)
"""Codes for a host that can't run a local VM at all, so no retry can succeed."""

_STARTABLE = frozenset({'stopped', 'created', 'paused'})
"""States of a VM that `machine start`, or `machine resume` when paused, brings back, as the Rust SDK's transport does."""


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
    if NAME.fullmatch(name) is None:
        raise ValueError(
            f'SmolvmCliBackend only uses VMs it created, named {NAME_PREFIX!r} and 32 hex digits, got {name!r}.'
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
    """

    def __init__(
        self,
        *,
        ref: WorkspaceRef | None = None,
        image: str | None = None,
        allow_internet_access: bool = False,
        working_dir: str = '/workspace',
        env: Mapping[str, str] | None = None,
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
            ref = self._ref = WorkspaceRef(provider='smol', id=f'{NAME_PREFIX}{uuid.uuid4().hex}')
            logger.info('Creating smol VM %s', ref.id)
            options = [*(['--net'] if self._network else []), *(['--image', self._image] if self._image else [])]
            try:
                await _check('create', '--name', ref.id, *options)
            except WorkspaceError:
                # smolvm reported it made no machine, so a retry creates one under a new name.
                self._ref = None
                raise
            await _check('start', '--name', ref.id)
        else:
            state = await _state(ref.id)
            if state is None:
                raise WorkspaceUnavailableError(_gone_message(ref.id))
            if state in _STARTABLE:
                await _check('resume' if state == 'paused' else 'start', '--name', ref.id)
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
            # Only a VM without `pwd` fails here.
            if result.exit_code != 0 or not printed.startswith('/'):  # pragma: no cover
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
        """Run a command in the VM; on timeout or cancellation, or past the output cap, the CLI is killed.

        With a `timeout`, smolvm also kills the command itself shortly after. Without one, smolvm ends a command
        whose CLI went away: the live tier's cancellation rule passed against smolvm 1.36.1 on Linux.
        """
        argv = command_argv(command, shell)
        check_timeout(timeout)
        check_env(env)
        name = await self._vm()
        return await self._exec(
            name, argv, env={**self._env, **(env or {})}, working_dir=self._working_dir, timeout=timeout
        )

    async def stop(self) -> None:
        """Stop the VM, writing its disk back, if this backend reached it; its next command starts it again."""
        async with self._lock:
            if self._ready:
                assert self._ref is not None
                result = await _cli('stop', '--name', self._ref.id)
                # A VM another backend stopped meanwhile, or deleted, is already down.
                if result.returncode != 0 and (await _state(self._ref.id) or 'stopped') not in _STARTABLE:
                    # Still reached, so a retry stops it.
                    raise _error('stop', result.stderr)
                self._ready = False

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
        result: subprocess.CompletedProcess[bytes] | None = None
        with anyio.move_on_after(timeout):
            result = await _cli('exec', *options, '--', '/bin/sh', '-s', stdin=script.encode())
        if result is None:
            raise WorkspaceTimeoutError(f'Command timed out after {timeout:g} seconds')
        if result.returncode != 0 and _failure(result.stderr) is not None:
            # smolvm's own failure, or the command's output looking like one: only a VM that is gone tells them apart.
            state = await _state(name)
            if state is None:
                raise WorkspaceUnavailableError(_gone_message(name))
            if state in _STARTABLE:
                # Stopped by something else, such as another CLAI closing it, maybe during the command, so it is not
                # run again here; the next command starts the VM.
                self._ready = False
                raise WorkspaceError(f'The smol VM {name!r} was {state} before the command ended; run it again.')
            if state != 'running':
                raise WorkspaceUnavailableError(f'The smol VM {name!r} is {state}.')
        return CommandResult(
            exit_code=result.returncode,
            stdout=result.stdout.decode(errors='replace'),
            stderr=result.stderr.decode(errors='replace'),
        )


async def delete_machine(name: str) -> None:
    """Delete a VM `SmolvmCliBackend` created, and its disk; one already gone returns quietly."""
    check_name(name)
    result = await _cli('delete', '--name', name, '--force')
    if result.returncode != 0 and not _is_gone(result.stderr):
        raise _error('delete', result.stderr)


def _is_gone(stderr: bytes) -> bool:
    return (failure := _failure(stderr)) is not None and failure['code'] == 'NOT_FOUND'


async def _cli(action: str, *args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    """Run `smolvm machine ACTION`, killing it on cancellation or once its output passes `_MAX_OUTPUT_BYTES`."""
    if (path := await anyio.to_thread.run_sync(find_cli)) is None:
        raise WorkspaceUnavailableError(
            'smolvm was not found on PATH: install it from https://github.com/smol-machines/smolvm/releases.'
        )
    command = [path, 'machine', action, '--error-format', 'json', *args]
    process = await open_process(command, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL)
    stdout: list[bytes] = []
    stderr: list[bytes] = []
    size = 0

    async def drain(stream: ByteReceiveStream | None, chunks: list[bytes]) -> None:
        nonlocal size
        assert stream is not None
        async for chunk in stream:
            # Past the cap the CLI is killed, and the rest read and dropped: a pipe left unread is never closed.
            if size <= _MAX_OUTPUT_BYTES:
                size += len(chunk)
                chunks.append(chunk)
                if size > _MAX_OUTPUT_BYTES:
                    process.kill()

    try:
        async with anyio.create_task_group() as tasks:
            if stdin is not None:
                tasks.start_soon(_feed, process.stdin, stdin)
            tasks.start_soon(drain, process.stdout, stdout)
            tasks.start_soon(drain, process.stderr, stderr)
        returncode = await process.wait()
    except BaseException:
        process.kill()
        with anyio.CancelScope(shield=True):
            await process.aclose()
        raise
    await process.aclose()
    if size > _MAX_OUTPUT_BYTES:
        raise WorkspaceOutputLimitError(
            'smol command output exceeded the 10 MiB limit; redirect it to a file and read part of it instead',
            limit=_MAX_OUTPUT_BYTES,
            stdout=b''.join(stdout)[:_OUTPUT_PREVIEW_BYTES].decode(errors='replace'),
            stderr=b''.join(stderr)[:_OUTPUT_PREVIEW_BYTES].decode(errors='replace'),
        )
    return subprocess.CompletedProcess(command, returncode, b''.join(stdout), b''.join(stderr))


async def _feed(stream: ByteSendStream | None, data: bytes) -> None:
    assert stream is not None
    # The CLI may exit, or be killed, before it reads everything, which asyncio can report as `ConnectionResetError`.
    with suppress(anyio.BrokenResourceError, anyio.ClosedResourceError, OSError):
        await stream.send(data)
        await stream.aclose()


def find_cli() -> str | None:
    """The `smolvm` CLI in an absolute `PATH` directory, never the current directory as `shutil.which` on Windows."""
    # Never a batch file, which Windows runs through `cmd.exe`.
    extensions = ['.exe', '.com'] if sys.platform == 'win32' else ['']
    for directory in os.environ.get('PATH', '').split(os.pathsep):
        if os.path.isabs(directory):
            for extension in extensions:
                candidate = os.path.join(directory, 'smolvm' + extension)
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    return candidate
    return None


async def _check(action: str, *args: str) -> bytes:
    result = await _cli(action, *args)
    if result.returncode != 0:
        raise _error(action, result.stderr)
    return result.stdout


async def _state(name: str) -> str | None:
    """The VM's state, such as `running` or `stopped`; `None` when there is no VM of that name."""
    listing = _LISTING.validate_json(await _check('ls', '--json'))
    machines = listing if isinstance(listing, list) else listing['machines']
    return next((machine['state'] for machine in machines if machine['name'] == name), None)


def _failure(stderr: bytes) -> _Failure | None:
    """The failure smolvm reported, from the last line of its stderr, which up to 10 MiB of a command's precede."""
    last = stderr.rstrip().rpartition(b'\n')[2]
    try:
        return _REPORT.validate_json(last)['error'] if last else None
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
