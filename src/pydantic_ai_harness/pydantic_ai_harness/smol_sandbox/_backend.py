"""A local [smolvm](https://github.com/smol-machines/smolvm) microVM behind Pydantic AI's `WorkspaceBackend` protocol.

SDK behavior verified 2026-10-12 against smolmachines 1.22.2 and 1.27.0 on macOS (Apple silicon), see
https://github.com/smol-machines/smol/tree/main/sdk/python. Re-check these before changing acquisition,
command, or error handling:

- `AsyncMachine` runs every blocking call with `asyncio.to_thread`, so the SDK needs asyncio.
- Without an image, a VM boots smolvm's own Alpine root filesystem with BusyBox, runs commands as root,
  and starts them in `/`.
- `exec_stream` yields `{'kind': 'stdout' | 'stderr', 'data': str}` events, then
  `{'kind': 'exit', 'exit_code': int}`, or `{'kind': 'error', 'message': '[CODE] ...'}` instead of
  raising. Closing the generator kills the command and the processes it started in the VM.
- A missing argv program exits 255 with an engine message rather than 127.
- Deleting a VM during a command ends the stream with an `io operation failed: connection closed` error
  event; the old handle's `state()` then reports `stopped`, and `connect` raises `NOT_FOUND`.
- A persistent VM outlives its process and `connect` boots it again, but only `stop()` (or `sync` in the
  guest) flushes its disk: a VM still running when its process exits loses its unflushed writes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from typing import TYPE_CHECKING, cast

import anyio

from pydantic_ai.exceptions import UserError
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
from pydantic_ai_harness._workspace_provider import absolute_path, check_timeout, command_argv, running_on_asyncio

if TYPE_CHECKING:
    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('SmolSandboxBackend',)

try:
    import smol
except ImportError as _import_error:
    raise ImportError(
        'Install `pydantic-ai-harness[smol]` to use SmolSandbox. smolmachines has wheels for macOS on Apple silicon '
        'and for Linux on x86_64 and aarch64 (glibc 2.34 or newer).'
    ) from _import_error

logger = logging.getLogger(__name__)

_HOST_UNAVAILABLE = frozenset(
    {'UNSUPPORTED_PLATFORM', 'RUNTIME_NOT_INSTALLED', 'KVM_UNAVAILABLE', 'HYPERVISOR_UNAVAILABLE'}
)
"""Codes for a host that can't run a local VM at all, so no retry can succeed."""

# `sh` resolves the program on `PATH` and exits 127 when it is missing, as the protocol asks;
# `exec` keeps the argv literal and replaces the shell with the program. smolmachines 1.22 leaves
# a streamed command's stdin open, so it is closed here.
_EXEC = ('/bin/sh', '-c', 'exec "$@" </dev/null', 'sh')

# The local backend's cap and E2B's preview: the output is held in memory until the command ends.
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_OUTPUT_PREVIEW_CHARS = 64 * 1024

_ASYNCIO_ONLY = '`SmolSandbox` needs the asyncio event loop: the smol SDK runs its calls on asyncio threads.'


def _local() -> smol.ConnectOptions:
    # Pinned, so a `SMOL_CLOUD_TOKEN` in the environment can't move the workspace to Smol Cloud.
    return smol.ConnectOptions(target='local')


def _check_env(env: Mapping[str, str] | None) -> None:
    if env is not None and any(type(key) is not str or type(value) is not str for key, value in env.items()):
        raise TypeError('env keys and values must be strings')


def _events(
    machine: smol.AsyncMachine, argv: list[str], options: smol.ExecOptions
) -> AsyncGenerator[Mapping[str, object]]:
    """`AsyncMachine.exec_stream` with its real type: an async generator, which `aclosing` needs.

    The SDK annotates it as an `AsyncIterator` of bare `dict`s; the event shapes are in the module docstring.
    """
    events = machine.exec_stream(argv, options)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    assert isinstance(events, AsyncGenerator)
    return cast('AsyncGenerator[Mapping[str, object]]', events)


class _Output:
    """A command's output so far, bounded by `_MAX_OUTPUT_BYTES`."""

    def __init__(self) -> None:
        self.stdout: list[str] = []
        self.stderr: list[str] = []
        self.size = 0

    def add(self, stream: list[str], data: str) -> None:
        self.size += len(data.encode())
        if self.size > _MAX_OUTPUT_BYTES:
            # Only the preview of the event that crosses the cap is kept, so a single huge event
            # can't grow what is retained past the cap.
            stream.append(data[:_OUTPUT_PREVIEW_CHARS])
            stdout, stderr = self.text(_OUTPUT_PREVIEW_CHARS)
            raise WorkspaceOutputLimitError(
                'smol command output exceeded the 10 MiB limit; redirect it to a file and read part of it instead',
                limit=_MAX_OUTPUT_BYTES,
                stdout=stdout,
                stderr=stderr,
            )
        stream.append(data)

    def text(self, chars: int | None = None) -> tuple[str, str]:
        return ''.join(self.stdout)[:chars], ''.join(self.stderr)[:chars]


class SmolSandboxBackend(WorkspaceBackend, SupportsCommands):
    """A local [smolvm](https://github.com/smol-machines/smolvm) microVM as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands run inside a VM on this machine, with its own kernel and filesystem, so they can't
    touch the host's files or processes. File operations run as shell commands in the VM (see
    [Writing a backend](https://pydantic.dev/docs/ai/core-concepts/workspace/#writing-a-backend)),
    so a custom image needs `/bin/sh` and the usual file utilities.

    Building one does no I/O. The first operation boots a new persistent VM, or attaches to the one
    `ref` names, and `get_machine()` returns the SDK's `smol.AsyncMachine`. The backend never deletes
    the VM; whoever holds the ref does. A ref also works from another process, which boots the VM
    again, but only the VM's `stop()` flushes its disk: stop it before this process exits to keep
    recent file changes.

    Args:
        ref: Identity of an existing VM to attach to on first use.
        image: OCI image a new VM runs, such as `'python:3.12-alpine'`; `None` boots smolvm's own
            Alpine Linux root filesystem without pulling anything.
        cpus: vCPUs for a new VM; smolvm's default when `None`.
        memory_mb: Memory for a new VM in MiB; smolvm's default when `None`.
        allow_internet_access: Whether a new VM gets outbound network access.
        working_dir: Absolute directory commands start in and relative paths resolve against, created
            if missing when the backend first reaches the VM; `None` uses the VM's own, `/` on the
            default image.
        env: Environment variables every command gets; per-command `env` is layered on top. Nothing is
            read from the host environment.
    """

    def __init__(
        self,
        *,
        ref: WorkspaceRef | None = None,
        image: str | None = None,
        cpus: int | None = None,
        memory_mb: int | None = None,
        allow_internet_access: bool = False,
        working_dir: str | None = '/workspace',
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None and ref.provider != 'smol':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'smol'")
        _check_env(env)
        self._ref = ref
        self._image = image
        self._resources = smol.ResourceSpec(cpus=cpus, memory_mb=memory_mb, network=allow_internet_access)
        self._working_dir = absolute_path('working_dir', working_dir)
        self._env = dict(env or {})
        self._machine: smol.AsyncMachine | None = None
        self._acquisition: asyncio.Task[smol.AsyncMachine] | None = None
        self._resolved_working_dir: str | None = None

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the VM, or `None` before one has been created."""
        return self._ref

    async def get_machine(self) -> smol.AsyncMachine:
        """Return the SDK's `smol.AsyncMachine`, for smol features the workspace API does not cover.

        On a backend with no VM yet, this boots one or attaches to the one `ref` names, just like the
        first operation. Attaching to a deleted VM raises `WorkspaceUnavailableError`; it does not
        create a replacement.
        """
        if not running_on_asyncio():
            raise UserError(_ASYNCIO_ONLY)
        if self._machine is not None:
            return self._machine
        task = self._acquisition
        if task is None:
            # Concurrent first uses share this task, so they boot one VM. It isn't cancelled with a
            # caller, so a VM that boots after every caller left still has its ref recorded.
            task = asyncio.create_task(self._acquire(), name='smol-sandbox-acquisition')
            # Waiting callers still get the failure; this only stops asyncio reporting it as never
            # retrieved when every caller was cancelled first.
            task.add_done_callback(lambda done: done.cancelled() or done.exception())
            self._acquisition = task
        # `asyncio.wait`, not `asyncio.shield`: on 3.14 a shield cancelled before its task fails
        # reports that failure to the loop exception handler even though it is retrieved here.
        await asyncio.wait([task])
        return task.result()

    async def _acquire(self) -> smol.AsyncMachine:
        try:
            try:
                if self._ref is None:
                    config = smol.MachineConfig(image=self._image, resources=self._resources, persistent=True)
                    machine = await smol.AsyncMachine.create(config, _local())
                    # Recorded before any setup that can fail, so the caller can always delete the VM.
                    self._ref = WorkspaceRef(provider='smol', id=machine.id)
                    logger.info('Created smol VM %s', machine.id)
                else:
                    machine = await smol.AsyncMachine.connect(self._ref.id, _local())
            except smol.SmolError as error:
                if error.code == 'NOT_FOUND' and self._ref is not None:
                    raise WorkspaceUnavailableError(self._gone_message()) from error
                if error.code in _HOST_UNAVAILABLE:
                    raise WorkspaceUnavailableError(
                        f'This host cannot run a local smol VM ({error.code}): {error}'
                    ) from error
                raise
            if self._working_dir is not None:
                result = await self._exec(machine, ['mkdir', '-p', self._working_dir], env={}, working_dir=None)
                if result.exit_code != 0:
                    raise WorkspaceError(
                        f'Could not create working_dir {self._working_dir!r} in smol VM {machine.id}: {result.stderr}'
                    )
            self._machine = machine
            return machine
        finally:
            self._acquisition = None

    def _gone_message(self) -> str:
        assert self._ref is not None
        return f"The smol VM {self._ref.id!r} no longer exists. Pass `workspace='new'` to start a fresh VM."

    async def working_dir(self) -> str:
        """The canonical absolute directory commands start in."""
        if self._resolved_working_dir is None:
            result = await self.run(['pwd', '-P'])
            printed = result.stdout.removesuffix('\n')
            if result.exit_code != 0 or not printed.startswith('/'):
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
        """Run a command in the VM, killing it and the processes it started on timeout or cancellation."""
        argv = command_argv(command, shell)
        check_timeout(timeout)
        _check_env(env)
        # Booting or attaching is bounded by the SDK; the timeout is the command's alone.
        machine = await self.get_machine()
        return await self._exec(
            machine, [*_EXEC, *argv], env={**self._env, **(env or {})}, working_dir=self._working_dir, timeout=timeout
        )

    async def _exec(
        self,
        machine: smol.AsyncMachine,
        argv: list[str],
        *,
        env: dict[str, str],
        working_dir: str | None,
        timeout: float | None = None,
    ) -> CommandResult:
        output = _Output()
        exit_code: int | None = None
        options = smol.ExecOptions(env=env or None, workdir=working_dir)
        try:
            with anyio.move_on_after(timeout):
                events = _events(machine, argv, options)
                # Closing the generator, on any way out of this block, kills the command in the VM.
                async with aclosing(events):
                    async for event in events:
                        kind = event.get('kind')
                        if kind == 'stdout' or kind == 'stderr':
                            output.add(output.stdout if kind == 'stdout' else output.stderr, str(event.get('data')))
                        elif kind == 'exit':
                            code = event.get('exit_code')
                            assert isinstance(code, int)
                            exit_code = code
                        else:
                            # `error`, the only other kind the SDK sends, carries a bracketed code.
                            raise smol.wrap_native_error(RuntimeError(str(event.get('message', event))))
        except smol.SmolError as error:
            translated = await self._command_error(machine, error)
            if translated is error:
                raise
            raise translated from error
        stdout, stderr = output.text()
        if exit_code is None:
            raise WorkspaceTimeoutError(f'Command timed out after {timeout:g} seconds', stdout=stdout, stderr=stderr)
        return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    async def _command_error(self, machine: smol.AsyncMachine, error: smol.SmolError) -> Exception:
        """Map a failed command onto the protocol: a VM that is gone is unavailable, anything else propagates.

        The VM is probed only after a failure, to keep the round trip off successful commands; a probe
        that fails leaves the original error to propagate rather than ending the run on a guess.
        """
        if error.code == 'NOT_FOUND':
            return WorkspaceUnavailableError(self._gone_message())
        try:
            state = await machine.state()
        except smol.SmolError as probe_error:
            if probe_error.code == 'NOT_FOUND':
                return WorkspaceUnavailableError(self._gone_message())
            return error
        if state != 'running':
            return WorkspaceUnavailableError(
                f"The smol VM {machine.id!r} is {state}: it was stopped or deleted. Pass `workspace='new'` to start a "
                'fresh VM.'
            )
        if 'No such file or directory' in str(error):
            return WorkspaceError(
                f'Could not start the command in smol VM {machine.id}: {error}. '
                f'The working_dir {self._working_dir!r} or `/bin/sh` is missing in the VM.'
            )
        return error


async def delete_machine(machine_id: str) -> None:
    """Delete a VM by ID; one that is already gone returns quietly.

    The SDK has no delete by ID for a local VM, so this attaches first, which boots a stopped VM.
    """
    if not running_on_asyncio():
        raise UserError(_ASYNCIO_ONLY)
    try:
        machine = await smol.AsyncMachine.connect(machine_id, _local())
    except smol.SmolError as error:
        if error.code == 'NOT_FOUND':
            return
        raise
    await machine.delete()
