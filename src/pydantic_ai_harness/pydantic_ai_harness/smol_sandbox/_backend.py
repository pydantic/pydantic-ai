"""A local [smolvm](https://github.com/smol-machines/smolvm) microVM behind Pydantic AI's `WorkspaceBackend` protocol.

SDK behavior verified 2026-10-12 against smolmachines 1.22.2 and 1.27.0 on macOS (Apple silicon), see
https://github.com/smol-machines/smol/tree/main/sdk/python. Re-check these before changing acquisition,
command, or error handling:

- `Machine` blocks, and `AsyncMachine` runs each of its calls with `asyncio.to_thread`. The backend calls
  `Machine` on worker threads itself, because `AsyncMachine.exec_stream` opens the stream before the
  `try` that closes it, so a caller cancelled mid-open can't close the command it started.
- Without an image, a VM boots smolvm's own Alpine root filesystem with BusyBox, runs commands as root,
  and starts them in `/`.
- `exec_stream` returns an `ExecStream` of `{'kind': 'stdout' | 'stderr', 'data': str}` events, then
  `{'kind': 'exit', 'exit_code': int}`, or `{'kind': 'error', 'message': '[CODE] ...'}` instead of
  raising. Its `close()`, safe from any thread, kills the command and the processes it started in the
  VM, and wakes a thread waiting for the next event.
- A missing argv program exits 255 with an engine message rather than 127.
- Deleting a VM during a command ends the stream with an `io operation failed: connection closed` error
  event; the old handle's `state()` then reports `stopped`, and `connect` raises `NOT_FOUND`.
- A persistent VM outlives its process and `connect` boots it again, but only `stop()` (or `sync` in the
  guest) flushes its disk: a VM still running when its process exits loses its unflushed writes.
- On the local target a VM's id is its name, and `connect` opens any VM this OS user has by name.
- `ExecOptions.timeout` kills the command in the VM, which then exits 124.
- smolvm stops a command's stdout and stderr once they pass 11 MiB, but the native SDK queues output
  events without a bound of its own until they are read. `_MAX_OUTPUT_BYTES` bounds what the backend
  keeps; what smolmachines holds in between is up to that 11 MiB of output plus each event's overhead,
  which nothing bounds.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, TypeVar, cast

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

_T = TypeVar('_T')

_NAME_PREFIX = 'pydantic-ai-'
_NAME = re.compile(rf'{_NAME_PREFIX}[0-9a-f]{{32}}')
"""The names the backend gives its VMs, and so the only refs it attaches to or deletes.

A local VM's id is its name and `connect` opens any VM of this OS user, so without this check a ref
from message history could reach a VM made by another tool, with host mounts or network. The name is
a namespace, not proof of origin: any process of this OS user could use it, and smolmachines can't report
an existing VM's mounts or network, so a ref is still trusted input.
"""

_SETUP_TIMEOUT = 60.0
"""Seconds the backend's own commands in the VM (`mkdir -p` and `pwd -P`) get."""

_ENGINE_TIMEOUT_GRACE = 5.0
"""Seconds past a command's timeout before smolvm kills it itself, in case this event loop can't."""

lifecycle: dict[str, asyncio.Task[object]] = {}
"""This process's latest unfinished creation, attachment or deletion of each VM, by name.

Each one waits for the one started before it, so the latest finishing means they all have, and none
stops when its callers are cancelled.
"""
_HOST_UNAVAILABLE = frozenset(
    {'UNSUPPORTED_PLATFORM', 'RUNTIME_NOT_INSTALLED', 'KVM_UNAVAILABLE', 'HYPERVISOR_UNAVAILABLE'}
)
"""Codes for a host that can't run a local VM at all, so no retry can succeed."""

# `sh` resolves the program on `PATH` and exits 127 when it is missing, as the protocol asks;
# `exec` keeps the argv literal and replaces the shell with the program. BusyBox and bash read a
# program named like `-c` as an option of `exec`, and dash's `exec` takes no `--`, so such a name is
# resolved to its path first. smolmachines 1.22 leaves a streamed command's stdin open, so it is
# closed here.
_EXEC_SCRIPT = """p=$1
shift
case $p in
-*/*) p=./$p ;;
-*) q=$(command -v -- "$p") || { printf 'sh: %s: not found\\n' "$p" >&2; exit 127; }; p=$q ;;
esac
exec "$p" "$@" </dev/null"""
_EXEC = ('/bin/sh', '-c', _EXEC_SCRIPT, 'sh')

# The local backend's cap and E2B's preview: the output is held in memory until the command ends.
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_OUTPUT_PREVIEW_CHARS = 64 * 1024

ASYNCIO_ONLY = '`SmolSandbox` needs the asyncio event loop: the smol SDK runs its calls on asyncio threads.'


def local_target() -> smol.ConnectOptions:
    # Pinned, so a `SMOL_CLOUD_TOKEN` in the environment can't move the workspace to Smol Cloud.
    return smol.ConnectOptions(target='local')


def _check_env(env: Mapping[str, str] | None) -> None:
    if env is None:
        return
    if any(type(key) is not str or type(value) is not str for key, value in env.items()):
        raise TypeError('env keys and values must be strings')
    # The engine passes them on as C strings, which would cut a value short at a NUL.
    if any(not key or '=' in key or '\0' in key or '\0' in value for key, value in env.items()):
        raise ValueError('env keys must be non-empty without `=` or NUL, and values must not contain NUL')


def check_ref(ref: WorkspaceRef) -> None:
    """Raise `ValueError` unless `ref` names a VM this backend could have created."""
    if ref.provider != 'smol':
        raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'smol'")
    if _NAME.fullmatch(ref.id) is None:
        raise ValueError(
            f'SmolSandbox only uses VMs it created, named {_NAME_PREFIX!r} and 32 hex digits, got {ref.id!r}'
        )


def start_lifecycle(machine_id: str, work: Callable[[], Awaitable[_T]]) -> asyncio.Task[_T]:
    """Run `work` on `machine_id` once its earlier operations finish, in a task its callers' cancellation doesn't stop."""
    previous = lifecycle.get(machine_id)

    async def run() -> _T:
        if previous is not None:
            # A deletion that overtook a boot would find nothing, and the boot would then leave its VM behind.
            await asyncio.wait([previous])
        return await work()

    task = asyncio.create_task(run(), name=f'smol-sandbox-{machine_id}')
    # Waiting callers still get the failure; this only stops asyncio reporting it as never retrieved
    # when every caller was cancelled first.
    task.add_done_callback(lambda done: done.cancelled() or done.exception())
    lifecycle[machine_id] = task
    task.add_done_callback(lambda done: lifecycle.pop(machine_id) if lifecycle.get(machine_id) is done else None)
    return task


async def settled(task: asyncio.Task[_T]) -> _T:
    # `asyncio.wait`, not `asyncio.shield`: on 3.14 a shield cancelled before its task fails reports
    # that failure to the loop exception handler even though it is retrieved here.
    await asyncio.wait([task])
    return task.result()


async def _open(machine: smol.Machine, argv: list[str], options: smol.ExecOptions) -> smol.ExecStream:
    """Start a command, closing its stream when it arrives after the caller was cancelled."""
    opening = asyncio.ensure_future(asyncio.to_thread(machine.exec_stream, argv, options))
    try:
        await asyncio.wait([opening])
    except asyncio.CancelledError:
        # The open goes on in its thread and may still start the command, which closing then kills.
        opening.add_done_callback(_close_opened)
        raise
    return opening.result()


def _close_opened(opening: asyncio.Future[smol.ExecStream]) -> None:
    if opening.exception() is None:
        opening.result().close()


def _next_event(stream: smol.ExecStream) -> Mapping[str, object] | None:
    """The stream's next event, or `None` once it ends; the SDK types events as bare `dict`s."""
    return cast('Mapping[str, object] | None', next(stream, None))


class _Output:
    """A command's output so far, bounded by `_MAX_OUTPUT_BYTES`."""

    def __init__(self) -> None:
        self.stdout: list[str] = []
        self.stderr: list[str] = []
        self.size = 0

    def add(self, stream: list[str], data: str) -> None:
        # A character is at least one byte, so an event longer than what is left is over the cap
        # without encoding a copy of it.
        self.size += len(data) if len(data) > _MAX_OUTPUT_BYTES - self.size else len(data.encode())
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
    `ref` names, and `get_sandbox()` returns the SDK's `smol.AsyncMachine`. `ref` is set as soon as a
    boot starts, and only names VMs the backend creates, `pydantic-ai-` and 32 hex digits. The backend
    never deletes the VM; whoever holds the ref does. A ref also works from another process, which boots the VM
    again, but only the VM's `stop()` flushes its disk: stop it before this process exits to keep
    recent file changes.

    Args:
        ref: Identity of an existing VM to attach to on first use, one this backend created.
        image: OCI image a new VM runs, such as `'python:3.12-alpine'`; `None` boots smolvm's own
            Alpine Linux root filesystem without pulling anything.
        cpus: vCPUs for a new VM; smolvm's default when `None`.
        memory_mb: Memory for a new VM in MiB; smolvm's default when `None`.
        allow_internet_access: Whether a new VM gets unrestricted outbound network access, made from
            this machine's network: the internet, but also its local network and services.
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
        if ref is not None:
            check_ref(ref)
        _check_env(env)
        self._ref = ref
        self._image = image
        self._resources = smol.ResourceSpec(cpus=cpus, memory_mb=memory_mb, network=allow_internet_access)
        self._working_dir = absolute_path('working_dir', working_dir)
        self._env = dict(env or {})
        self._machine: smol.Machine | None = None
        self._sandbox: smol.AsyncMachine | None = None
        self._acquisition: asyncio.Task[smol.Machine] | None = None
        self._resolved_working_dir: str | None = None

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the VM, or `None` before one starts booting.

        Set before the boot finishes, so cleanup that runs while it is still booting can pass it to
        `SmolSandbox.destroy`, which waits for the boot.
        """
        return self._ref

    async def get_sandbox(self) -> smol.AsyncMachine:
        """Return the SDK's `smol.AsyncMachine`, for smol features the workspace API does not cover.

        On a backend with no VM yet, this boots one or attaches to the one `ref` names, just like the
        first operation. Attaching to a deleted VM raises `WorkspaceUnavailableError`; it does not
        create a replacement.
        """
        machine = await self._vm()
        if self._sandbox is None:
            self._sandbox = smol.AsyncMachine(machine)
        return self._sandbox

    async def _vm(self) -> smol.Machine:
        if not running_on_asyncio():
            raise UserError(ASYNCIO_ONLY)
        if self._machine is not None:
            return self._machine
        task = self._acquisition
        if task is None:
            # Concurrent first uses share this task, so they boot one VM.
            if (ref := self._ref) is None:
                # Named and recorded before the boot starts, so cleanup has the ref whatever ends the run.
                ref = self._ref = WorkspaceRef(provider='smol', id=f'{_NAME_PREFIX}{uuid.uuid4().hex}')
                logger.info('Creating smol VM %s', ref.id)
                task = start_lifecycle(ref.id, lambda: self._acquire(ref.id, create=True))
            else:
                task = start_lifecycle(ref.id, lambda: self._acquire(ref.id, create=False))
            self._acquisition = task
        return await settled(task)

    async def _acquire(self, machine_id: str, *, create: bool) -> smol.Machine:
        try:
            try:
                if create:
                    config = smol.MachineConfig(
                        name=machine_id, image=self._image, resources=self._resources, persistent=True
                    )
                    machine = await asyncio.to_thread(smol.Machine.create, config, local_target())
                else:
                    machine = await asyncio.to_thread(smol.Machine.connect, machine_id, local_target())
            except smol.SmolError as error:
                if create:
                    # The SDK reported the boot failed, so a retry boots a VM under a new name.
                    self._ref = None
                elif error.code == 'NOT_FOUND':
                    raise WorkspaceUnavailableError(_gone_message(machine_id)) from error
                if error.code in _HOST_UNAVAILABLE:
                    raise WorkspaceUnavailableError(
                        f'This host cannot run a local smol VM ({error.code}): {error}'
                    ) from error
                raise
            if self._working_dir is not None:
                result = await self._exec(
                    machine, ['mkdir', '-p', self._working_dir], env={}, working_dir=None, timeout=_SETUP_TIMEOUT
                )
                if result.exit_code != 0:
                    raise WorkspaceError(
                        f'Could not create working_dir {self._working_dir!r} in smol VM {machine.id}: {result.stderr}'
                    )
            self._machine = machine
            return machine
        finally:
            self._acquisition = None

    async def working_dir(self) -> str:
        """The canonical absolute directory commands start in."""
        if self._resolved_working_dir is None:
            result = await self.run(['pwd', '-P'], timeout=_SETUP_TIMEOUT)
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
        machine = await self._vm()
        return await self._exec(
            machine, [*_EXEC, *argv], env={**self._env, **(env or {})}, working_dir=self._working_dir, timeout=timeout
        )

    async def _exec(
        self,
        machine: smol.Machine,
        argv: list[str],
        *,
        env: dict[str, str],
        working_dir: str | None,
        timeout: float | None = None,
    ) -> CommandResult:
        output = _Output()
        exit_code: int | None = None
        engine_timeout = None if timeout is None else timeout + _ENGINE_TIMEOUT_GRACE
        options = smol.ExecOptions(env=env or None, workdir=working_dir, timeout=engine_timeout)
        try:
            with anyio.move_on_after(timeout) as scope:
                stream = await _open(machine, argv, options)
                try:
                    while (event := await asyncio.to_thread(_next_event, stream)) is not None:
                        kind = event.get('kind')
                        if kind == 'stdout' or kind == 'stderr':
                            if type(data := event.get('data')) is not str:
                                raise WorkspaceError(f'smol sent {kind} output that is not text: {type(data).__name__}')
                            output.add(output.stdout if kind == 'stdout' else output.stderr, data)
                        elif kind == 'exit':
                            if type(code := event.get('exit_code')) is not int:
                                raise WorkspaceError(f'smol sent an exit status that is not an integer: {code!r:.100}')
                            exit_code = code
                        else:
                            # `error`, the only other kind the SDK sends, carries a bracketed code.
                            raise smol.wrap_native_error(RuntimeError(str(event.get('message', event))))
                finally:
                    # On any way out, this kills the command in the VM and wakes a thread waiting for its next event.
                    stream.close()
        except smol.SmolError as error:
            translated = await self._command_error(machine, error)
            if translated is error:
                raise
            raise translated from error
        stdout, stderr = output.text()
        if exit_code is not None:
            return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)
        if scope.cancelled_caught:
            raise WorkspaceTimeoutError(f'Command timed out after {timeout:g} seconds', stdout=stdout, stderr=stderr)
        raise WorkspaceError(f'The command in smol VM {machine.id} ended without an exit status')

    async def _command_error(self, machine: smol.Machine, error: smol.SmolError) -> Exception:
        """Map a failed command onto the protocol: a VM that is gone is unavailable, anything else propagates.

        The VM is probed only after a failure, to keep the round trip off successful commands; a probe
        that fails leaves the original error to propagate rather than ending the run on a guess.
        """
        # Even a `NOT_FOUND` is checked: the error's code comes from its text, which may be the guest's.
        try:
            state = await asyncio.to_thread(machine.state)
        except smol.SmolError as probe_error:
            if probe_error.code == 'NOT_FOUND':
                return WorkspaceUnavailableError(_gone_message(machine.id))
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


def _gone_message(machine_id: str) -> str:
    return f"The smol VM {machine_id!r} no longer exists. Pass `workspace='new'` to start a fresh VM."
