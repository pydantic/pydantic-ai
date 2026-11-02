"""A local [BoxLite](https://github.com/boxlite-ai/boxlite) micro-VM behind Pydantic AI's `WorkspaceBackend` protocol.

SDK behavior verified 2026-10-20 against boxlite 0.10.7 on macOS (Apple silicon), see
https://github.com/boxlite-ai/boxlite/tree/main/sdks/python. Re-check these before changing acquisition,
command, or error handling:

- A process opens one runtime per BoxLite home directory (`~/.boxlite`, or `BOXLITE_HOME`). A second
  process opening the same one fails with a Rust panic, which Python sees as a `BaseException`.
- `Boxlite.create` only records a box; `Box.start()` boots it, and an image that can't be pulled fails
  there, leaving the record behind. `start()` on a running box returns at once. On Linux it jails the
  VM engine with bubblewrap, and raises `ConfigError` where it can't, such as Ubuntu 24.04 with only
  BoxLite's bundled `bwrap` (verified on GitHub's `ubuntu-latest`).
- A box stops when this process drops its last handle to it or exits. With `auto_delete=0` its disk,
  unflushed writes included, survives, and `get` plus `start()` boots it again.
- `Box.exec` raises instead of exiting 127 for a missing program, and leaves the command's stdin open.
- Output and exit status come over separate streams: `Execution.wait()` returns when the process exits,
  but its output ends only once every process holding its pipes has exited; a caller can stop reading
  it there, and the box's next exec works as usual. The guest buffers 1 MiB of
  output and, when the host falls behind, drops the oldest and sends `[boxlite] stdout output dropped N
  bytes` in its place, as a chunk of its own.
- Each exec'd process leads its own session and process group, but `Execution.signal`, `kill` and
  `timeout_secs` reach only that process, not its children.
- An exec'd process killed by a signal exits negative, and signalling one that has exited raises
  `InternalError`.
- `Boxlite.create`, `Box.exec` and `Boxlite.remove` finish after their caller is cancelled: a box being
  created appears after a deletion of it found nothing, and a command being started runs. A cancelled
  `start()` stops the boot, leaving the box created but not running.
- Removing a box mid-command ends the command with `-15` rather than an error, and later calls on its
  handle raise `UpstreamUnavailableError`. `Box.stop()` invalidates every handle to the box: later calls
  on one raise `StoppedError`, and a new handle from `get` boots it again.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

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
from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    check_env,
    check_timeout,
    command_argv,
    running_on_asyncio,
)

if TYPE_CHECKING:
    from anyio.abc import TaskGroup

    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('BoxLiteSandboxBackend',)

try:
    import boxlite as _boxlite
except ImportError as _import_error:
    raise ImportError(
        'Install `pydantic-ai-harness[boxlite]` to use BoxLiteSandbox. BoxLite has wheels for macOS 14 or newer on '
        'Apple silicon and for Linux on x86_64 and aarch64 (glibc 2.28 or newer).'
    ) from _import_error

boxlite: Any = _boxlite
"""The SDK, untyped: it ships no type information."""

logger = logging.getLogger(__name__)


# The SDK ships no type information; these describe the part of it the backend uses.
class _BoxState(Protocol):
    status: str


class _BoxInfo(Protocol):
    state: _BoxState


class _ExecResult(Protocol):
    exit_code: int


class _Execution(Protocol):
    """`boxlite.Execution`: a command started in a box."""

    def stdout(self) -> AsyncIterator[str]: ...
    def stderr(self) -> AsyncIterator[str]: ...
    async def wait(self) -> _ExecResult: ...
    async def signal(self, signal: int) -> None: ...


class _Box(Protocol):
    """`boxlite.Box`: a handle on one box."""

    @property
    def id(self) -> str: ...
    @property
    def name(self) -> str | None: ...
    async def start(self) -> None: ...
    async def exec(
        self, command: str, args: list[str], env: list[tuple[str, str]] | None = None, cwd: str | None = None
    ) -> _Execution: ...


class _Runtime(Protocol):
    """`boxlite.Boxlite`: the runtime that creates and finds boxes."""

    async def create(self, options: Any, name: str) -> _Box: ...
    async def get(self, id_or_name: str) -> _Box | None: ...
    async def get_info(self, id_or_name: str) -> _BoxInfo | None: ...
    async def remove(self, id_or_name: str, force: bool = False) -> None: ...


_T = TypeVar('_T')

_NAME_PREFIX = 'pydantic-ai-'
_NAME = re.compile(rf'{_NAME_PREFIX}[0-9a-f]{{32}}')
"""The names the backend gives its boxes, and so the only refs it attaches to or deletes.

`Boxlite.get` opens any box of this BoxLite home by name or id, so without this check a ref from
message history could reach a box made by another tool, with host mounts or published ports. The name
is a namespace, not proof of origin: any process of this OS user could use it, so a ref is still trusted input.
"""

_RESERVED_OPTIONS = frozenset(
    {'image', 'cpus', 'memory_mib', 'network', 'working_dir', 'env', 'auto_delete', 'auto_remove'}
)
"""`boxlite.BoxOptions` arguments the named settings own, so `box_options` can't pass them too."""

_SETUP_TIMEOUT = 60.0
"""Seconds the backend's own commands in the box get: `mkdir -p`, `pwd -P`, and the check that the box still runs."""

_SDK_TIMEOUT = 60.0
"""Seconds the backend waits for BoxLite to create or delete a box, or to start or stop a command.

BoxLite finishes those calls after their caller is cancelled, so a cancelled caller waits for them rather
than leave them running with no owner. Past this, the backend gives up and logs that the call may still finish.
"""

# The local backend's cap and E2B's preview: the output is held in memory until the command ends.
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_OUTPUT_PREVIEW_CHARS = 64 * 1024

_OUTPUT_DRAIN_GRACE = 2.0
"""Seconds the output is still read after the command exits, as the local backend does: a background
job can hold the output pipes open."""

_STOP = 1
"""SIGHUP, which the backend sends to stop a command; it has the same number on every platform.

Not TERM: BoxLite sends TERM to every process when a box stops or is deleted, and the script must die
of it then, so the backend can tell that apart from a command that exited.
"""

_STOP_GRACE = 5.0
"""Seconds a stopped command gets to end, so the next command doesn't find it still running."""

# The command runs in the background, in its own session where `setsid` exists, so the script can act on
# `_STOP` while waiting and then reap it: an orphan goes to the box's PID 1, the image's command, which
# never reaps. Without `setsid`, a stop reaches only the command, not processes it started, and a program
# named like a shell builtin runs the builtin. A stop that arrives before the command starts sets `t`, which
# starts empty whatever the command's `env` says. Stdin is at EOF; a missing program exits 127 and a
# program killed by a signal 128 plus its number. The second `wait` reaps a command after `_STOP`
# interrupts the first. Only builtins run before the command, as the command's `env` applies to the whole
# script.
_EXEC_SCRIPT = """t=
trap 't=1' HUP
if command -v setsid >/dev/null; then
setsid -- "$@" </dev/null &
else
"$@" </dev/null &
fi
c=$!
trap 'kill -KILL "-$c" 2>/dev/null || kill -KILL "$c"' HUP
[ -z "$t" ] || kill -KILL "-$c" 2>/dev/null || kill -KILL "$c"
wait "$c"
r=$?
wait "$c" 2>/dev/null
exit "$r"
"""

_DROPPED = re.compile(r'\[boxlite\] (?:stdout|stderr) output dropped \d+ bytes\r\n')
"""The line BoxLite sends in place of output its guest buffer dropped."""

_acquiring: dict[str, anyio.Event] = {}
"""The boxes this process is creating or attaching to, by name: a deletion waits for them."""

_deleting: dict[str, anyio.Event] = {}
"""The boxes this process is deleting, by name.

A box reports itself running for a moment after a deletion has ended its commands, so a command that
ended that way waits for the deletion before checking the box.
"""

ASYNCIO_ONLY = '`BoxLiteSandbox` needs the asyncio event loop: the BoxLite SDK returns asyncio futures.'


def runtime() -> _Runtime:
    """This process's BoxLite runtime, opened on first use."""
    try:
        return boxlite.Boxlite.default()
    except BaseException as error:
        # The runtime reports failing to open, such as on a home directory another process holds, as a Rust panic:
        # a `BaseException` that an application's `except Exception` would miss.
        if type(error).__name__ != 'PanicException':
            raise
        raise WorkspaceUnavailableError(f'Could not open the BoxLite runtime: {error}') from error


def check_ref(ref: WorkspaceRef) -> None:
    """Raise `ValueError` unless `ref` names a box this backend could have created.

    For refs from message history and for deletion: `get` opens any box of this BoxLite home by name or
    id, so such a ref could otherwise reach a box made by another tool, with host mounts or published ports.
    """
    _check_provider(ref)
    if _NAME.fullmatch(ref.id) is None:
        raise ValueError(
            f'BoxLiteSandbox only uses boxes it created, named {_NAME_PREFIX!r} and 32 hex digits, got {ref.id!r}. '
            'To use another box, pass `workspace=BoxLiteSandboxBackend(sandbox=box)` to the run; '
            "`workspace='new'` starts a fresh box."
        )


def _check_provider(ref: WorkspaceRef) -> None:
    if ref.provider != 'boxlite':
        raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'boxlite'")


def _key(box: _Box) -> str:
    """The name the runtime finds `box` by: its name, or its id if it has none."""
    return box.name or box.id


def _network(allow_internet_access: bool, allowed_hosts: Sequence[str] | None) -> Any:
    """The network a new box gets; raise `ValueError` for an allow-list that wouldn't apply."""
    if allowed_hosts is not None:
        if isinstance(allowed_hosts, str) or not all(type(host) is str and host for host in allowed_hosts):
            raise ValueError('allowed_hosts must be a sequence of non-empty host, IP or CIDR strings')
        if not allowed_hosts or not allow_internet_access:
            # BoxLite reads an empty allow-list as no restriction.
            raise ValueError('allowed_hosts needs allow_internet_access=True and at least one host')
    if not allow_internet_access:
        return boxlite.NetworkSpec(outbound=boxlite.OutboundNetworkSpec(mode='disabled'))
    return boxlite.NetworkSpec(
        outbound=boxlite.OutboundNetworkSpec(mode='enabled', allow_net=list(allowed_hosts or ()))
    )


async def _finish(call: Awaitable[_T], action: str, undo: Callable[[_T], Awaitable[object]] | None = None) -> _T:
    """Await an SDK call for up to `_SDK_TIMEOUT` seconds, even if the caller is cancelled meanwhile.

    The caller's cancellation is raised once the call has ended, after `undo` has run on its result.
    The call runs as a shielded task-group child, which a native `Task.cancel()` of the caller doesn't
    reach, unlike a shield in the caller's own task. `action` names the call in the error and warning
    raised when BoxLite doesn't answer in time.
    """
    done: list[_T] = []
    failed: list[BaseException] = []

    async def run() -> None:
        with anyio.move_on_after(_SDK_TIMEOUT, shield=True) as scope:
            try:
                done.append(await call)
            except anyio.get_cancelled_exc_class():
                raise
            except BaseException as error:
                # Raised as itself below, not in an `ExceptionGroup`, a Rust panic included.
                failed.append(error)
        if scope.cancelled_caught:
            logger.warning('BoxLite did not finish %s within %g seconds; it may still finish', action, _SDK_TIMEOUT)
            failed.append(
                WorkspaceUnavailableError(f'BoxLite did not finish {action} within {_SDK_TIMEOUT:g} seconds.')
            )

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(run)
    except BaseException:
        if done and undo is not None:
            await _finish(undo(done[0]), f'cleaning up after {action}')
        raise
    if failed:
        raise failed[0]
    if not done:
        # The SDK cancelled the call itself, as when its runtime shuts down.
        raise WorkspaceUnavailableError(f'BoxLite cancelled {action}.')
    return done[0]


@contextmanager
def _marking(registry: dict[str, anyio.Event], name: str) -> Generator[None]:
    """Record box `name` in `registry` until the block ends, for operations that wait for it."""
    registry[name] = event = anyio.Event()
    try:
        yield
    finally:
        event.set()
        if registry.get(name) is event:
            del registry[name]


async def delete(name: str) -> None:
    """Delete box `name` once this process has finished creating or attaching to it; quiet if it's gone."""
    # A deletion that overtook a creation would find nothing, and the creation would then leave its box behind.
    if (acquiring := _acquiring.get(name)) is not None:
        await acquiring.wait()
    with _marking(_deleting, name):
        await _remove(runtime(), name)


async def _remove(boxes: _Runtime, name: str) -> None:
    with suppress(boxlite.NotFoundError):
        await _finish(boxes.remove(name, force=True), f'deleting box {name!r}')


class _StaleHandle(Exception):
    """The box's handle was stopped before a command started on it."""


class _Output:
    """A command's output and exit code so far, the output bounded by `_MAX_OUTPUT_BYTES` across both streams.

    `read` and `wait` run in one task group. They record a failure and cancel the group rather than raise
    it, so `_exec` raises it as itself, not wrapped in an `ExceptionGroup`. The state lives here, not in
    variables of `_exec` set by nested functions, because pyright would narrow those to `None` and stop
    checking the code that reads them.
    """

    def __init__(self) -> None:
        self.stdout: list[str] = []
        self.stderr: list[str] = []
        self.size = 0
        self.exit_code: int | None = None
        self.error: Exception | None = None

    async def read(self, stream: AsyncIterator[str], chunks: list[str], group: TaskGroup) -> None:
        try:
            async for data in stream:
                self.add(chunks, data)
        except Exception as error:
            self.error = error
            group.cancel_scope.cancel()

    async def wait(self, execution: _Execution, group: TaskGroup) -> None:
        try:
            self.exit_code = (await execution.wait()).exit_code
        except Exception as error:
            self.error = error
            group.cancel_scope.cancel()
        else:
            # A background job the command started can hold its output open; read on briefly.
            group.cancel_scope.deadline = anyio.current_time() + _OUTPUT_DRAIN_GRACE

    def add(self, stream: list[str], data: str) -> None:
        if _DROPPED.fullmatch(data):
            raise WorkspaceError(f'BoxLite dropped part of the command output: {data.strip()}')
        # A character is at least one byte, so an event longer than what is left is over the cap
        # without encoding a copy of it.
        self.size += len(data) if len(data) > _MAX_OUTPUT_BYTES - self.size else len(data.encode())
        if self.size > _MAX_OUTPUT_BYTES:
            stream.append(data[:_OUTPUT_PREVIEW_CHARS])
            stdout, stderr = self.text(_OUTPUT_PREVIEW_CHARS)
            raise WorkspaceOutputLimitError(
                'BoxLite command output exceeded the 10 MiB limit; redirect it to a file and read part of it instead',
                limit=_MAX_OUTPUT_BYTES,
                stdout=stdout,
                stderr=stderr,
            )
        stream.append(data)

    def text(self, chars: int | None = None) -> tuple[str, str]:
        return ''.join(self.stdout)[:chars], ''.join(self.stderr)[:chars]


class BoxLiteSandboxBackend(WorkspaceBackend, SupportsCommands):
    """A local [BoxLite](https://github.com/boxlite-ai/boxlite) micro-VM as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands run inside a VM on this machine, with its own kernel and filesystem, so they can't
    touch the host's files or processes. File operations run as shell commands in the box (see
    [Writing a backend](https://pydantic.dev/docs/ai/core-concepts/workspace/#writing-a-backend)),
    so a custom image needs `/bin/sh` and the usual file utilities.

    Building one does no I/O. The first operation creates and boots a new box, or boots the one `sandbox`
    is or `ref` names, and `get_sandbox()` returns the SDK's `boxlite.Box`. `ref` is set as soon as a
    creation starts; a box the backend creates is named `pydantic-ai-` and 32 hex digits. The backend
    never deletes the box; whoever holds the ref does. The box stops when this process lets go of it or
    exits, keeping its disk, and the next use of the ref boots it again.

    Args:
        sandbox: A `boxlite.Box` you already have. Whoever created it owns deleting it.
        ref: Identity of an existing box to attach to on first use: its name, or its id if it has none.
        image: OCI image a new box runs, such as `'python:3.12-slim'`; BoxLite's default, Alpine Linux, when
            `None`.
        cpus: vCPUs for a new box; BoxLite's default when `None`.
        memory_mb: Memory for a new box in MiB; BoxLite's default when `None`.
        allow_internet_access: Whether a new box gets outbound network access, made from this machine's
            network: the internet, but also its local network and services.
        allowed_hosts: With `allow_internet_access`, the only hosts, IPs or CIDRs a new box can connect to;
            `None` allows any.
        working_dir: Absolute directory commands start in and relative paths resolve against, created
            if missing when the backend first reaches the box; `None` uses the box's own.
        env: Environment variables every command gets; per-command `env` is layered on top. Nothing is
            read from the host environment.
        box_options: Other keyword arguments for a new box's `boxlite.BoxOptions`, such as `volumes`,
            `ports`, `secrets` or `disk_size_gb`. The ones the arguments above set are refused.
    """

    def __init__(
        self,
        *,
        sandbox: Any = None,
        ref: WorkspaceRef | None = None,
        image: str | None = None,
        cpus: int | None = None,
        memory_mb: int | None = None,
        allow_internet_access: bool = False,
        allowed_hosts: Sequence[str] | None = None,
        working_dir: str | None = '/workspace',
        env: Mapping[str, str] | None = None,
        box_options: Mapping[str, Any] | None = None,
    ) -> None:
        if ref is not None:
            _check_provider(ref)
        if sandbox is not None and ref is not None:
            raise ValueError('pass either `sandbox` or `ref`, not both')
        check_env(env)
        if reserved := sorted(_RESERVED_OPTIONS.intersection(box_options or {})):
            raise ValueError(f'box_options must not set {", ".join(reserved)}; use the named argument instead')
        self._options: Any = boxlite.BoxOptions(
            **(box_options or {}),
            image=image,
            cpus=cpus,
            memory_mib=memory_mb,
            network=_network(allow_internet_access, allowed_hosts),
            # Kept after it stops, so the ref can boot it again.
            auto_delete=0,
        )
        self._given: _Box | None = sandbox
        self._ref = ref if sandbox is None else WorkspaceRef(provider='boxlite', id=_key(sandbox))
        self._working_dir = absolute_path('working_dir', working_dir)
        self._env = dict(env or {})
        self._box: _Box | None = None
        self._lock = anyio.Lock()
        self._resolved_working_dir: str | None = None

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the box, or `None` before one starts being created.

        Set before the creation finishes, so cleanup that runs while it is still in progress can pass it to
        `BoxLiteSandbox.destroy`, which waits for it.
        """
        return self._ref

    async def get_sandbox(self) -> Any:
        """Return the SDK's `boxlite.Box`, for BoxLite features the workspace API does not cover.

        On a backend with no box yet, this creates one or attaches to the one `ref` names, just like the
        first operation. Attaching to a deleted box raises `WorkspaceUnavailableError`; it does not
        create a replacement.
        """
        # Every operation gets the box first, so this one check covers them all.
        if not running_on_asyncio():
            raise UserError(ASYNCIO_ONLY)
        if self._box is not None:
            return self._box
        # Concurrent first uses wait for each other, so they create one box.
        async with self._lock:
            if self._box is not None:
                return self._box
            boxes = runtime()
            if (ref := self._ref) is not None:
                return await self._acquire(boxes, ref.id, create=False)
            # Named and recorded before the creation starts, so cleanup has the ref whatever ends the run.
            ref = self._ref = WorkspaceRef(provider='boxlite', id=f'{_NAME_PREFIX}{uuid.uuid4().hex}')
            logger.info('Creating BoxLite box %s', ref.id)
            return await self._acquire(boxes, ref.id, create=True)

    async def _acquire(self, boxes: _Runtime, name: str, *, create: bool) -> _Box:
        with _marking(_acquiring, name):
            if create:
                try:
                    box = await _finish(boxes.create(self._options, name=name), f'creating box {name!r}')
                    await _boot(box)
                except BaseException:
                    # A box that never booted, cancelled while booting included, is no use: delete it, and a
                    # retry creates another under a new name. If the deletion fails, the ref stays so its
                    # holder can delete the box later.
                    try:
                        await _remove(boxes, name)
                    except boxlite.BoxliteError:
                        logger.warning('Could not delete BoxLite box %s after it failed to start', name, exc_info=True)
                    else:
                        self._ref = None
                    raise
            elif (box := self._given) is not None:
                # Later lookups by `ref`, such as after a stop invalidates this handle, go through this
                # process's default runtime, which can't see a box of another BoxLite home.
                if await boxes.get_info(box.id) is None:
                    raise UserError(
                        f"The BoxLite box {name!r} passed as `sandbox` is not in this process's default BoxLite "
                        'runtime. Create it with `boxlite.Boxlite.default()`, after `boxlite.Boxlite.init_default(...)` '
                        'to use another home directory.'
                    )
                self._given = None
                await _boot(box)
            else:
                box = await boxes.get(name)
                if box is None:
                    raise WorkspaceUnavailableError(_gone_message(name))
                await _boot(box)
            if self._working_dir is not None:
                try:
                    result = await self._exec(
                        box, ['mkdir', '-p', self._working_dir], env={}, working_dir=None, timeout=_SETUP_TIMEOUT
                    )
                except _StaleHandle as error:
                    raise WorkspaceUnavailableError(f'The BoxLite box {name!r} was stopped while starting.') from error
                if result.exit_code != 0:
                    raise WorkspaceError(
                        f'Could not create working_dir {self._working_dir!r} in BoxLite box {name}: {result.stderr}'
                    )
            self._box = box
            return box

    async def working_dir(self) -> str:
        """The canonical absolute directory commands start in."""
        if self._resolved_working_dir is None:
            result = await self.run(['pwd', '-P'], timeout=_SETUP_TIMEOUT)
            printed = result.stdout.removesuffix('\n')
            if result.exit_code != 0 or not printed.startswith('/'):
                raise WorkspaceError(
                    'Could not determine the working directory of the BoxLite box: '
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
        """Run a command in the box, stopping it and the processes it started on timeout or cancellation."""
        argv = command_argv(command, shell)
        check_timeout(timeout)
        check_env(env)
        env = {**self._env, **(env or {})}
        # Creating or attaching is bounded by the SDK; the timeout is the command's alone.
        try:
            return await self._exec(
                await self.get_sandbox(), argv, env=env, working_dir=self._working_dir, timeout=timeout
            )
        except _StaleHandle:
            # `stop()` invalidates every handle to a box for good, but a new one boots it again. The
            # command never started, so it starts on the new handle.
            self._box = None
            box = await self.get_sandbox()
            try:
                return await self._exec(box, argv, env=env, working_dir=self._working_dir, timeout=timeout)
            except _StaleHandle as error:
                raise WorkspaceUnavailableError(f'The BoxLite box {_key(box)!r} was stopped while starting.') from error

    async def _exec(
        self,
        box: _Box,
        argv: list[str],
        *,
        env: dict[str, str],
        working_dir: str | None,
        timeout: float | None,
    ) -> CommandResult:
        output = _Output()
        execution: _Execution | None = None
        try:
            # The deadline covers starting the command too, but can't cut a start short: `_start` waits for the
            # command to start, up to `_SDK_TIMEOUT`, and stops it.
            with anyio.move_on_after(timeout):
                execution = await self._start(box, argv, env=env, working_dir=working_dir)
                async with anyio.create_task_group() as group:
                    group.start_soon(output.read, execution.stdout(), output.stdout, group)
                    group.start_soon(output.read, execution.stderr(), output.stderr, group)
                    group.start_soon(output.wait, execution, group)
            if output.error is not None:
                raise output.error
            if output.exit_code is None:
                stdout, stderr = output.text()
                raise WorkspaceTimeoutError(
                    f'Command timed out after {timeout:g} seconds', stdout=stdout, stderr=stderr
                )
        except BaseException as error:
            if execution is not None:
                # Timed out, cancelled, over the output cap, or the SDK failed during the command.
                await _finish(_stop(execution), f'stopping a command in box {_key(box)!r}')
            if isinstance(error, boxlite.BoxliteError) and (unavailable := await self._unavailable(box)) is not None:
                raise unavailable from error
            raise
        stdout, stderr = output.text()
        exit_code = output.exit_code
        if exit_code < 0:
            # Only the script itself dies by a signal, so the box was stopped or deleted, or the command
            # killed the script; in that last case, report it as `sh` would.
            await self._check_running(box)
            exit_code = 128 - exit_code
        return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    async def _start(self, box: _Box, argv: list[str], *, env: dict[str, str], working_dir: str | None) -> _Execution:
        try:
            # BoxLite starts the command even when its caller is cancelled first: wait for it, then stop it.
            return await _finish(
                box.exec('/bin/sh', ['-c', _EXEC_SCRIPT, 'sh', *argv], env=list(env.items()), cwd=working_dir),
                f'starting a command in box {_key(box)!r}',
                undo=_stop,
            )
        except boxlite.BoxliteError as error:
            if isinstance(error, boxlite.StoppedError):
                raise _StaleHandle from error
            if 'spawn_failed' in str(error):
                raise WorkspaceError(
                    f'Could not start the command in BoxLite box {_key(box)}: {error}. '
                    f'The working_dir {working_dir!r} or `/bin/sh` is missing in the box.'
                ) from error
            raise

    async def _check_running(self, box: _Box) -> None:
        """Raise `WorkspaceUnavailableError` if the box was stopped or deleted during a command."""
        if (unavailable := await self._unavailable(box)) is not None:
            raise unavailable
        # A box stopped from elsewhere still reports itself running for a moment, but can't start a command.
        try:
            with anyio.fail_after(_SETUP_TIMEOUT):
                await (await box.exec('/bin/sh', ['-c', ':'])).wait()
        except (boxlite.BoxliteError, TimeoutError) as error:
            raise WorkspaceUnavailableError(
                f'The BoxLite box {_key(box)!r} stopped during the command: {error}'
            ) from error

    async def _unavailable(self, box: _Box) -> WorkspaceUnavailableError | None:
        """A `WorkspaceUnavailableError` if the box is gone or stopped; `None` if it's running or can't be checked.

        Checked only after a failure, to keep the round trip off successful commands; a check that fails
        leaves the original error to propagate rather than ending the run on a guess.
        """
        name = _key(box)
        if (deleting := _deleting.get(name)) is not None:
            await deleting.wait()
        try:
            info = await runtime().get_info(name)
        except (boxlite.BoxliteError, WorkspaceUnavailableError):
            return None
        if info is None:
            return WorkspaceUnavailableError(_gone_message(name))
        if info.state.status != 'running':
            return WorkspaceUnavailableError(
                f'The BoxLite box {name!r} is {info.state.status}: it was stopped during the command.'
            )
        return None


async def _stop(execution: _Execution) -> None:
    """Stop a command and the processes it started, and wait up to `_STOP_GRACE` seconds for it to end."""
    # A command that exits as it is stopped can't be signalled any more.
    with anyio.move_on_after(_STOP_GRACE), suppress(boxlite.BoxliteError):
        await execution.signal(_STOP)
        await execution.wait()


async def _boot(box: _Box) -> None:
    try:
        await box.start()
    except boxlite.ConfigError as error:
        # Raised for a host that can't run boxes at all, such as one where BoxLite can't jail its engine.
        raise WorkspaceUnavailableError(f'This host cannot run a BoxLite box: {error}') from error


def _gone_message(name: str) -> str:
    return f"The BoxLite box {name!r} no longer exists. Pass `workspace='new'` to start a fresh box."
