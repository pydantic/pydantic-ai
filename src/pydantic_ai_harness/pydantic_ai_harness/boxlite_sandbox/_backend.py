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
  but its output ends only once every process holding its pipes has exited. The guest buffers 1 MiB of
  output and, when the host falls behind, drops the oldest and sends `[boxlite] stdout output dropped N
  bytes` in its place, as a chunk of its own.
- Each exec'd process leads its own session and process group, but `Execution.signal`, `kill` and
  `timeout_secs` reach only that process, not its children.
- An exec'd process killed by a signal exits negative, and signalling one that has exited raises
  `InternalError`.
- Removing a box mid-command ends the command with `-15` rather than an error, and later calls on its
  handle raise `UpstreamUnavailableError`. `Box.stop()` invalidates every handle to the box: later calls
  on one raise `StoppedError`, and a new handle from `get` boots it again.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

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
    stop_shielded,
)

if TYPE_CHECKING:
    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('BoxLiteSandboxBackend',)

try:
    import boxlite
except ImportError as _import_error:
    raise ImportError(
        'Install `pydantic-ai-harness[boxlite]` to use BoxLiteSandbox. BoxLite has wheels for macOS 14 or newer on '
        'Apple silicon and for Linux on x86_64 and aarch64 (glibc 2.28 or newer).'
    ) from _import_error

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
    def name(self) -> str: ...
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

RESERVED_OPTIONS = frozenset(
    {'image', 'cpus', 'memory_mib', 'network', 'working_dir', 'env', 'auto_delete', 'auto_remove'}
)
"""`boxlite.BoxOptions` arguments the named settings own, so `box_options` can't pass them too."""

_SETUP_TIMEOUT = 60.0
"""Seconds the backend's own commands in the box (`mkdir -p` and `pwd -P`) get."""

_OUTPUT = '/tmp/pydantic-ai-'
"""Prefix of the files in the box each command writes its output to, named per command."""

# The local backend's cap and E2B's preview: the output is held in memory until the command ends.
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_OUTPUT_PREVIEW_CHARS = 64 * 1024

_STOP = 1
"""SIGHUP, which the backend sends to stop a command; it has the same number on every platform.

Not TERM: BoxLite sends TERM to every process when a box stops or is deleted, and the script must die
of it then, so the backend can tell that apart from a command that exited.
"""

_STOP_GRACE = 5.0
"""Seconds a stopped command gets to end before its partial output is read anyway."""

# The command runs in the background, in its own session where `setsid` exists, so the script can act on
# `_STOP` while waiting and then reap it: an orphan goes to the box's PID 1, the image's command, which
# never reaps. Without `setsid`, a stop reaches only the command, not processes it started, and a program
# named like a shell builtin runs the builtin. A stop that arrives before the command starts sets `t`, which
# starts empty whatever the command's `env` says.
# Output goes to files, not the exec's pipes, so a background job the command starts can't hold the
# command open. Stdin is at EOF; a missing program exits 127 and a program killed by a signal 128 plus its
# number. The second `wait` reaps a command after `_STOP` interrupts the first. Only builtins run before the
# command, as the command's `env` applies to the whole script.
_EXEC_SCRIPT = """t=
trap 't=1' HUP
o=$1
shift
if command -v setsid >/dev/null; then
setsid -- "$@" </dev/null >"$o.stdout" 2>"$o.stderr" &
else
"$@" </dev/null >"$o.stdout" 2>"$o.stderr" &
fi
c=$!
trap 'kill -KILL "-$c" 2>/dev/null || kill -KILL "$c"' HUP
[ -z "$t" ] || kill -KILL "-$c" 2>/dev/null || kill -KILL "$c"
wait "$c"
r=$?
wait "$c" 2>/dev/null
exit "$r"
"""

# Reads a command's output, at most one byte past the cap of each, and removes it. Missing output fails
# unless `$2` is set.
_READ_SCRIPT = f"""[ -f "$1.stdout" ] || {{ [ -n "$2" ]; exit; }}
head -c {_MAX_OUTPUT_BYTES + 1} "$1.stdout" && head -c {_MAX_OUTPUT_BYTES + 1} "$1.stderr" >&2
r=$?
rm -f "$1.stdout" "$1.stderr"
exit "$r"
"""

_DROPPED = re.compile(r'\[boxlite\] (?:stdout|stderr) output dropped \d+ bytes\r\n')
"""The line BoxLite sends in place of output its guest buffer dropped."""

lifecycle: dict[str, asyncio.Task[object]] = {}
"""This process's latest unfinished creation, attachment or deletion of each box, by name.

Each one waits for the one started before it, so the latest finishing means they all have, and none
stops when its callers are cancelled. A box reports itself running for a moment after a deletion has
ended its commands, so a command that ended that way waits for the deletion before checking the box.
"""

_late_stops: set[asyncio.Task[None]] = set()
"""Stops of commands that started after their caller was cancelled, held until they end."""

ASYNCIO_ONLY = '`BoxLiteSandbox` needs the asyncio event loop: the BoxLite SDK returns asyncio futures.'


def sdk() -> Any:
    """The `boxlite` module, untyped: the SDK ships no type information."""
    return boxlite


def runtime() -> _Runtime:
    """This process's BoxLite runtime, opened on first use."""
    try:
        return cast(_Runtime, sdk().Boxlite.default())
    except BaseException as error:
        # The runtime reports failing to open, such as on a home directory another process holds, as a Rust panic.
        if type(error).__name__ != 'PanicException':
            raise
        raise WorkspaceUnavailableError(f'Could not open the BoxLite runtime: {error}') from error


def check_ref(ref: WorkspaceRef) -> None:
    """Raise `ValueError` unless `ref` names a box this backend could have created."""
    if ref.provider != 'boxlite':
        raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'boxlite'")
    if _NAME.fullmatch(ref.id) is None:
        raise ValueError(
            f'BoxLiteSandbox only uses boxes it created, named {_NAME_PREFIX!r} and 32 hex digits, got {ref.id!r}'
        )


def network(allow_internet_access: bool, allowed_hosts: Sequence[str] | None) -> Any:
    """The network a new box gets; raise `ValueError` for an allow-list that wouldn't apply."""
    if allowed_hosts is not None:
        if isinstance(allowed_hosts, str) or not all(type(host) is str and host for host in allowed_hosts):
            raise ValueError('allowed_hosts must be a sequence of non-empty host, IP or CIDR strings')
        if not allowed_hosts or not allow_internet_access:
            # BoxLite reads an empty allow-list as no restriction.
            raise ValueError('allowed_hosts needs allow_internet_access=True and at least one host')
    if not allow_internet_access:
        return sdk().NetworkSpec(outbound=sdk().OutboundNetworkSpec(mode='disabled'))
    return sdk().NetworkSpec(outbound=sdk().OutboundNetworkSpec(mode='enabled', allow_net=list(allowed_hosts or ())))


def start_lifecycle(name: str, work: Callable[[], Awaitable[_T]]) -> asyncio.Task[_T]:
    """Run `work` on box `name` once its earlier operations finish, in a task its callers' cancellation doesn't stop."""
    previous = lifecycle.get(name)

    async def run() -> _T:
        if previous is not None:
            # A deletion that overtook a creation would find nothing, and the creation would then leave its box behind.
            await asyncio.wait([previous])
        return await work()

    task = asyncio.create_task(run(), name=f'boxlite-sandbox-{name}')
    # Waiting callers still get the failure; this only stops asyncio reporting it as never retrieved
    # when every caller was cancelled first.
    task.add_done_callback(lambda done: done.cancelled() or done.exception())
    lifecycle[name] = task
    task.add_done_callback(lambda done: lifecycle.pop(name) if lifecycle.get(name) is done else None)
    return task


async def settled(task: asyncio.Task[_T]) -> _T:
    # `asyncio.wait`, not `asyncio.shield`: on 3.14 a shield cancelled before its task fails reports
    # that failure to the loop exception handler even though it is retrieved here.
    await asyncio.wait([task])
    return task.result()


class _StaleHandle(Exception):
    """The box's handle was stopped before a command started on it."""


class _Output:
    """A command's output, bounded by `_MAX_OUTPUT_BYTES` across both streams."""

    def __init__(self) -> None:
        self.stdout: list[str] = []
        self.stderr: list[str] = []
        self.size = 0

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

    Building one does no I/O. The first operation creates and boots a new box, or attaches to the one
    `ref` names, and `get_sandbox()` returns the SDK's `boxlite.Box`. `ref` is set as soon as a
    creation starts, and only names boxes the backend creates, `pydantic-ai-` and 32 hex digits. The
    backend never deletes the box; whoever holds the ref does. The box stops when this process lets go of
    it or exits, keeping its disk, and the next use of the ref boots it again.

    Args:
        ref: Identity of an existing box to attach to on first use, one this backend created.
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
            check_ref(ref)
        check_env(env)
        if reserved := sorted(RESERVED_OPTIONS.intersection(box_options or {})):
            raise ValueError(f'box_options must not set {", ".join(reserved)}; use the named argument instead')
        self._options: Any = sdk().BoxOptions(
            **(box_options or {}),
            image=image,
            cpus=cpus,
            memory_mib=memory_mb,
            network=network(allow_internet_access, allowed_hosts),
            # Kept after it stops, so the ref can boot it again.
            auto_delete=0,
        )
        self._ref = ref
        self._working_dir = absolute_path('working_dir', working_dir)
        self._env = dict(env or {})
        self._box: _Box | None = None
        self._acquisition: asyncio.Task[_Box] | None = None
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
        return await self._get_box()

    async def _get_box(self) -> _Box:
        if not running_on_asyncio():
            raise UserError(ASYNCIO_ONLY)
        if self._box is not None:
            return self._box
        task = self._acquisition
        if task is None:
            # Concurrent first uses share this task, so they create one box; it runs on when they are cancelled.
            if (ref := self._ref) is None:
                # Named and recorded before the creation starts, so cleanup has the ref whatever ends the run.
                ref = self._ref = WorkspaceRef(provider='boxlite', id=f'{_NAME_PREFIX}{uuid.uuid4().hex}')
                logger.info('Creating BoxLite box %s', ref.id)
                task = start_lifecycle(ref.id, lambda: self._acquire(ref.id, create=True))
            else:
                task = start_lifecycle(ref.id, lambda: self._acquire(ref.id, create=False))
            self._acquisition = task
        return await settled(task)

    async def _acquire(self, name: str, *, create: bool) -> _Box:
        try:
            boxes = runtime()
            if create:
                try:
                    box = await boxes.create(self._options, name=name)
                    await _boot(box)
                except BaseException:
                    # A box that never booted is no use: delete it, and a retry creates another under a new
                    # name. If the deletion fails, the ref stays so its holder can delete the box later.
                    try:
                        await boxes.remove(name, force=True)
                    except boxlite.NotFoundError:
                        self._ref = None
                    except boxlite.BoxliteError:
                        logger.warning('Could not delete BoxLite box %s after it failed to start', name, exc_info=True)
                    else:
                        self._ref = None
                    raise
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
        finally:
            self._acquisition = None

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
                await self._get_box(), argv, env=env, working_dir=self._working_dir, timeout=timeout
            )
        except _StaleHandle:
            # `stop()` invalidates every handle to a box for good, but a new one boots it again. The
            # command never started, so it starts on the new handle.
            self._box = None
            return await self._exec(
                await self._get_box(), argv, env=env, working_dir=self._working_dir, timeout=timeout
            )

    async def _exec(
        self,
        box: _Box,
        argv: list[str],
        *,
        env: dict[str, str],
        working_dir: str | None,
        timeout: float | None,
    ) -> CommandResult:
        output_files = f'{_OUTPUT}{uuid.uuid4().hex}'
        signalled = False
        try:
            exit_code = await self._run_script(
                box, argv, output_files, env=env, working_dir=working_dir, timeout=timeout
            )
            if signalled := exit_code < 0:
                # Only the script itself dies by a signal, so the box was stopped or deleted, or a command
                # killed the script; in that last case, report it as `sh` would.
                if (unavailable := await self._unavailable(box)) is not None:
                    raise unavailable
                exit_code = 128 - exit_code
            stdout, stderr = await self._read_output(box, output_files)
        except boxlite.BoxliteError as error:
            if (unavailable := await self._unavailable(box)) is not None:
                raise unavailable from error
            if signalled:
                # The box still reports itself running for a moment after being stopped from elsewhere.
                raise WorkspaceUnavailableError(
                    f'The BoxLite box {box.name!r} stopped during the command: {error}'
                ) from error
            raise
        return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    async def _run_script(
        self,
        box: _Box,
        argv: list[str],
        output_files: str,
        *,
        env: dict[str, str],
        working_dir: str | None,
        timeout: float | None,
    ) -> int:
        """The script's exit code; on a timeout, stop the command and raise with its partial output."""
        execution: _Execution | None = None
        with anyio.move_on_after(timeout) as deadline:
            execution = started = await self._start(box, argv, output_files, env=env, working_dir=working_dir)
            try:
                return (await started.wait()).exit_code
            except BaseException:
                if not deadline.cancel_called:
                    await stop_shielded(lambda: self._stop(box, started, output_files))
                raise
        message = f'Command timed out after {timeout:g} seconds'
        if execution is None:
            # It timed out starting, so it has no output; if it starts later, `_start` stops it.
            raise WorkspaceTimeoutError(message)
        await self._signal_stop(execution)
        with anyio.move_on_after(_STOP_GRACE):
            await execution.wait()
        # A command stopped before the script started it has no output.
        stdout, stderr = await self._read_output(box, output_files, missing_ok=True)
        raise WorkspaceTimeoutError(message, stdout=stdout, stderr=stderr)

    async def _start(
        self, box: _Box, argv: list[str], output_files: str, *, env: dict[str, str], working_dir: str | None
    ) -> _Execution:
        opening = asyncio.ensure_future(
            box.exec('/bin/sh', ['-c', _EXEC_SCRIPT, 'sh', output_files, *argv], env=list(env.items()), cwd=working_dir)
        )
        try:
            await asyncio.wait([opening])
        except asyncio.CancelledError:
            # The SDK starts the command even when its caller is cancelled first; stop it once it has.
            opening.add_done_callback(lambda done: self._stop_late(box, done, output_files))
            raise
        try:
            return opening.result()
        except boxlite.BoxliteError as error:
            if isinstance(error, boxlite.StoppedError):
                raise _StaleHandle from error
            if (unavailable := await self._unavailable(box)) is not None:
                raise unavailable from error
            if 'spawn_failed' in str(error):
                raise WorkspaceError(
                    f'Could not start the command in BoxLite box {box.name}: {error}. '
                    f'The working_dir {working_dir!r} or `/bin/sh` is missing in the box.'
                ) from error
            raise

    def _stop_late(self, box: _Box, opening: asyncio.Future[_Execution], output_files: str) -> None:
        if opening.cancelled() or opening.exception() is not None:
            return
        stop = asyncio.ensure_future(self._stop(box, opening.result(), output_files))
        _late_stops.add(stop)
        stop.add_done_callback(_late_stops.discard)

    async def _signal_stop(self, execution: _Execution) -> None:
        # A command that exits as it is stopped can't be signalled any more.
        with suppress(boxlite.BoxliteError):
            await execution.signal(_STOP)

    async def _stop(self, box: _Box, execution: _Execution, output_files: str) -> None:
        await self._signal_stop(execution)
        with suppress(boxlite.BoxliteError):
            await box.exec('rm', ['-f', f'{output_files}.stdout', f'{output_files}.stderr'])

    async def _read_output(self, box: _Box, output_files: str, *, missing_ok: bool = False) -> tuple[str, str]:
        reader = await box.exec('/bin/sh', ['-c', _READ_SCRIPT, 'sh', output_files, 'missing_ok' if missing_ok else ''])
        collected = _Output()
        failures: list[Exception] = []

        async def collect(stream: AsyncIterator[str], chunks: list[str]) -> None:
            try:
                async for chunk in stream:
                    collected.add(chunks, chunk)
            except Exception as error:
                # Raised below, after the task group, so it doesn't reach the caller wrapped in an `ExceptionGroup`.
                failures.append(error)
                group.cancel_scope.cancel()

        async with anyio.create_task_group() as group:
            group.start_soon(collect, reader.stdout(), collected.stdout)
            group.start_soon(collect, reader.stderr(), collected.stderr)
        if failures:
            raise failures[0]
        exit_code = (await reader.wait()).exit_code
        if exit_code != 0:
            raise WorkspaceError(
                f'Could not read the command output in BoxLite box {box.name} (exit code {exit_code}). '
                'The image needs `/bin/sh`, `head` and `rm`, and a writable `/tmp`.'
            )
        return collected.text()

    async def _unavailable(self, box: _Box) -> WorkspaceUnavailableError | None:
        """A `WorkspaceUnavailableError` if the box is gone or stopped; `None` if it's running or can't be checked.

        Checked only after a failure, to keep the round trip off successful commands; a check that fails
        leaves the original error to propagate rather than ending the run on a guess.
        """
        # Not while acquiring: an operation queued behind the acquisition waits for it to finish.
        if (pending := lifecycle.get(box.name)) is not None and asyncio.current_task() is not self._acquisition:
            await asyncio.wait([pending])
        try:
            info = await runtime().get_info(box.name)
        except (boxlite.BoxliteError, WorkspaceUnavailableError):
            return None
        if info is None:
            return WorkspaceUnavailableError(_gone_message(box.name))
        if info.state.status != 'running':
            return WorkspaceUnavailableError(
                f'The BoxLite box {box.name!r} is {info.state.status}: it was stopped during the command.'
            )
        return None


async def _boot(box: _Box) -> None:
    try:
        await box.start()
    except boxlite.ConfigError as error:
        # Raised for a host that can't run boxes at all, such as one where BoxLite can't jail its engine.
        raise WorkspaceUnavailableError(f'This host cannot run a BoxLite box: {error}') from error


def _gone_message(name: str | None) -> str:
    return f"The BoxLite box {name!r} no longer exists. Pass `workspace='new'` to start a fresh box."
