"""A Daytona sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

Daytona's session logs can deliver a block of stdout as stderr or the reverse, so each command writes
both streams to files that are downloaded once it exits; the log stream only signals that it ended.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import posixpath
import shlex
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal, NoReturn

import anyio
import anyio.lowlevel

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    SupportsRealpath,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    check_timeout,
    command_argv,
    command_deadline,
    run_to_completion,
    running_on_asyncio,
    safe_credential_reason,
    stop_shielded,
)

if TYPE_CHECKING:
    from daytona import AsyncDaytona, AsyncSandbox

    # Not re-exported at the package root; typing-only.
    from daytona._async.process import AsyncProcess

    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('DaytonaSandboxBackend',)

try:
    import daytona
except ImportError as error:
    raise ImportError('Install `pydantic-ai-harness[daytona]` to use DaytonaSandbox.') from error

_AUTH_MESSAGE = (
    'Daytona rejected the credentials. Set DAYTONA_API_KEY, or pass a configured `AsyncDaytona` as `client=`.'
)
_MISSING_CREDENTIALS_MESSAGE = (
    'No Daytona credentials found. Set DAYTONA_API_KEY, or pass a configured `AsyncDaytona` as `client=`.'
)
_REQUEST_TIMEOUT = 30
# Bound sandbox creation and attachment so a wedged control plane cannot hang acquisition.
_CREATE_TIMEOUT = 120
# For create, start, and delete.
_LIFECYCLE_TIMEOUT = 60.0
_TEARDOWN_TIMEOUT = 30.0
_EVENT_CONNECT_GRACE = 6.0
# Between attempts to delete a command session.
_RETRY_DELAY = 1.0
# The most partial output a timeout reports per stream.
_PARTIAL_OUTPUT_LIMIT = 4096
# A command's whole output is downloaded into memory.
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_OUTPUT_PREVIEW_BYTES = 64 * 1024

_logger = logging.getLogger(__name__)


def _require_asyncio() -> None:
    if not running_on_asyncio():
        raise UserError('Daytona needs the asyncio event loop: the Daytona SDK runs its calls on asyncio tasks.')


def _new_client() -> AsyncDaytona:
    """An `AsyncDaytona` configured from the environment: `DAYTONA_API_KEY`, or a JWT token and organization ID."""
    # The SDK's engineio logger reports routine empty queues at ERROR; mute it unless the application set a level.
    engineio_logger = logging.getLogger('engineio.client')
    if engineio_logger.level == logging.NOTSET:
        engineio_logger.setLevel(logging.CRITICAL)
    try:
        return daytona.AsyncDaytona()
    except daytona.DaytonaAuthenticationError as error:
        # The SDK reads the credentials here, so this is a missing key rather than a rejected one.
        raise WorkspaceUnavailableError(_MISSING_CREDENTIALS_MESSAGE) from error
    except Exception as error:
        _raise_translated(error, 'Could not configure the Daytona client')


async def destroy_sandbox(client: AsyncDaytona | None, sandbox_id: str) -> None:
    """Delete a sandbox by ID without starting it, including one Daytona has stopped or archived.

    A sandbox that no longer exists returns quietly; rejected credentials raise
    `WorkspaceUnavailableError`, like any other operation. Without `client`, one is opened from
    the environment and closed afterwards.

    Raises:
        UserError: The event loop is not asyncio.
    """
    _require_asyncio()
    if not sandbox_id.strip():
        raise ValueError('Daytona workspace ref id cannot be empty')
    owned: AsyncDaytona | None = None
    if client is None:
        client = owned = _new_client()
    try:
        sandbox = await client.get(sandbox_id, request_timeout=_REQUEST_TIMEOUT)
        if not _in_deleted_state(sandbox):
            await sandbox.delete(timeout=_LIFECYCLE_TIMEOUT)
    except daytona.DaytonaNotFoundError:
        return
    except (daytona.DaytonaAuthenticationError, daytona.DaytonaForbiddenError) as error:
        raise WorkspaceUnavailableError(f'{safe_credential_reason(error)}. {_AUTH_MESSAGE}') from error
    finally:
        if owned is not None:
            with anyio.move_on_after(_TEARDOWN_TIMEOUT, shield=True):
                await _close_sdk_client(owned)


async def _delete_session(process: AsyncProcess, session_id: str) -> None:
    """Delete a command session, which kills its command, retrying until the caller's `stop_shielded` bound.

    A failure is logged, not raised, so it cannot replace the caller's outcome.
    """
    deleted = False
    try:
        while True:
            try:
                await process.delete_session(session_id, request_timeout=_REQUEST_TIMEOUT)
                deleted = True
                return
            except daytona.DaytonaNotFoundError:
                deleted = True
                return
            except Exception:
                await anyio.sleep(_RETRY_DELAY)
    finally:
        if not deleted:
            _logger.warning(
                'Could not delete Daytona command session %s; its command may still be running.', session_id
            )


def _output_limit_error(stdout: str, stderr: str) -> WorkspaceOutputLimitError:
    return WorkspaceOutputLimitError(
        'Daytona command output exceeded the 10 MiB limit; redirect it to a file and read part of it instead',
        limit=_MAX_OUTPUT_BYTES,
        stdout=stdout,
        stderr=stderr,
    )


def _output_paths(marker: str) -> tuple[str, str]:
    return f'/tmp/{marker}.out', f'/tmp/{marker}.err'


def _command_line(argv: Sequence[str], *, cwd: str | None, env: Mapping[str, str], marker: str) -> str:
    """The session command: `argv` with its stdout and stderr sent to files, then `marker` printed on stdout.

    The `sh` exits with the argv's status, or 125 after writing `<marker>-cwd` when `cwd` is missing.
    """
    if env:
        # Inside the `sh`, since a `sh` such as dash drops names that are not shell identifiers.
        # `--` ends `env`'s options, so a name starting with `-` is not read as one.
        argv = ['env', '--', *(f'{name}={value}' for name, value in env.items()), *argv]
    out, err = _output_paths(marker)
    prefix = f'cd -- {shlex.quote(cwd)} || {{ printf %s {marker}-cwd >{out}; exit 125; }}; ' if cwd is not None else ''
    script = f'{prefix}"$@" </dev/null >{out} 2>{err}; status=$?; printf %s {marker}; exit "$status"'
    return shlex.join(['sh', '-c', script, 'sh', *argv])


class _DaytonaProcess:
    """A command running in its own Daytona process session."""

    def __init__(self, sandbox: AsyncSandbox, session_id: str, command_id: str, marker: str) -> None:
        self._sandbox = sandbox
        self._process = sandbox.process
        self._session_id = session_id
        self._command_id = command_id
        self._marker = marker
        self._ended = anyio.Event()
        self._tail = ''
        self._finished = False
        self.missing_cwd = False
        # The partial output a timeout reports, read from the output files by `kill()`.
        self.stdout = ''
        self.stderr = ''

    def _on_stdout(self, chunk: str) -> None:
        # The stream carries little besides the end marker, which can straddle two chunks.
        candidate = self._tail + chunk
        self._tail = candidate[-(len(self._marker) - 1) :]
        if self._marker in candidate:
            self._ended.set()

    def _on_stderr(self, chunk: str) -> None:
        pass

    async def wait(self) -> CommandResult:
        logs_finished = anyio.Event()
        logs_error: list[Exception] = []

        async def read_logs() -> None:
            try:
                await self._process.get_session_command_logs_async(
                    self._session_id, self._command_id, self._on_stdout, self._on_stderr
                )
            except Exception as error:
                logs_error.append(error)
            finally:
                logs_finished.set()

        exit_code: int | None = None
        failure: Exception | None = None
        # The log reader must stop before the session or its client can be closed.
        async with anyio.create_task_group() as readers:
            readers.start_soon(read_logs)
            try:
                exit_code = await self._exit_code(logs_finished, logs_error)
            except Exception as error:
                failure = error
            finally:
                readers.cancel_scope.cancel()
        if failure is not None:
            await _raise_failure(self._sandbox, failure, 'Could not read the command result')
        if exit_code is None:
            raise WorkspaceError('Daytona closed the command output before reporting an exit status.')
        self._finished = True
        return await self._result(exit_code)

    async def _exit_code(self, logs_finished: anyio.Event, logs_error: list[Exception]) -> int | None:
        """Poll the command status, more often once the log stream shows the end marker.

        The status is authoritative: the log stream can withhold the marker while a child holds its descriptors.
        """
        follow_closed_at: float | None = None
        while True:
            with anyio.move_on_after(0.2):
                await self._ended.wait()
            if logs_error:
                raise logs_error[0]
            command = await self._process.get_session_command(
                self._session_id, self._command_id, request_timeout=_REQUEST_TIMEOUT
            )
            if command.exit_code is not None:
                return command.exit_code
            if logs_finished.is_set():
                # The log stream can close before the status catches up: allow a short grace.
                follow_closed_at = follow_closed_at or anyio.current_time()
                if anyio.current_time() - follow_closed_at >= 2:
                    return None
                await anyio.sleep(0.1)
            elif self._ended.is_set():
                await anyio.sleep(0.1)

    async def _result(self, exit_code: int) -> CommandResult:
        out_path, err_path = _output_paths(self._marker)
        try:
            with anyio.fail_after(_REQUEST_TIMEOUT):
                # Sizes first: a command can grow a file without writing it (`truncate`).
                sizes = await self._process.exec(
                    f'for f in {out_path} {err_path}; do if [ -e "$f" ]; then wc -c < "$f"; else echo 0; fi; done',
                    timeout=_REQUEST_TIMEOUT,
                )
                try:
                    out_size, err_size = (int(size) for size in sizes.result.split())
                except ValueError:
                    raise WorkspaceError(f'Could not read the command output sizes: {sizes.result!r}') from None
                if out_size + err_size > _MAX_OUTPUT_BYTES:
                    raise _output_limit_error(*await self._excerpts('head', _OUTPUT_PREVIEW_BYTES))
                stdout = await self._sandbox.fs.download_file(out_path, _REQUEST_TIMEOUT) if out_size else b''
                stderr = await self._sandbox.fs.download_file(err_path, _REQUEST_TIMEOUT) if err_size else b''
                # A background child can keep writing after the sizes were read.
                if len(stdout) + len(stderr) > _MAX_OUTPUT_BYTES:
                    raise _output_limit_error(
                        stdout[:_OUTPUT_PREVIEW_BYTES].decode(errors='replace'),
                        stderr[:_OUTPUT_PREVIEW_BYTES].decode(errors='replace'),
                    )
        except WorkspaceOutputLimitError:
            raise
        except Exception as error:
            await _raise_failure(self._sandbox, error, 'Could not read the command output')
        finally:
            await self._remove_output_files()
        output = stdout.decode(errors='replace')
        if f'{self._marker}-cwd' in output:
            self.missing_cwd = True
            return CommandResult(exit_code=exit_code, stdout='', stderr='')
        return CommandResult(exit_code=exit_code, stdout=output, stderr=stderr.decode(errors='replace'))

    async def _excerpts(self, end: Literal['head', 'tail'], limit: int) -> tuple[str, str]:
        """At most `limit` bytes from one end of each output file, cut in the sandbox before transfer."""
        parts = [f'{path}.part' for path in _output_paths(self._marker)]
        cuts = '; '.join(
            f'{end} -c {limit} -- {path} > {part} 2>/dev/null || : > {part}'
            for path, part in zip(_output_paths(self._marker), parts)
        )
        await self._process.exec(cuts, timeout=_REQUEST_TIMEOUT)
        out, err = [await self._sandbox.fs.download_file(part, _REQUEST_TIMEOUT) for part in parts]
        return out.decode(errors='replace'), err.decode(errors='replace')

    async def _remove_output_files(self) -> None:
        """Best effort: a file left behind only takes space in the sandbox's `/tmp`."""
        paths = [path for output in _output_paths(self._marker) for path in (output, f'{output}.part')]
        with anyio.move_on_after(_REQUEST_TIMEOUT, shield=True):
            try:
                await self._process.exec(f'rm -f -- {" ".join(paths)}', timeout=_REQUEST_TIMEOUT)
            except Exception:
                pass

    async def kill(self) -> None:
        """Delete the session, which kills its command, and keep the tail of its output for a timeout."""
        await _delete_session(self._process, self._session_id)
        if self._finished:
            return  # The result already read and removed the output files.
        with anyio.move_on_after(_REQUEST_TIMEOUT):
            try:
                self.stdout, self.stderr = await self._excerpts('tail', _PARTIAL_OUTPUT_LIMIT)
            except Exception:
                self.stdout, self.stderr = '', ''
        await self._remove_output_files()


async def _close_quietly(client: AsyncDaytona) -> bool:
    """Close an API client, bounded and shielded; log a failure rather than raise it, and report whether it closed."""
    with anyio.move_on_after(_TEARDOWN_TIMEOUT, shield=True) as deadline:
        try:
            await _close_sdk_client(client)
        except Exception:
            _logger.warning('Could not close the Daytona API client.', exc_info=True)
            return False
    if deadline.cancelled_caught:
        _logger.warning('Closing the Daytona API client did not complete within %ss.', _TEARDOWN_TIMEOUT)
        return False
    return True


async def _close_sdk_client(client: AsyncDaytona) -> None:
    """Close an SDK client and its event transport.

    `AsyncDaytona.close()` cancels its event connection task without waiting for it, which can
    leave that connection's HTTP session open, and a failed socket handshake leaves engineio's
    session open, so both are settled here. The attributes are private to the SDK; if they are
    renamed, this falls back to a plain `close()`.
    """
    dispatcher = getattr(client, '_event_dispatcher', None)
    if dispatcher is not None and (connecting := getattr(dispatcher, '_connect_task', None)) is not None:
        with anyio.move_on_after(_EVENT_CONNECT_GRACE, shield=True) as connect_deadline:
            try:
                # The SDK's own asyncio task: wait for it without cancelling it.
                await asyncio.shield(connecting)
            except Exception:
                pass  # A failed connection still needs the normal close.
            except asyncio.CancelledError:
                if not connecting.cancelled():
                    raise
        if connect_deadline.cancelled_caught:
            _logger.warning('Daytona event connection did not settle before client close.')
    sio = getattr(dispatcher, '_sio', None)
    try:
        await client.close()
    finally:
        if (eio := getattr(sio, 'eio', None)) is not None:
            await eio.disconnect()


class DaytonaSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem, SupportsRealpath):
    """A [Daytona](https://www.daytona.io) sandbox as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands and file operations run inside a Daytona sandbox, so the host is never exposed.

    Building one does no I/O. The first operation creates or attaches to a sandbox, and the typed
    `daytona.AsyncSandbox` is available through `get_sandbox()`. The backend does not stop or
    delete the sandbox; that is the application's job.

    Commands run in Daytona process sessions, with complete output returned after they finish. A
    shell string runs under `/bin/sh -c`; an argv sequence is shell-quoted into the session command.
    The deadline is enforced client-side: on timeout or cancellation the session is deleted, which
    kills its command.

    A deleted sandbox raises `WorkspaceUnavailableError`, as do rejected credentials and a sandbox
    Daytona refuses to create (an unknown snapshot, say). Path failures raise the built-in errors
    (`FileNotFoundError`, `PermissionError`, ...); any other request Daytona refuses raises
    `WorkspaceError`; connection failures, rate limits, and other SDK errors propagate unchanged.

    Args:
        sandbox: A live `daytona.AsyncSandbox` you already have. Whoever created it owns deleting it.
        client: A `daytona.AsyncDaytona` API client to create or attach with. The caller owns closing
            it. Without one, the backend opens its own from the environment on first use and closes
            it in `aclose()`.
        ref: Identity of an existing sandbox to attach to on first use.
        snapshot: Daytona snapshot a newly created sandbox starts from; Daytona's default when `None`.
        auto_stop_interval: Idle minutes before Daytona stops a newly created sandbox; `0` disables
            it, and `None` keeps Daytona's default (15 minutes for a container; a VM pauses after 60
            instead).
        allow_internet_access: Whether a newly created sandbox may reach the internet.
        working_dir: Absolute directory commands start in and relative paths resolve against; the
            sandbox's own default when `None`, discovered with `pwd -P` on first use.
        env: Environment variables every command gets; a command's own `env` is layered on top.
            They are also set on a newly created sandbox.
    """

    def __init__(
        self,
        *,
        sandbox: AsyncSandbox | None = None,
        client: AsyncDaytona | None = None,
        ref: WorkspaceRef | None = None,
        snapshot: str | None = None,
        auto_stop_interval: int | None = None,
        allow_internet_access: bool = True,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None and ref.provider != 'daytona':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'daytona'")
        if ref is not None and not ref.id.strip():
            raise ValueError('Daytona workspace ref id cannot be empty')
        if sandbox is not None and ref is not None:
            raise ValueError('pass either `sandbox` or `ref`, not both')
        if env is not None and any(type(key) is not str or type(value) is not str for key, value in env.items()):
            raise TypeError('env keys and values must be strings')
        self._ref = ref if sandbox is None else WorkspaceRef(provider='daytona', id=sandbox.id)
        self._sandbox = sandbox
        self._client = client
        self._owns_client = client is None
        self._snapshot = snapshot
        self._create_name: str | None = None
        self._auto_stop_interval = auto_stop_interval
        self._allow_internet_access = allow_internet_access
        self._env = dict(env or {})
        self._working_dir = absolute_path('working_dir', working_dir)
        # `working_dir()` must return a canonical absolute path, resolved once with `pwd -P`.
        self._resolved_working_dir: str | None = None
        # A sandbox this backend created still needs its `working_dir` made.
        self._working_dir_pending = False
        self._lock = anyio.Lock()
        # Set while nothing will call `aclose()`: the owned client then closes when the last
        # operation in flight ends.
        self._close_after_operation = False
        self._operations = 0
        self._operations_drained = anyio.Event()
        self._operations_drained.set()

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the sandbox, or `None` before one has been created.

        The `id` is always Daytona's sandbox ID, also after attaching by a `ref` that names the sandbox.
        """
        return self._ref

    async def get_sandbox(self) -> AsyncSandbox:
        """Return the typed `daytona.AsyncSandbox`, creating or attaching to it on first use.

        Attaching by `ref` to a sandbox that no longer exists raises `WorkspaceUnavailableError`; it
        does not create a replacement. A sandbox Daytona created is recorded in `ref` even if the
        caller is cancelled meanwhile.
        """
        _require_asyncio()
        async with self._lock:
            if (sandbox := self._sandbox) is None:
                sandbox = await self._acquire_locked()
            if self._working_dir_pending:
                # A caller cancelled during creation stops here; the next operation makes the directory.
                await anyio.lowlevel.checkpoint_if_cancelled()
                await self._make_working_dir(sandbox)
        await anyio.lowlevel.checkpoint_if_cancelled()
        return sandbox

    async def _acquire_locked(self) -> AsyncSandbox:
        """Create or attach to the sandbox; the caller holds `_lock`."""
        if self._client is None:
            self._client = _new_client()
        client = self._client

        async def create() -> AsyncSandbox:
            created = await self._create(client)
            self._record(created)
            self._working_dir_pending = self._working_dir is not None
            # Logged at creation, so the id can be found whatever ends the run before it is stored.
            _logger.info('Created Daytona sandbox %s', created.id)
            return created

        try:
            if self._ref is not None:
                sandbox = await self._attach(client, self._ref.id)
                self._record(sandbox)
                return sandbox
            # Creation finishes even if the caller is cancelled, so a billed sandbox is never left unrecorded.
            return await run_to_completion(create)
        except BaseException:
            if self._sandbox is None:
                # Nothing was acquired: a retry starts from a fresh client.
                await self._close_owned_client()
            raise

    def _record(self, sandbox: AsyncSandbox) -> None:
        self._sandbox = sandbox
        self._ref = WorkspaceRef(provider='daytona', id=sandbox.id)

    async def _make_working_dir(self, sandbox: AsyncSandbox) -> None:
        directory = self._working_dir
        assert directory is not None
        with anyio.CancelScope(shield=True):
            try:
                result = await sandbox.process.exec(f'mkdir -p -- {shlex.quote(directory)}', timeout=_REQUEST_TIMEOUT)
            except Exception as error:
                await _raise_failure(sandbox, error, f'Could not create working_dir {directory!r}')
            if result.exit_code != 0:
                raise WorkspaceError(
                    f'Could not create working_dir {directory!r} in Daytona sandbox {sandbox.id}: '
                    f'{result.result.strip()}'
                )
            self._working_dir_pending = False

    async def aclose(self) -> None:
        """Close the `AsyncDaytona` API client this backend opened for itself.

        A client passed as `client=` is left alone, and the sandbox keeps running. A later
        operation opens a new client and attaches by `ref`. `DaytonaSandbox` calls this when the
        run that used the backend ends.
        """
        if self._client is None or not self._owns_client or not running_on_asyncio():
            return

        async def close() -> None:
            # Let in-flight operations, such as a cancelled command's session deletion, finish first.
            with anyio.move_on_after(_TEARDOWN_TIMEOUT):
                await self._operations_drained.wait()
            async with self._lock:
                if (client := self._client) is not None and await _close_quietly(client):
                    # Kept on a failed close, so the next `aclose()` retries it.
                    self._client, self._sandbox, self._resolved_working_dir = None, None, None

        await run_to_completion(close)

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[None]:
        self._operations += 1
        if self._operations == 1:
            self._operations_drained = anyio.Event()
        try:
            yield
        finally:
            self._operations -= 1
            if self._operations == 0:
                self._operations_drained.set()
            if self._operations == 0 and self._close_after_operation and self._owns_client and self._client is not None:
                # Detached before any await, so an operation starting meanwhile opens its own client.
                client, self._client, self._sandbox, self._resolved_working_dir = self._client, None, None, None
                await run_to_completion(lambda: _close_quietly(client))

    async def _close_owned_client(self) -> None:
        if (client := self._client) is not None and self._owns_client and await _close_quietly(client):
            self._client, self._sandbox, self._resolved_working_dir = None, None, None

    async def realpath(self, path: str) -> str:
        # Native, because the shell fallback would cost a whole session command per call.
        async with self._operation():
            _check_path(path)
            sandbox = await self.get_sandbox()
            try:
                result = await sandbox.process.exec(
                    f'realpath -m -- {shlex.quote(path)}', cwd=self._working_dir, timeout=_REQUEST_TIMEOUT
                )
            except Exception as error:
                await _raise_failure(sandbox, error, f'Could not resolve {path!r}', path=path)
            if result.exit_code != 0:
                raise WorkspaceError(f'Could not resolve {path!r}: {result.result}')
            return result.result.strip()

    async def read_bytes(self, path: str) -> bytes:
        async with self._operation():
            _check_path(path)
            sandbox = await self.get_sandbox()
            async with _translated_filesystem_error(sandbox, path):
                # The toolbox download blocks forever on a FIFO with no writer.
                probe = await sandbox.process.exec(f'[ -p {shlex.quote(path)} ]', timeout=_REQUEST_TIMEOUT)
                if probe.exit_code == 0:
                    raise OSError(f'Cannot read FIFO in the Daytona sandbox: {path!r}')
                try:
                    return await sandbox.fs.download_file(path, _REQUEST_TIMEOUT)
                except daytona.DaytonaError as error:
                    # The toolbox has no distinct error for reading a directory.
                    if await _entry_is_dir(sandbox, path, error):
                        raise IsADirectoryError(f'Is a directory in the Daytona sandbox: {path!r}') from error
                    raise

    async def write_bytes(self, path: str, data: bytes) -> None:
        async with self._operation():
            _check_path(path)
            sandbox = await self.get_sandbox()
            parent = posixpath.dirname(path)
            if parent not in ('', '.', '/'):
                try:
                    mkdir = await sandbox.process.exec(f'mkdir -p -- {shlex.quote(parent)}', timeout=_REQUEST_TIMEOUT)
                except Exception as error:
                    await _raise_failure(sandbox, error, f'Could not create {parent!r}')
                if mkdir.exit_code != 0:
                    raise _mkdir_error(mkdir.result, parent)
            # Written to a sibling stage and renamed over the resolved destination, so the write
            # follows a symlink and stays on one filesystem.
            destination = await self.realpath(path)
            staged = f'{destination}.pydantic-ai-{uuid.uuid4().hex}.tmp'
            async with _translated_filesystem_error(sandbox, path):
                try:
                    info = await sandbox.fs.get_file_info(destination, request_timeout=_REQUEST_TIMEOUT)
                except daytona.DaytonaNotFoundError:
                    pass
                else:
                    if info.is_dir:
                        raise IsADirectoryError(f'Is a directory in the Daytona sandbox: {path!r}')
                try:
                    await sandbox.fs.upload_file(data, staged, timeout=_REQUEST_TIMEOUT)
                    command = (
                        f'if [ -e {shlex.quote(destination)} ] && [ ! -w {shlex.quote(destination)} ]; then '
                        'echo Permission denied >&2; exit 13; fi; '
                        f'if [ -e {shlex.quote(destination)} ]; then '
                        f'chmod --reference={shlex.quote(destination)} -- {shlex.quote(staged)}; fi && '
                        f'mv -fT -- {shlex.quote(staged)} {shlex.quote(destination)}'
                    )
                    result = await sandbox.process.exec(command, timeout=_REQUEST_TIMEOUT)
                    if result.exit_code == 13:
                        raise PermissionError(f'Permission denied in the Daytona sandbox: {path!r}')
                    if result.exit_code != 0:
                        raise WorkspaceError(f'Could not replace {path!r}: {result.result}')
                except BaseException:
                    with anyio.move_on_after(5, shield=True):
                        try:
                            await sandbox.fs.delete_file(staged, request_timeout=_REQUEST_TIMEOUT)
                        except Exception:
                            pass
                    raise

    async def stat(self, path: str) -> FileEntry:
        async with self._operation():
            _check_path(path)
            sandbox = await self.get_sandbox()
            async with _translated_filesystem_error(sandbox, path):
                entry = await sandbox.fs.get_file_info(path, request_timeout=_REQUEST_TIMEOUT)
            return FileEntry(
                name=posixpath.basename(path.rstrip('/')),
                path=path,
                is_dir=entry.is_dir,
                size=None if entry.is_dir else entry.size,
            )

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        async with self._operation():
            _check_path(path)
            sandbox = await self.get_sandbox()
            async with _translated_filesystem_error(sandbox, path):
                try:
                    entries = await sandbox.fs.list_files(path, request_timeout=_REQUEST_TIMEOUT)
                except daytona.DaytonaError as error:
                    if await _entry_is_dir(sandbox, path, error) is False:
                        raise NotADirectoryError(f'Not a directory in the Daytona sandbox: {path!r}') from error
                    raise
            result = [
                FileEntry(
                    name=entry.name,
                    path=posixpath.join(path, entry.name),
                    is_dir=entry.is_dir,
                    size=None if entry.is_dir else entry.size,
                )
                for entry in entries
            ]
            # The toolbox omits looping symlinks from listings; `find -type l` lists them without following them.
            try:
                links = await sandbox.process.exec(
                    f'find {shlex.quote(path)} -mindepth 1 -maxdepth 1 -type l', timeout=_REQUEST_TIMEOUT
                )
            except Exception as error:
                await _raise_failure(sandbox, error, f'Could not list symlinks in {path!r}', path=path)
            if links.exit_code == 0:
                existing = {entry.path for entry in result}
                for link in links.result.splitlines():
                    if link not in existing:
                        result.append(FileEntry(name=posixpath.basename(link), path=link, is_dir=False, size=None))
            return sorted(result, key=lambda entry: entry.name)

    async def make_dir(self, path: str) -> None:
        async with self._operation():
            _check_path(path)
            sandbox = await self.get_sandbox()
            async with _translated_filesystem_error(sandbox, path):
                try:
                    await sandbox.fs.create_folder(path, '755', request_timeout=_REQUEST_TIMEOUT)
                except daytona.DaytonaBadRequestError as error:
                    # The toolbox says "not a directory" for a file at the target and for one in its
                    # parents; a file at the target is `FileExistsError`.
                    if 'not a directory' in str(error).lower():
                        try:
                            info = await sandbox.fs.get_file_info(path, request_timeout=_REQUEST_TIMEOUT)
                        except daytona.DaytonaError:
                            pass
                        else:
                            if not info.is_dir:
                                raise FileExistsError(f'File exists in the Daytona sandbox: {path!r}') from error
                    raise

    async def remove(self, path: str) -> None:
        async with self._operation():
            _check_path(path)
            sandbox = await self.get_sandbox()
            async with _translated_filesystem_error(sandbox, path):
                # Looked up first, so a missing path is `FileNotFoundError`.
                await sandbox.fs.get_file_info(path, request_timeout=_REQUEST_TIMEOUT)
                # The removal is recursive, so refuse the working directory and its ancestors, like the
                # built-in backends. The leaf stays unresolved: removing a link removes only the link.
                root = await self.working_dir()
                parent = await self.realpath(posixpath.dirname(path) or '.')
                target = posixpath.normpath(posixpath.join(parent, posixpath.basename(path)))
                if root == target or root.startswith(target.rstrip('/') + '/'):
                    raise ValueError('cannot remove the workspace root or its ancestor')
                await sandbox.fs.delete_file(path, recursive=True, request_timeout=_REQUEST_TIMEOUT)

    async def exists(self, path: str) -> bool:
        async with self._operation():
            try:
                await self.stat(path)
            except FileNotFoundError:
                return False
            except OSError as error:
                if error.errno == errno.ELOOP:
                    return False
                raise
            return True

    async def _create(self, client: AsyncDaytona) -> AsyncSandbox:
        """Create a sandbox under a name kept across retries, so a lost reply is found rather than duplicated."""
        if self._create_name is None:
            self._create_name = f'pydantic-ai-{uuid.uuid4().hex}'
        name = self._create_name
        try:
            return await client.get(name, request_timeout=_REQUEST_TIMEOUT)
        except daytona.DaytonaNotFoundError:
            pass
        except Exception as error:
            _raise_translated(error, 'Could not start Daytona sandbox')
        params = daytona.CreateSandboxFromSnapshotParams(
            name=name,
            snapshot=self._snapshot,
            env_vars=self._env or None,
            auto_stop_interval=self._auto_stop_interval,
            network_block_all=not self._allow_internet_access,
            labels={'created-by': 'pydantic-ai'},
        )
        with anyio.move_on_after(_CREATE_TIMEOUT):
            try:
                return await client.create(params, timeout=_LIFECYCLE_TIMEOUT)
            except (daytona.DaytonaConnectionError, daytona.DaytonaTimeoutError, TimeoutError):
                # The sandbox may have been created before the reply was lost.
                if (created := await _find(client, name)) is not None:
                    return created
                raise
            except Exception as error:
                translated = _translated(error, 'Could not start Daytona sandbox')
                if translated is error:
                    raise
                if type(translated) is WorkspaceError:
                    # Daytona refused the request (an unknown snapshot, an invalid setting): no retry can fix it.
                    translated = WorkspaceUnavailableError(str(translated))
                raise translated from error
        if (created := await _find(client, name)) is not None:
            return created
        # A stalled control plane is transient, not a command deadline.
        raise TimeoutError(f'Daytona sandbox creation did not complete within {_CREATE_TIMEOUT}s.')

    async def _attach(self, client: AsyncDaytona, sandbox_id: str) -> AsyncSandbox:
        """Attach to a sandbox that already exists, starting it if it is stopped.

        A sandbox in the `destroying` state, which `delete()` leaves until it is gone, counts as gone.
        """
        with anyio.move_on_after(_CREATE_TIMEOUT):
            try:
                sandbox = await client.get(sandbox_id, request_timeout=_REQUEST_TIMEOUT)
                if not _in_deleted_state(sandbox):
                    await sandbox.start(timeout=_LIFECYCLE_TIMEOUT)
            except Exception as error:
                if isinstance(error, daytona.DaytonaNotFoundError):
                    raise WorkspaceUnavailableError(
                        f'The Daytona sandbox {sandbox_id!r} was not found (it was deleted, or never existed '
                        "in this Daytona organization). Pass `workspace='new'` to start a fresh sandbox."
                    ) from error
                _raise_translated(error, f'Could not attach to Daytona sandbox {sandbox_id!r}')
            if _in_deleted_state(sandbox):
                raise WorkspaceUnavailableError(_unavailable_message(sandbox.id))
            return sandbox
        raise TimeoutError(f'Connecting to Daytona sandbox {sandbox_id!r} did not complete within {_CREATE_TIMEOUT}s.')

    async def working_dir(self) -> str:
        """Return the canonical directory commands start in, probed once per connection."""
        async with self._operation():
            if self._resolved_working_dir is None:
                sandbox = await self.get_sandbox()
                try:
                    result = await sandbox.process.exec('pwd -P', cwd=self._working_dir, timeout=_REQUEST_TIMEOUT)
                except Exception as error:
                    await _raise_failure(sandbox, error, 'Could not determine the working directory')
                printed = result.result.removesuffix('\n')
                if result.exit_code != 0 or not posixpath.isabs(printed):
                    raise WorkspaceError(
                        f'Could not determine the working directory of Daytona sandbox {sandbox.id}: {result.result}'
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
        async with self._operation():
            check_timeout(timeout)
            argv = command_argv(command, shell)
            # Acquiring the sandbox has its own bound; the timeout is the command's alone.
            sandbox = await self.get_sandbox()
            process: _DaytonaProcess | None = None

            async def stop() -> None:
                if process is not None:
                    await process.kill()

            def output() -> tuple[str, str]:
                return (process.stdout, process.stderr) if process is not None else ('', '')

            async def attempt() -> CommandResult:
                nonlocal process
                async with command_deadline(timeout, stop=stop, output=output):
                    process = await self._start(sandbox, argv, env)
                    result = await process.wait()
                    if process.missing_cwd:
                        raise WorkspaceError(
                            f'working_dir {self._working_dir!r} does not exist in Daytona sandbox {sandbox.id}. '
                            'Create it there, or pass a working_dir that exists.'
                        )
                return result

            try:
                result = await attempt()
            except WorkspaceTimeoutError:
                raise
            except Exception:
                # Daytona may have stopped the sandbox while idle. Only setup is retried, where no
                # command was accepted, so the command cannot run twice; the restart is not timed.
                if process is not None or not await _restart_if_stopped(sandbox):
                    raise
                result = await attempt()
            # Delete the finished session before the operation ends and the client may close.
            await run_to_completion(lambda: stop_shielded(stop))
            return result

    async def _start(
        self, sandbox: AsyncSandbox, argv: Sequence[str], env: Mapping[str, str] | None
    ) -> _DaytonaProcess:
        marker = f'pydantic-ai-end-{uuid.uuid4().hex}'
        line = _command_line(argv, cwd=self._working_dir, env={**self._env, **(env or {})}, marker=marker)
        session_id = f'pydantic-ai-{uuid.uuid4().hex}'
        process = sandbox.process

        async def setup() -> _DaytonaProcess:
            try:
                with anyio.fail_after(_REQUEST_TIMEOUT):
                    await process.create_session(session_id, request_timeout=_REQUEST_TIMEOUT)
                    response = await process.execute_session_command(
                        session_id,
                        daytona.SessionExecuteRequest(command=line, run_async=True),
                        timeout=_REQUEST_TIMEOUT,
                    )
                return _DaytonaProcess(sandbox, session_id, response.cmd_id, marker)
            except BaseException as error:
                # The session may exist even without an acknowledgement; its ID is the cleanup token.
                await stop_shielded(lambda: _delete_session(process, session_id))
                if isinstance(error, TimeoutError):
                    raise TimeoutError(
                        f'Daytona command session setup did not complete within {_REQUEST_TIMEOUT}s.'
                    ) from error
                if isinstance(error, Exception):
                    await _raise_failure(sandbox, error, 'Could not start the command')
                raise

        # Setup runs in its own task, which owns cleanup of a session accepted before a cancellation.
        task = asyncio.create_task(setup())
        try:
            await asyncio.wait([task])
        except asyncio.CancelledError:
            task.cancel()
            with anyio.CancelScope(shield=True):
                while not task.done():
                    try:
                        await asyncio.wait([task])
                    except asyncio.CancelledError:
                        task.cancel()
                if not task.cancelled() and task.exception() is None:
                    await stop_shielded(task.result().kill)
            await anyio.lowlevel.checkpoint_if_cancelled()
            raise asyncio.CancelledError
        return task.result()


async def _find(client: AsyncDaytona, name: str) -> AsyncSandbox | None:
    """Look a sandbox up by name; `None` when it is missing or the control plane does not answer."""
    try:
        return await client.get(name, request_timeout=_REQUEST_TIMEOUT)
    except (daytona.DaytonaNotFoundError, daytona.DaytonaConnectionError, daytona.DaytonaTimeoutError):
        return None


async def _restart_if_stopped(sandbox: AsyncSandbox) -> bool:
    """Start a sandbox Daytona stopped, archived, or paused while idle; report whether it did.

    An inconclusive state lookup leaves the original error to decide. A failed start propagates a
    transient error unchanged and reports a refused one as `WorkspaceUnavailableError`.
    """
    try:
        with anyio.fail_after(_REQUEST_TIMEOUT):
            await sandbox.refresh_data(request_timeout=_REQUEST_TIMEOUT)
    except Exception:
        return False
    if sandbox.state not in (
        daytona.SandboxState.STOPPED,
        daytona.SandboxState.STOPPING,
        daytona.SandboxState.ARCHIVED,
        daytona.SandboxState.PAUSED,
        daytona.SandboxState.PAUSING,
    ):
        return False
    try:
        with anyio.fail_after(_LIFECYCLE_TIMEOUT):
            await sandbox.start(timeout=_LIFECYCLE_TIMEOUT)
    except Exception as error:
        translated = _translated(error, 'Could not restart the Daytona sandbox')
        if translated is error:
            raise
        if isinstance(translated, WorkspaceUnavailableError):
            raise translated from error
        raise WorkspaceUnavailableError(f'Could not restart stopped Daytona sandbox {sandbox.id!r}.') from error
    return True


def _in_deleted_state(sandbox: AsyncSandbox) -> bool:
    return sandbox.state in (daytona.SandboxState.DESTROYING, daytona.SandboxState.DESTROYED)


def _unavailable_message(sandbox_id: str) -> str:
    return (
        f'The Daytona sandbox {sandbox_id!r} no longer exists (it was deleted). '
        "Pass `workspace='new'` to start a fresh sandbox."
    )


def _translated(error: Exception, context: str, *, path: str | None = None) -> Exception:
    """Map a Daytona SDK error onto the failures the workspace protocol promises.

    Returns `error` itself for what must propagate unchanged: connection failures, SDK timeouts,
    rate limits, server errors, and anything that is not a `DaytonaError`, which durable engines
    retry as transient. A not-found answer is a missing `path` for a path operation, and a refused
    request otherwise.
    """
    if path is not None and isinstance(error, daytona.DaytonaError) and error.status_code == 400:
        # A 400 carries the POSIX `strerror` text.
        message = str(error).lower()
        if 'too many levels of symbolic links' in message:
            return OSError(errno.ELOOP, 'Too many levels of symbolic links in the Daytona sandbox', path)
        for phrase, error_type in (
            ('Not a directory', NotADirectoryError),
            ('Is a directory', IsADirectoryError),
            ('Permission denied', PermissionError),
            ('File exists', FileExistsError),
        ):
            if phrase.lower() in message:
                return error_type(f'{phrase} in the Daytona sandbox: {path!r}')
    if path is not None and isinstance(error, daytona.DaytonaForbiddenError):
        # Toolbox authorization is about the requested file, not the API credentials.
        return PermissionError(f'Permission denied in the Daytona sandbox: {path!r}')
    if isinstance(error, (daytona.DaytonaAuthenticationError, daytona.DaytonaForbiddenError)):
        return WorkspaceUnavailableError(f'{safe_credential_reason(error)}. {_AUTH_MESSAGE}')
    if path is not None and isinstance(error, daytona.DaytonaNotFoundError):
        return FileNotFoundError(f'No such file or directory in the Daytona sandbox: {path!r}')
    if isinstance(error, (daytona.DaytonaNotFoundError, daytona.DaytonaBadRequestError, daytona.DaytonaConflictError)):
        return WorkspaceError(f'{context}: {error}')
    if (
        isinstance(error, daytona.DaytonaError)
        and not isinstance(error, (daytona.DaytonaRateLimitError, daytona.DaytonaTimeoutError))
        and error.status_code is not None
        and 400 <= error.status_code < 500  # a refusal; 5xx and status-less errors are transient
    ):
        return WorkspaceError(f'{context}: {error}')
    return error


def _raise_translated(error: Exception, context: str, *, path: str | None = None) -> NoReturn:
    translated = _translated(error, context, path=path)
    if translated is error:
        raise error
    raise translated from error


async def _raise_failure(sandbox: AsyncSandbox, error: Exception, context: str, *, path: str | None = None) -> NoReturn:
    """Raise the translation of a failed call on a sandbox handle.

    The toolbox answers a missing path and a deleted sandbox with the same not-found error, so
    unless the error already ends the run, one control-plane lookup tells them apart: a deleted
    sandbox raises `WorkspaceUnavailableError`.
    """
    if isinstance(error, daytona.DaytonaError) and 'client is closed' in str(error).lower():
        raise WorkspaceError('Daytona backend was closed while an operation was in flight.') from error
    translated = _translated(error, context, path=path)
    if not isinstance(translated, WorkspaceUnavailableError) and await _is_deleted(sandbox):
        raise WorkspaceUnavailableError(_unavailable_message(sandbox.id)) from error
    if translated is error:
        raise error
    raise translated from error


async def _is_deleted(sandbox: AsyncSandbox) -> bool:
    """Ask the control plane whether the sandbox was deleted; an inconclusive lookup counts as alive."""
    try:
        with anyio.fail_after(_REQUEST_TIMEOUT):
            await sandbox.refresh_data(request_timeout=_REQUEST_TIMEOUT)
    except daytona.DaytonaNotFoundError:
        return True
    except Exception:
        return False
    return _in_deleted_state(sandbox)


def _check_path(path: str) -> None:
    # The toolbox uses newline-delimited paths.
    if '\n' in path or '\r' in path:
        raise ValueError('Daytona file paths cannot contain a newline or carriage return')


async def _entry_is_dir(sandbox: AsyncSandbox, path: str, error: daytona.DaytonaError) -> bool | None:
    """Ask for the entry type when the toolbox reports an ambiguous wrong-type error."""
    if isinstance(error, daytona.DaytonaNotFoundError):
        return None
    try:
        return (await sandbox.fs.get_file_info(path, request_timeout=_REQUEST_TIMEOUT)).is_dir
    except Exception:
        return None


def _mkdir_error(output: str, parent: str) -> Exception:
    """Classify a failed `mkdir -p` by the `strerror` text GNU and BusyBox `mkdir` print."""
    if 'Not a directory' in output or 'File exists' in output:
        return NotADirectoryError(f'Not a directory in the Daytona sandbox: {parent!r}')
    if 'Permission denied' in output:
        return PermissionError(f'Permission denied in the Daytona sandbox: {parent!r}')
    return WorkspaceError(output or f'Could not create {parent!r}.')


@asynccontextmanager
async def _translated_filesystem_error(sandbox: AsyncSandbox, path: str) -> AsyncGenerator[None]:
    """Map Daytona's filesystem errors onto the ones the protocol promises."""
    try:
        yield
    except IsADirectoryError:
        raise
    except Exception as error:
        await _raise_failure(sandbox, error, f'Could not access {path!r} in the sandbox', path=path)
