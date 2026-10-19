"""A Daytona sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

External assumptions last verified 2026-09-08 against Daytona Python SDK 0.198.0:

* `AsyncDaytona.create`, `get`, and `close` and `AsyncSandbox.start` cover the lifecycle used
  here; `get` accepts a sandbox ID or name. The SDK lazily reopens its HTTP session after
  `AsyncDaytona.close()`. Reusing a borrowed client or sandbox after its owner closes it can
  leak an aiohttp session:
  https://www.daytona.io/docs/en/python-sdk/async/async-daytona/
* process sessions provide asynchronous execution, separate stdout and stderr callbacks,
  exit status, and deletion as the per-command kill mechanism:
  https://www.daytona.io/docs/en/python-sdk/async/async-process/
* the session log stream ends a stream that did not end in a newline with one (`printf abc` streams
  `abc` plus a newline; observed live 2026-09-25, and the SDK passes frames through unchanged), so each command
  prints an end marker last on both streams and the output is cut at it.
* the log stream's demultiplexer (`daytona/_utils/stream.py` `_std_demux_loop`) misreads a stream prefix
  that ends a websocket frame, injecting the prefix bytes into the output and misrouting it
  (observed live 2026-09-26 above ~4 KB); the non-follow `get_session_command_logs` is exact, so a
  finished command's output comes from it.
* `sandbox.fs` provides metadata, byte upload/download, and directory operations:
  https://www.daytona.io/docs/en/python-sdk/async/async-file-system/
* `auto_stop_interval` is a creation-time setting; left unset, Daytona stops an idle container
  sandbox after 15 minutes and archives it after 7 days stopped, pauses an idle VM sandbox after
  60 minutes instead, and never deletes either. Starting a paused sandbox resumes it:
  https://www.daytona.io/docs/en/persistence/
* `FileInfo` has `is_dir`, and the SDK does not say whether it follows a symlink.
* SDK errors are typed by HTTP status (`DaytonaNotFoundError` 404, `DaytonaAuthenticationError`
  401, `DaytonaAuthorizationError` 403, `DaytonaValidationError` 400, `DaytonaConflictError` 409,
  `DaytonaRateLimitError` 429); transport failures become `DaytonaConnectionError` or
  `DaytonaTimeoutError`, and anything else a plain `DaytonaError`.

Re-check those sources and the installed 0.198.0 signatures before changing lifecycle,
command, filesystem, or error handling.
"""

from __future__ import annotations

# Native Task.cancel can pierce AnyIO shields; lifecycle and session setup need
# detached asyncio tasks to record accepted sandboxes and finish cleanup.
import asyncio
import logging
import math
import posixpath
import shlex
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

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
    WorkspaceRef,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    command_argv,
    command_deadline,
    run_to_completion,
    running_on_asyncio,
    safe_credential_reason,
    stop_shielded,
)

if TYPE_CHECKING:
    from daytona import AsyncDaytona, AsyncSandbox

    # Not re-exported at the package root; typing-only, so the private path never runs.
    from daytona._async.process import AsyncProcess

    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('DaytonaSandboxBackend',)

try:
    import daytona
except ImportError as error:  # exercised by the isolated missing-extra test
    raise ImportError('Install `pydantic-ai-harness[daytona]` to use DaytonaSandbox.') from error

_AUTH_MESSAGE = (
    'Daytona rejected the credentials. Set DAYTONA_API_KEY, or pass a configured `AsyncDaytona` as `client=`.'
)
_MISSING_CREDENTIALS_MESSAGE = (
    'No Daytona credentials found. Set DAYTONA_API_KEY, or pass a configured `AsyncDaytona` as `client=`.'
)
# Bound sandbox acquisition so a wedged control plane cannot hang creation or connection.
_CREATE_TIMEOUT = 120
# Bound routine SDK requests so a stalled control plane cannot hang an operation.
_REQUEST_TIMEOUT = 30
# Bound provider lifecycle RPCs such as create, start, and delete.
_LIFECYCLE_TIMEOUT = 60.0
# Bound cleanup RPCs so teardown cannot wedge the caller.
_TEARDOWN_TIMEOUT = 30.0
# Bound the wait for the SDK's event connection to settle before its client is closed.
_EVENT_CONNECT_GRACE = 6.0
# Bound the wait for a finished command's stored logs to catch up with its exit status.
_LOGS_SETTLE_GRACE = 5.0
# Pause between attempts to delete a command session.
_RETRY_DELAY = 1.0
# Streamed logs are only a timeout preview; finished output comes from stored logs.
_PARTIAL_OUTPUT_LIMIT = 4096

_logger = logging.getLogger(__name__)


def _require_asyncio() -> None:
    if not running_on_asyncio():
        raise UserError('Daytona needs the asyncio event loop: the Daytona SDK runs its calls on asyncio tasks.')


def _new_client() -> AsyncDaytona:
    """An `AsyncDaytona` configured from the environment: `DAYTONA_API_KEY`, or a JWT token and organization ID."""
    # The SDK installs an ERROR-level engineio handler that logs routine empty queues.
    # Preserve an application-configured level while muting that SDK default.
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
    except (daytona.DaytonaAuthenticationError, daytona.DaytonaAuthorizationError) as error:
        raise WorkspaceUnavailableError(f'{safe_credential_reason(error)}. {_AUTH_MESSAGE}') from error
    finally:
        if owned is not None:
            # Deleting a sandbox must release the API client even if deletion is cancelled.
            with anyio.move_on_after(_TEARDOWN_TIMEOUT, shield=True):
                await _close_sdk_client(owned)


async def _delete_session(process: AsyncProcess, session_id: str) -> None:
    """Delete a command session, which kills its command, even when the caller was cancelled or timed out.

    A surviving session keeps its command running in the sandbox, so a failed deletion is retried
    until `_TEARDOWN_TIMEOUT`, then logged. Its failure must not replace the outcome the caller is
    already raising or returning.
    """
    _deleted = False
    try:
        # The independent stop child supplies cancellation safety; an inner shield would
        # defeat its two-second grace when the control plane keeps refusing deletion.
        with anyio.move_on_after(_TEARDOWN_TIMEOUT):
            while True:
                try:
                    await process.delete_session(session_id, request_timeout=_REQUEST_TIMEOUT)
                    _deleted = True
                    return
                except daytona.DaytonaNotFoundError:
                    _deleted = True
                    return  # already gone, and its command with it
                except Exception:
                    await anyio.sleep(_RETRY_DELAY)
    finally:
        if not _deleted:
            _logger.warning(
                'Could not delete Daytona command session %s; its command may still be running.', session_id
            )


def _command_line(
    command: WorkspaceCommand, *, shell: bool, cwd: str | None, env: Mapping[str, str], marker: str
) -> str:
    """Build the session command: the quoted argv, with the environment and directory applied.

    The argv runs under a `sh` that prints `marker` last on stdout and stderr and exits with the
    argv's status, so `_until_marker` can drop what Daytona appends after a stream's last byte.
    `env` runs inside that `sh`: a `sh` such as dash drops variables whose names are not shell
    identifiers from the environment it passes on.
    """
    argv = command_argv(command, shell)
    if env:
        # `--` ends `env`'s options, so a name starting with `-` is not read as one.
        argv = ['env', '--', *(f'{name}={value}' for name, value in env.items()), *argv]
    # Distinct marker lets run() distinguish a missing cwd from a command exiting with the same status.
    prefix = f'cd -- {shlex.quote(cwd)} || {{ printf %s {marker}-cwd >&2; exit 125; }}; ' if cwd is not None else ''
    script = f'{prefix}"$@" </dev/null; status=$?; printf %s {marker}; printf %s {marker} >&2; exit "$status"'
    return shlex.join(['sh', '-c', script, 'sh', *argv])


def _until_marker(chunks: list[str], marker: str) -> str:
    """The stream's output before `marker`; all of it when the command never printed the marker."""
    output = ''.join(chunks)
    head, found, _ = output.rpartition(marker)
    return head if found else output


def _stream_callbacks(
    marker: str,
) -> tuple[list[str], list[str], anyio.Event, Callable[[str], None], Callable[[str], None]]:
    stdout: list[str] = ['']
    stderr: list[str] = ['']
    ended = anyio.Event()
    tails = ['', '']
    markers_seen = [False, False]

    def on_chunk(chunk: str, index: int, preview: list[str]) -> None:
        # Only the marker can straddle frames; neither detection nor timeout output
        # needs the full follow stream (which grows quadratically when joined).
        candidate = tails[index] + chunk
        if marker in candidate:
            markers_seen[index] = True
        tails[index] = candidate[-(len(marker) - 1) :]
        preview[0] = (preview[0] + chunk)[-_PARTIAL_OUTPUT_LIMIT:]
        if all(markers_seen):
            ended.set()

    def on_stdout(chunk: str) -> None:
        on_chunk(chunk, 0, stdout)

    def on_stderr(chunk: str) -> None:
        on_chunk(chunk, 1, stderr)

    return stdout, stderr, ended, on_stdout, on_stderr


@dataclass(kw_only=True)
class _DaytonaProcess:
    """Output and session identity for a single command."""

    _process: AsyncProcess
    _sandbox: AsyncSandbox
    _session_id: str
    _command_id: str
    stdout: list[str]
    stderr: list[str]
    marker: str
    _ended: anyio.Event
    _on_stdout: Callable[[str], None]
    _on_stderr: Callable[[str], None]

    async def wait(self) -> CommandResult:
        logs_finished = anyio.Event()
        logs_error: list[Exception] = []
        failure: Exception | None = None
        exit_code: int | None = None
        follow_closed_at: float | None = None

        async def read_logs() -> None:
            try:
                await self._process.get_session_command_logs_async(
                    self._session_id, self._command_id, self._on_stdout, self._on_stderr
                )
            except Exception as error:
                logs_error.append(error)
            finally:
                logs_finished.set()

        try:
            # The follow reader belongs to this wait: status errors and cancellation must
            # drain it before the command session or its client can be closed.
            async with anyio.create_task_group() as readers:
                readers.start_soon(read_logs)
                try:
                    try:
                        while True:
                            with anyio.move_on_after(0.2):
                                await self._ended.wait()
                            if logs_error:
                                raise logs_error[0]
                            # The follow websocket may withhold markers while a child holds descriptors;
                            # the polled command status is authoritative in that case.
                            command = await self._process.get_session_command(
                                self._session_id, self._command_id, request_timeout=_REQUEST_TIMEOUT
                            )
                            exit_code = command.exit_code
                            if exit_code is not None:
                                break
                            if logs_finished.is_set():
                                # The SDK can close its follow socket before the status RPC catches
                                # up; allow a short grace, but do not poll forever on a lost status.
                                follow_closed_at = follow_closed_at or anyio.current_time()
                                if anyio.current_time() - follow_closed_at >= 2:
                                    break
                                await anyio.sleep(0.1)
                            elif self._ended.is_set():
                                await anyio.sleep(0.1)
                    except Exception as error:
                        failure = error
                finally:
                    readers.cancel_scope.cancel()
            if failure is not None:
                raise failure
        except Exception as error:
            await _raise_failure(self._sandbox, error, 'Could not read the command result')
        if exit_code is None:
            raise WorkspaceError('Daytona closed the command output before reporting an exit status.')
        # The streamed copy is only for partial output on a timeout: SDK 0.198.0's stream
        # demultiplexer misreads a stream prefix split across websocket frames, corrupting output
        # over a few KB. The finished command's stored logs are exact.
        stdout, stderr = await self._stored_logs(exit_code)
        if f'{self.marker}-cwd' in stderr:
            return CommandResult(exit_code=exit_code, stdout='', stderr=f'{self.marker}-cwd')
        return CommandResult(
            exit_code=exit_code,
            stdout=_until_marker([stdout], self.marker),
            stderr=_until_marker([stderr], self.marker),
        )

    async def _stored_logs(self, exit_code: int) -> tuple[str, str]:
        """The finished command's stored stdout and stderr, once they are complete.

        The stored logs can trail the exit status; the wrapper's end markers show they are complete.
        """
        settle_deadline = anyio.current_time() + _LOGS_SETTLE_GRACE
        while True:
            try:
                logs = await self._process.get_session_command_logs(
                    self._session_id, self._command_id, request_timeout=_REQUEST_TIMEOUT
                )
            except Exception as error:
                await _raise_failure(self._sandbox, error, 'Could not read the command output')
            stdout, stderr = logs.stdout or '', logs.stderr or ''
            if f'{self.marker}-cwd' in stderr or (self.marker in stdout and self.marker in stderr):
                return stdout, stderr
            if anyio.current_time() >= settle_deadline:
                if exit_code == 0:
                    raise WorkspaceError('Daytona returned incomplete output for a finished command.')
                return stdout, stderr  # The wrapper itself was killed, so no marker is coming.
            await anyio.sleep(0.2)

    async def kill(self) -> None:
        """Delete the Daytona process session, which kills its command."""
        await _delete_session(self._process, self._session_id)


async def _close_quietly(client: AsyncDaytona) -> bool:
    """Close an API client, bounded and shielded so a cancelled run still releases it; report whether it closed.

    A failure is logged, not raised: it must not replace the outcome of the run being cleaned up.
    """
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
    """Close an SDK client and its event transport even after a failed socket handshake."""
    # SDK 0.198.0 cancels its event connect task in close() without joining it.
    # If it has opened engineio's aiohttp session, disconnect cannot close it yet.
    # These are private SDK attributes; if a release renames them, fall back to a plain close.
    dispatcher = getattr(client, '_event_dispatcher', None)
    if dispatcher is not None and (connecting := getattr(dispatcher, '_connect_task', None)) is not None:
        with anyio.move_on_after(_EVENT_CONNECT_GRACE, shield=True) as connect_deadline:
            try:
                # This is an asyncio-only SDK task; do not cancel it on our grace timeout.
                await asyncio.shield(connecting)
            except Exception:
                pass  # A failed connection still needs the normal SDK close.
            except asyncio.CancelledError:
                # A connection the SDK already cancelled still needs the normal SDK close.
                if not connecting.cancelled():
                    raise
        if connect_deadline.cancelled_caught:
            _logger.warning('Daytona event connection did not settle before client close.')
    sio = getattr(dispatcher, '_sio', None)
    try:
        await client.close()
    finally:
        if (eio := getattr(sio, 'eio', None)) is not None:
            # On a failed socket handshake socketio.disconnect() skips engineio's
            # aiohttp session; explicitly disconnect the underlying transport.
            await eio.disconnect()


class DaytonaSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem, SupportsRealpath):
    """A [Daytona](https://www.daytona.io) sandbox as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands and file operations run inside a Daytona sandbox, so the host is never exposed.

    Building one does no I/O. The first operation creates or attaches to a sandbox, and the typed
    `daytona.AsyncSandbox` is available through `get_sandbox()`. The backend does not stop or
    delete the sandbox; that is the application's job, through the native handle.

    Commands run in Daytona process sessions, with complete output returned after they finish. A
    shell string runs under `/bin/sh -c`; an argv sequence is shell-quoted into the session command. The deadline is enforced
    client-side, and the session is deleted, which kills its command, when the deadline expires or
    the caller is cancelled.

    Daytona answers a request for a missing path and a request to a deleted sandbox with the same
    not-found error, so a failed request is followed by one control-plane lookup of the sandbox:
    a missing path raises `FileNotFoundError`, a deleted sandbox raises `WorkspaceUnavailableError`.
    Rejected credentials and a sandbox Daytona refuses to create (an unknown snapshot, say) raise
    `WorkspaceUnavailableError`; any other request Daytona refuses raises `WorkspaceError`;
    connection failures, rate limits, and other SDK errors propagate unchanged.

    The protocol is structural, but subclassing it here makes a signature drift fail the type
    check on this class instead of at a distant workspace call.

    Args:
        workspace: A live `daytona.AsyncSandbox` you already have. Whoever created it owns deleting it.
        client: A `daytona.AsyncDaytona` API client to create or attach with. The caller owns closing
            it. Without one, the backend opens its own from the environment on first use and closes
            it in `aclose()`.
        ref: Identity of an existing sandbox to attach to on first use.
        snapshot: Daytona snapshot a newly created sandbox starts from; Daytona's default when `None`.
        auto_stop_interval: Idle minutes before Daytona stops a newly created sandbox; `0` disables
            it, and `None` keeps Daytona's default (15 minutes for a container; a VM pauses after 60
            instead).
        working_dir: Absolute directory commands start in and relative paths resolve against; the
            sandbox's own default when `None`, discovered with `pwd -P` on first use.
        env: Environment variables every command gets; a command's own `env` is layered on top.
            They are also set on a newly created sandbox.
        network_block_all: Whether a newly created sandbox is blocked from outbound network access.
    """

    def __init__(
        self,
        *,
        sandbox: AsyncSandbox | None = None,
        client: AsyncDaytona | None = None,
        ref: WorkspaceRef | None = None,
        snapshot: str | None = None,
        auto_stop_interval: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        network_block_all: bool = False,
    ) -> None:
        if ref is not None and ref.provider != 'daytona':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'daytona'")
        if ref is not None and not ref.id.strip():
            raise ValueError('Daytona workspace ref id cannot be empty')
        if sandbox is not None and ref is not None:
            raise ValueError('pass either `sandbox` or `ref`, not both')
        self._ref = ref if sandbox is None else WorkspaceRef(provider='daytona', id=sandbox.id)
        self._sandbox = sandbox
        self._client = client
        self._owns_client = client is None
        self._snapshot = snapshot
        self._create_name: str | None = None
        self._auto_stop_interval = auto_stop_interval
        self._env = dict(env or {})
        self._network_block_all = network_block_all
        self._working_dir = absolute_path('working_dir', working_dir)
        # `pwd -P` of `_working_dir` (or of the image default): the protocol needs a canonical absolute path.
        self._resolved_working_dir: str | None = None
        # A sandbox this backend creates starts without `working_dir`; it is made on first acquisition.
        self._working_dir_pending = False
        self._lock = anyio.Lock()
        # Set while nothing will call `aclose()`: the owned client then closes when the last
        # operation in flight ends, so a dropped backend leaves no open connection behind.
        self._close_after_operation = False
        self._operations = 0
        self._operations_drained = anyio.Event()
        self._operations_drained.set()

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the sandbox, or `None` before one has been created.

        The `id` is always Daytona's sandbox ID. Attaching by a `ref` whose `id` is a sandbox name
        also works, since Daytona looks sandboxes up by ID or name, and the ref is then rewritten to
        the ID so it stays valid if the sandbox is renamed.
        """
        return self._ref

    async def get_sandbox(self) -> AsyncSandbox:
        """Return the typed `daytona.AsyncSandbox`, creating or attaching to it on first use.

        This is the sandbox handle, not the `AsyncDaytona` API client passed as `client=`.

        The only place `_client` and `_sandbox` are read, so nothing can reach an
        unhydrated one: both stay optional and every other method comes through here.
        The lock serializes concurrent first uses -- two callers each creating a sandbox
        would leave the loser billed and unreferenced. A failed acquisition releases an
        API client this backend owns, so a retry starts from a clean one. Attaching by `ref`
        to a sandbox that no longer exists raises `WorkspaceUnavailableError`; it does not
        create a replacement. Creation is not cut short by cancellation, so
        a sandbox Daytona created is always recorded in `ref` before the cancellation propagates.
        """
        _require_asyncio()
        cancelled = False
        async with self._lock:
            if (sandbox := self._sandbox) is None:
                sandbox, cancelled = await self._acquire_locked()
            if self._working_dir_pending and not cancelled:
                # Still pending after a failed attempt, so the next acquisition retries it.
                await self._make_working_dir(sandbox)
        # A caller cancelled during the shielded creation stops here, with the sandbox recorded.
        await anyio.lowlevel.checkpoint_if_cancelled()
        if cancelled:
            raise asyncio.CancelledError
        return sandbox

    async def _acquire_locked(self) -> tuple[AsyncSandbox, bool]:
        """Create or attach to the sandbox; the caller holds `_lock`.

        Also returns whether the caller was cancelled while a creation it started was completing.
        """
        ref = self._ref
        cancelled = False
        try:
            if self._client is None:
                self._client = _new_client()
            client = self._client
            if ref is not None:
                sandbox = await self._attach(client, ref.id)
            else:

                async def create_and_record() -> AsyncSandbox:
                    created = await self._create(client)
                    # A native Task.cancel interrupts inline shields; record the accepted ID in
                    # the independent task before allowing cancellation to escape the caller.
                    self._ref = WorkspaceRef(provider='daytona', id=created.id)
                    # Logged at creation, so the id can be found whatever ends the run before it is stored.
                    _logger.info('Created Daytona sandbox %s', created.id)
                    return created

                # Keep the child handle until creation completes even after native cancellation.
                task = asyncio.create_task(create_and_record())
                while True:
                    try:
                        sandbox = await asyncio.shield(task)
                        break
                    except asyncio.CancelledError:
                        cancelled = True
                        if task.done():
                            sandbox = task.result()
                            break
                self._working_dir_pending = self._working_dir is not None
        except BaseException:
            await self._close_owned_client()
            raise
        self._sandbox = sandbox
        # `client.get` also accepts a name; record the ID it resolved to.
        self._ref = WorkspaceRef(provider='daytona', id=sandbox.id)
        return sandbox, cancelled

    async def _make_working_dir(self, sandbox: AsyncSandbox) -> None:
        directory = self._working_dir
        assert directory is not None
        # Shielded so a cancelled first operation cannot leave the directory half made and the flag unset.
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

        A client passed as `client=` is left alone, and the sandbox keeps running. A sandbox
        SDK handles lazily reopen their HTTP sessions after close; this backend drops its owned
        client and handle so a later operation opens a new client and attaches by `ref`. Using a
        `client=` or `sandbox=` backend after its caller closes that client leaks an aiohttp session.
        `DaytonaSandbox` calls this when the run that used the backend ends.
        """
        # Under another event loop every operation refused before opening a client: nothing to close.
        if self._client is None or not self._owns_client or not running_on_asyncio():
            return

        async def close() -> None:
            # Agent cancellation may unwind the run before a tool's stop child deletes its session,
            # and a caller may close while a file operation is in flight. Give them a chance to finish
            # before closing their HTTP client.
            with anyio.move_on_after(_TEARDOWN_TIMEOUT):
                await self._operations_drained.wait()
            async with self._lock:
                if (client := self._client) is not None and await _close_quietly(client):
                    # Dropped only once closed, so a failed close is retried by the next `aclose()`. The
                    # next operation reattaches and checks the working directory again, so a sandbox
                    # deleted meanwhile is reported rather than hidden by the cached path.
                    self._client, self._sandbox, self._resolved_working_dir = None, None, None

        # Finished even when the caller (a run being cancelled) is cancelled meanwhile.
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
        """Close the API client this backend opened after a failed acquisition, so a retry starts from a clean one."""
        if (client := self._client) is not None and self._owns_client and await _close_quietly(client):
            self._client, self._sandbox, self._resolved_working_dir = None, None, None

    async def realpath(self, path: str) -> str:
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
                # The toolbox download opens FIFOs for reading and can wait forever for a writer.
                probe = await sandbox.process.exec(f'[ -p {shlex.quote(path)} ]', timeout=_REQUEST_TIMEOUT)
                if probe.exit_code == 0:
                    raise OSError(f'Cannot read FIFO in the Daytona sandbox: {path!r}')
                try:
                    return await sandbox.fs.download_file(path, _REQUEST_TIMEOUT)
                except daytona.DaytonaError as error:
                    # The toolbox's answer for reading a directory is not a documented error type; the
                    # entry type tells it apart from other failures.
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
            # Resolve the destination before renaming: replacing the symlink itself would change
            # write_bytes' previous follow-symlink behavior. A sibling stage keeps rename on one filesystem.
            destination = await self.realpath(path)
            staged = f'{destination}.pydantic-ai-{uuid.uuid4().hex}.tmp'
            async with _translated_filesystem_error(sandbox, path):
                # Staging bypasses the toolbox's directory check on the original destination.
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
                    # SIGKILL cannot clean an in-flight upload, but ordinary errors and cancellation can.
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
            # Daytona's toolbox silently omits looping symlinks (verified against live SDK);
            # find sees the directory entries without following their targets.
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
                except daytona.DaytonaValidationError as error:
                    # Toolbox says "not a directory" for both a file at the target and
                    # a file in its parents; a target file is the protocol's FileExistsError.
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
                # Whether the toolbox rejects removing a missing path is not documented; looking it up
                # first reports that as the protocol's `FileNotFoundError` either way.
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
            except WorkspaceError as error:
                if isinstance(error.__cause__, daytona.DaytonaValidationError) and (
                    'Too many levels of symbolic links' in str(error.__cause__)
                ):
                    return False
                raise
            return True

    async def _create(self, client: AsyncDaytona) -> AsyncSandbox:
        # Keep the same unique name across retries: a lost create reply must not orphan the
        # accepted sandbox or launch a second one.
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
            network_block_all=self._network_block_all,
            labels={'created-by': 'pydantic-ai'},
        )
        # The create RPC can outlive its acknowledgement; recover by the stable name on timeout.
        with anyio.move_on_after(_CREATE_TIMEOUT):
            try:
                return await client.create(params, timeout=_LIFECYCLE_TIMEOUT)
            except Exception as error:
                if isinstance(error, (daytona.DaytonaConnectionError, daytona.DaytonaTimeoutError, TimeoutError)):
                    try:
                        return await client.get(name, request_timeout=_REQUEST_TIMEOUT)
                    except (daytona.DaytonaNotFoundError, daytona.DaytonaConnectionError, daytona.DaytonaTimeoutError):
                        pass
                translated = _translated(error, 'Could not start Daytona sandbox')
                if translated is error:
                    raise
                if type(translated) is WorkspaceError:
                    # Daytona refused the request (an unknown snapshot, an invalid setting): no retry or
                    # model turn can fix it, so the run ends instead of handing the model an error.
                    translated = WorkspaceUnavailableError(str(translated))
                raise translated from error
        # Every path through the block returns or raises, so reaching here means the deadline expired.
        try:
            return await client.get(name, request_timeout=_REQUEST_TIMEOUT)
        except (daytona.DaytonaNotFoundError, daytona.DaytonaConnectionError, daytona.DaytonaTimeoutError):
            pass
        # A stalled control plane is transient, not a command deadline.
        raise TimeoutError(f'Daytona sandbox creation did not complete within {_CREATE_TIMEOUT}s.')

    async def _attach(self, client: AsyncDaytona, sandbox_id: str) -> AsyncSandbox:
        """Attach to a sandbox that already exists, starting it if it is stopped.

        `delete()` returns once Daytona accepts the request, and the sandbox stays visible in the
        `destroying` state until it is gone, so that state counts as gone here rather than being
        started.
        """
        with anyio.move_on_after(_CREATE_TIMEOUT):
            try:
                sandbox = await client.get(sandbox_id, request_timeout=_REQUEST_TIMEOUT)
                if not _in_deleted_state(sandbox):
                    await sandbox.start(timeout=_LIFECYCLE_TIMEOUT)
            except Exception as error:
                if isinstance(error, daytona.DaytonaNotFoundError):
                    # A missing ID may never have existed; do not claim confirmed deletion.
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
        """Return the filesystem-canonical default directory inside the workspace.

        The probe runs only on the first call per connection. If the sandbox was deleted, that
        call raises `WorkspaceUnavailableError`; later calls return the cached path without probing.
        """
        async with self._operation():
            if self._resolved_working_dir is None:
                sandbox = await self.get_sandbox()
                try:
                    result = await sandbox.process.exec('pwd -P', cwd=self._working_dir, timeout=_REQUEST_TIMEOUT)
                except Exception as error:
                    await _raise_failure(sandbox, error, 'Could not determine the working directory')
                printed = result.result.removesuffix('\n')
                if result.exit_code != 0 or not posixpath.isabs(printed):
                    raise WorkspaceUnavailableError(
                        f'Could not determine the working directory of Daytona sandbox {sandbox.id}: {result.result}'
                    )
                self._resolved_working_dir = printed
            return self._resolved_working_dir

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        async with self._operation():
            if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
                raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
            argv = command_argv(command, shell)
            checked_cwd = absolute_path('cwd', cwd)
            # Acquiring the sandbox has its own bound; the timeout is the command's alone.
            sandbox = await self.get_sandbox()
            process: _DaytonaProcess | None = None

            async def stop() -> None:
                if process is not None:
                    await process.kill()

            def output() -> tuple[str, str]:
                if process is None:
                    return '', ''
                return _until_marker(process.stdout, process.marker), _until_marker(process.stderr, process.marker)

            async with command_deadline(timeout, stop=stop, output=output):
                try:
                    process = await self._start(sandbox, argv, shell=False, cwd=checked_cwd, env=env)
                except Exception:
                    # A held handle may outlive Daytona's idle auto-stop; only retry setup,
                    # where no command was accepted, so the command cannot be duplicated.
                    if not await _restart_if_stopped(sandbox):
                        raise
                    process = await self._start(sandbox, argv, shell=False, cwd=checked_cwd, env=env)
                result = await process.wait()
                if f'{process.marker}-cwd' in result.stderr:
                    # The command never started: the directory it was to start in is missing.
                    raise FileNotFoundError(checked_cwd or self._working_dir)
            # Native Task.cancel() pierces AnyIO shields. Keep the bounded stop in a child
            # and wait for it before reporting operations drained or releasing the HTTP client.
            await run_to_completion(lambda: stop_shielded(stop))
            return result

    async def _start(
        self,
        sandbox: AsyncSandbox,
        command: WorkspaceCommand,
        *,
        shell: bool,
        cwd: str | None,
        env: Mapping[str, str] | None,
    ) -> _DaytonaProcess:
        marker = f'pydantic-ai-end-{uuid.uuid4().hex}'
        line = _command_line(
            command,
            shell=shell,
            cwd=absolute_path('cwd', cwd) if cwd is not None else self._working_dir,
            env={**self._env, **(env or {})},
            marker=marker,
        )
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
                stdout, stderr, ended, on_stdout, on_stderr = _stream_callbacks(marker)

                return _DaytonaProcess(
                    _process=process,
                    _sandbox=sandbox,
                    _session_id=session_id,
                    _command_id=response.cmd_id,
                    stdout=stdout,
                    stderr=stderr,
                    marker=marker,
                    _ended=ended,
                    _on_stdout=on_stdout,
                    _on_stderr=on_stderr,
                )
            except BaseException as error:
                # The server may commit create_session before its acknowledgement. The preallocated
                # ID is the cleanup token even if setup times out or the caller is cancelled.
                await stop_shielded(lambda: _delete_session(process, session_id))
                if isinstance(error, TimeoutError):
                    raise TimeoutError(
                        f'Daytona command session setup did not complete within {_REQUEST_TIMEOUT}s.'
                    ) from error
                if isinstance(error, Exception):
                    await _raise_failure(sandbox, error, 'Could not start the command')
                raise

        # Native cancellation bypasses an inline AnyIO shield; let setup settle in its own task.
        # The setup child owns cleanup of a session accepted before cancellation.
        task = asyncio.create_task(setup())
        cancelled = False
        result: _DaytonaProcess | None = None
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            task.cancel()
            # Cleanup belongs to the child, so parent scope cancellation cannot cut it off.
            with anyio.CancelScope(shield=True):
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        task.cancel()
                    except Exception:  # pragma: no cover - only an SDK that reports a cancellation as an error
                        break
                if not task.cancelled() and task.exception() is None:
                    await stop_shielded(task.result().kill)
        if cancelled:
            await anyio.lowlevel.checkpoint_if_cancelled()
            raise asyncio.CancelledError
        assert result is not None
        return result


async def _restart_if_stopped(sandbox: AsyncSandbox) -> bool:
    """Start a sandbox Daytona stopped, archived, or paused (a VM) while idle; report whether it did.

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
        # Both typed validation errors and upload's plain 400 include POSIX strerror text.
        for phrase, error_type in (
            ('Not a directory', NotADirectoryError),
            ('Is a directory', IsADirectoryError),
            ('Permission denied', PermissionError),
            ('File exists', FileExistsError),
        ):
            if phrase.lower() in str(error).lower():
                return error_type(f'{phrase} in the Daytona sandbox: {path!r}')
    if path is not None and isinstance(error, daytona.DaytonaAuthorizationError):
        # Toolbox authorization is about the requested file, not the API credentials.
        return PermissionError(f'Permission denied in the Daytona sandbox: {path!r}')
    if isinstance(error, (daytona.DaytonaAuthenticationError, daytona.DaytonaAuthorizationError)):
        return WorkspaceUnavailableError(f'{safe_credential_reason(error)}. {_AUTH_MESSAGE}')
    if path is not None and isinstance(error, daytona.DaytonaNotFoundError):
        return FileNotFoundError(f'No such file or directory in the Daytona sandbox: {path!r}')
    if isinstance(error, (daytona.DaytonaNotFoundError, daytona.DaytonaValidationError, daytona.DaytonaConflictError)):
        return WorkspaceError(f'{context}: {error}')
    if (
        type(error) is daytona.DaytonaError
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
    """Raise the translation of a failed call on a live sandbox handle.

    Unless the error already ends the run, one control-plane lookup follows: whatever the
    toolbox answered, a deleted sandbox is reported as `WorkspaceUnavailableError`. On a live
    sandbox, a not-found answer is a missing `path`, or a refused request (such as a command
    session that no longer exists) for a call without one.
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
    """Ask the control plane whether the sandbox was deleted.

    Only called after a failure, so successful operations make no extra request. An inconclusive
    lookup counts as alive, leaving the original error to decide.
    """
    try:
        with anyio.fail_after(_REQUEST_TIMEOUT):
            await sandbox.refresh_data(request_timeout=_REQUEST_TIMEOUT)
    except daytona.DaytonaNotFoundError:
        return True
    except Exception:
        return False
    return _in_deleted_state(sandbox)


def _check_path(path: str) -> None:
    # Daytona's toolbox uses newline-delimited paths; embedded line breaks change the request's meaning.
    if '\n' in path or '\r' in path:
        raise ValueError('Daytona file paths cannot contain a newline or carriage return')


async def _entry_is_dir(sandbox: AsyncSandbox, path: str, error: daytona.DaytonaError) -> bool | None:
    """Ask for the entry type when Daytona's toolbox reports an ambiguous wrong-type error."""
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
