"""A Boat sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

Every command runs as a detached Boat command: it has a process ID that a timeout or a cancellation
can stop, no server-side time limit, and Boat reports its exit once its own process exits.
"""

from __future__ import annotations

import base64
import errno
import logging
import os
import posixpath
import shlex
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote

import anyio
import anyio.lowlevel
import httpx

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
    WorkspaceUnavailableError,
)
from pydantic_ai.workspaces.workspace import _ShellFilesystem  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    check_env,
    check_timeout,
    command_argv,
    command_deadline,
    run_to_completion,
    running_on_asyncio,
    safe_credential_reason,
)
from pydantic_ai_harness.boat_sandbox import _api

if TYPE_CHECKING:
    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('BoatSandboxBackend',)

_AUTH_MESSAGE = (
    'Boat rejected the credentials. Set BOAT_API_KEY, or pass a configured `httpx.AsyncClient` as `client=`.'
)
_MISSING_CREDENTIALS_MESSAGE = (
    'No Boat credentials found. Set BOAT_API_KEY, or pass a configured `httpx.AsyncClient` as `client=`.'
)
_OUTPUT_LIMIT_MESSAGE = (
    'Boat command output exceeded the 8 MiB limit per stream; redirect it to a file and read part of it instead'
)
# Bound sandbox creation and attachment so a wedged control plane cannot hang acquisition.
_CREATE_TIMEOUT = 120
# Boat gives a stopped command about 5 seconds to exit before it kills it, and answers after that.
_KILL_TIMEOUT = 15
_TEARDOWN_TIMEOUT = 30.0
# Internal commands (`pwd -P`, the shell file operations) run synchronously; Boat stops one after this.
_HELPER_TIMEOUT = 30
# Between attempts to create a sandbox whose earlier request may have been lost.
_RETRY_DELAY = 1.0
# Between checks of a sandbox that is starting, and of a running command.
_READY_POLL = 0.5
_POLL_INTERVAL = 0.2
# The most partial output a timeout reports per stream, which is all each check of a running command reads.
_PARTIAL_OUTPUT_LIMIT = 4096
# Boat keeps the last 8 MiB of each stream of a command.
_MAX_OUTPUT_BYTES = 8 * 1024 * 1024
_OUTPUT_PREVIEW_BYTES = 64 * 1024
# Boat's file API moves at most 5 MiB in one call; a larger file goes through a signed transfer URL.
_MAX_INLINE_FILE_BYTES = 5 * 1024 * 1024

_READY_STATES = frozenset({'ready', 'idle', 'running'})
_STOPPED_STATES = frozenset({'archiving', 'archived'})
_FAILED_STATES = frozenset({'error', 'cancelled'})
# The POSIX error a file call failed with, from the start of Boat's message; a write below a file
# fails creating the file as a directory.
_PATH_ERRNOS = {
    'ENOENT': errno.ENOENT,
    'ENOTDIR': errno.ENOTDIR,
    'EISDIR': errno.EISDIR,
    'EEXIST': errno.ENOTDIR,
    'EACCES': errno.EACCES,
    'ELOOP': errno.ELOOP,
}

logger = logging.getLogger(__name__)


def _require_asyncio() -> None:
    if not running_on_asyncio():
        raise UserError(
            'Boat needs the asyncio event loop: the backend finishes sandbox creation and command cleanup '
            'on asyncio tasks of its own.'
        )


def _new_client() -> httpx.AsyncClient:
    """An API client authenticated with `BOAT_API_KEY`."""
    key = os.environ.get('BOAT_API_KEY', '').strip()
    if not key:
        raise WorkspaceUnavailableError(_MISSING_CREDENTIALS_MESSAGE)
    return _api.open_client(key)


@dataclass(frozen=True)
class _Sandbox:
    """A Boat sandbox reached through one API client."""

    client: httpx.AsyncClient
    id: str

    def path(self, suffix: str = '') -> str:
        return f'/sandboxes/{quote(self.id, safe="")}{suffix}'


async def destroy_sandbox(client: httpx.AsyncClient | None, sandbox_id: str) -> None:
    """Delete a sandbox by ID without resuming it, including one Boat has stopped.

    A sandbox that no longer exists returns quietly; rejected credentials raise
    `WorkspaceUnavailableError`, like any other operation. Without `client`, one is opened from
    the environment and closed afterwards.

    Raises:
        UserError: The event loop is not asyncio.
    """
    _require_asyncio()
    if not sandbox_id.strip():
        raise ValueError('Boat workspace ref id cannot be empty')
    owned: httpx.AsyncClient | None = None
    if client is None:
        client = owned = _new_client()
    sandbox = _Sandbox(client, sandbox_id)
    try:
        async with _mapped_errors(f'Could not delete Boat sandbox {sandbox_id!r}'):
            try:
                await _api.call(
                    client, 'DELETE', sandbox.path(), headers={'X-Ascii-Confirm-Delete': sandbox_id}, model=_api.Empty
                )
            except _api.BoatAPIError as error:
                if error.status != 404:
                    raise
    finally:
        if owned is not None:
            await _close_quietly(owned)


def _not_found_message(sandbox_id: str) -> str:
    return (
        f'The Boat sandbox {sandbox_id!r} was not found (it was deleted, or never existed in this Boat '
        "account). Pass `workspace='new'` to start a fresh sandbox."
    )


def _unavailable_message(sandbox_id: str) -> str:
    return (
        f'The Boat sandbox {sandbox_id!r} no longer exists (it was deleted). '
        "Pass `workspace='new'` to start a fresh sandbox."
    )


def _stopped_message(sandbox_id: str) -> str:
    return (
        f'The Boat sandbox {sandbox_id!r} was stopped; a later run that attaches to it resumes it. '
        "Pass `workspace='new'` to start a fresh sandbox."
    )


def _failed_message(sandbox_id: str, state: str) -> str:
    return (
        f'The Boat sandbox {sandbox_id!r} is in the {state!r} state and cannot be used. '
        "Pass `workspace='new'` to start a fresh sandbox."
    )


def _missing_cwd(cwd: str | None, sandbox_id: str) -> WorkspaceError:
    return WorkspaceError(
        f'working_dir {cwd!r} does not exist in Boat sandbox {sandbox_id}. '
        'Create it there, or pass a working_dir that exists.'
    )


def _translate(error: Exception, context: str, *, sandbox_id: str | None = None, path: str | None = None) -> Exception:
    """Map a Boat API error onto the failures the workspace protocol promises.

    Returns `error` itself for what must propagate unchanged: transport failures, rate limits,
    server errors, and anything that is not a `BoatAPIError`, which durable engines retry as
    transient.
    """
    if not isinstance(error, _api.BoatAPIError):
        return error
    if error.status == 401 or (error.status == 403 and error.code == 'forbidden'):
        return WorkspaceUnavailableError(f'{safe_credential_reason(error)}. {_AUTH_MESSAGE}')
    if error.status in (402, 403):
        # Billing and plan refusals, which no retry fixes.
        return WorkspaceUnavailableError(f'{context}: {error.message}')
    if sandbox_id is not None and error.status == 404 and error.code == 'not_found':
        return WorkspaceUnavailableError(_unavailable_message(sandbox_id))
    if sandbox_id is not None and error.status == 409 and error.code in ('machine_not_running', 'sandbox_not_ready'):
        return WorkspaceUnavailableError(_stopped_message(sandbox_id))
    if path is not None and error.status == 400:
        number = _PATH_ERRNOS.get(error.message.partition(':')[0])
        if number is not None:
            return OSError(number, f'{os.strerror(number)} in the Boat sandbox', path)
    if 400 <= error.status < 500 and error.status != 429:
        return WorkspaceError(f'{context}: {error}')
    return error


def _state_error(sandbox_id: str, state: str) -> WorkspaceUnavailableError | None:
    if state in _STOPPED_STATES:
        return WorkspaceUnavailableError(_stopped_message(sandbox_id))
    if state in _FAILED_STATES:
        return WorkspaceUnavailableError(_failed_message(sandbox_id, state))
    return None


async def _probe(sandbox: _Sandbox) -> WorkspaceUnavailableError | None:
    """`WorkspaceUnavailableError` if Boat deleted or stopped `sandbox`; an inconclusive lookup counts as running."""
    try:
        info = await _api.call(sandbox.client, 'GET', sandbox.path(), model=_api.SandboxResponse)
    except _api.BoatAPIError as error:
        return WorkspaceUnavailableError(_unavailable_message(sandbox.id)) if error.status == 404 else None
    except Exception:
        return None
    return _state_error(sandbox.id, info.sandbox.state)


@asynccontextmanager
async def _mapped_errors(
    context: str, *, sandbox: _Sandbox | None = None, path: str | None = None
) -> AsyncGenerator[None]:
    """Raise the protocol's typed failure for a Boat API error, and let the backend's own failures through.

    With `sandbox`, a failure that would propagate unchanged is followed by one lookup, so a sandbox
    that was deleted or stopped meanwhile ends the run instead of being retried.
    """
    try:
        yield
    except (WorkspaceError, OSError):
        raise
    except Exception as error:
        translated = _translate(error, context, sandbox_id=None if sandbox is None else sandbox.id, path=path)
        if translated is not error:
            raise translated from error
        if sandbox is not None and (gone := await _probe(sandbox)) is not None:
            raise gone from error
        raise


async def _close_quietly(client: httpx.AsyncClient) -> None:
    """Close an API client, bounded and shielded; a failure is logged, not raised."""
    with anyio.move_on_after(_TEARDOWN_TIMEOUT, shield=True) as deadline:
        try:
            await client.aclose()
        except Exception as error:
            logger.warning('Could not close the Boat API client: %r', error)
    if deadline.cancelled_caught:
        logger.warning('Closing the Boat API client did not complete within %ss.', _TEARDOWN_TIMEOUT)


def _marker() -> str:
    return f'pydantic-ai-{uuid.uuid4().hex}'


def _command_line(argv: Sequence[str], *, marker: str, cwd: str | None, env: Mapping[str, str]) -> str:
    """The Boat command string that runs `argv` in `cwd` with stdin at EOF.

    Boat runs the string with `bash -lc`. `exec` keeps that login shell from running
    `~/.bash_logout`, which prints to the output. The `sh` it becomes stays the program's parent,
    with `marker` in its script, so the process list finds it; it exits 125 after printing
    `<marker>-cwd` when `cwd` is missing.
    """
    if env:
        # Inside the `sh`, since a `sh` such as dash drops names that are not shell identifiers.
        # `--` ends `env`'s options, so a name starting with `-` is not read as one.
        argv = ['env', '--', *(f'{name}={value}' for name, value in env.items()), *argv]
    guard = f'cd -- {shlex.quote(cwd)} || {{ printf %s {marker}-cwd; exit 125; }}' if cwd is not None else f': {marker}'
    # `sh` as `$0`, which the shell's own messages, such as a missing program's, start with.
    return shlex.join(['exec', 'sh', '-c', f'{guard}; "$@" </dev/null', 'sh', *argv])


async def _exec(sandbox: _Sandbox, argv: Sequence[str], *, cwd: str | None) -> CommandResult:
    """Run a short internal command synchronously: one round trip, bounded by Boat."""
    marker = _marker()
    async with _mapped_errors('Could not run a command in the sandbox', sandbox=sandbox):
        finished = await _api.call(
            sandbox.client,
            'POST',
            sandbox.path('/commands'),
            json={'command': _command_line(argv, marker=marker, cwd=cwd, env={}), 'timeoutSeconds': _HELPER_TIMEOUT},
            model=_api.CommandFinished,
            # Boat answers within 10 seconds of its own timeout, even when a child keeps the output open.
            timeout=_HELPER_TIMEOUT + 15,
            retry='not_run',
        )
    if finished.timed_out:
        raise WorkspaceError(
            f'An internal command did not finish within {_HELPER_TIMEOUT}s in Boat sandbox {sandbox.id}.'
        )
    if finished.stdout_truncated or finished.stderr_truncated:
        raise WorkspaceOutputLimitError(_OUTPUT_LIMIT_MESSAGE, limit=_MAX_OUTPUT_BYTES)
    if finished.exit_code == 125 and finished.stdout == f'{marker}-cwd':
        raise _missing_cwd(cwd, sandbox.id)
    return CommandResult(exit_code=finished.exit_code, stdout=finished.stdout, stderr=finished.stderr)


class _SyncShell:
    """Synchronous Boat commands as the runner core's shell filesystem drives."""

    def __init__(self, sandbox: _Sandbox, working_dir: Callable[[], Awaitable[str]]) -> None:
        self._sandbox = sandbox
        self._working_dir = working_dir

    async def working_dir(self) -> str:
        return await self._working_dir()

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        # The shell filesystem passes absolute paths and no `env` or `timeout`.
        return await _exec(self._sandbox, command_argv(command, shell), cwd=None)


class _BoatProcess:
    """A detached Boat command, polled until its own process exits."""

    def __init__(self, sandbox: _Sandbox) -> None:
        self._sandbox = sandbox
        # In the command's script, so a command whose start Boat never confirmed can still be found and stopped.
        self.marker = _marker()
        self._pid: int | None = None
        # Nothing is left to stop: the command exited, or Boat refused to start it.
        self._finished = False
        self.missing_cwd = False
        # The partial output a timeout reports: the tail of each stream at the latest check.
        self.stdout = ''
        self.stderr = ''

    async def start(self, line: str) -> None:
        async def submit() -> None:
            started = await _api.call(
                self._sandbox.client,
                'POST',
                self._sandbox.path('/commands'),
                json={'command': line, 'detached': True},
                model=_api.CommandStarted,
                retry='not_run',
            )
            self._pid = started.process_id

        try:
            async with _mapped_errors('Could not start the command', sandbox=self._sandbox):
                # Finished even if the caller is cancelled, so the command can be stopped by its ID.
                await run_to_completion(submit)
        except WorkspaceError:
            # A refusal, or a sandbox found stopped or deleted: there is no command to look for.
            self._finished = True
            raise
        except (httpx.HTTPError, _api.BoatAPIError) as error:
            if isinstance(error, _api.BoatAPIError) and error.status == 429:
                self._finished = True
                raise
            # Not transient: a retry could run the command a second time.
            raise WorkspaceError(
                f'Boat did not confirm that the command started, so it may be running: {error}'
            ) from error

    async def wait(self) -> CommandResult:
        while True:
            status = await self._status(tailBytes=_PARTIAL_OUTPUT_LIMIT)
            self.stdout, self.stderr = status.stdout, status.stderr
            if not status.running or not status.known:
                break
            await anyio.sleep(_POLL_INTERVAL)
        if not status.known or (exit_code := status.exit_code) is None:
            # Boat forgets a command's exit when the sandbox's agent restarts.
            raise await _probe(self._sandbox) or WorkspaceError(
                'Boat lost track of the command; its result is unknown.'
            )
        self._finished = True
        if status.stdout_truncated or status.stderr_truncated:
            status = await self._status(tailBytes=_MAX_OUTPUT_BYTES)
            if status.stdout_truncated or status.stderr_truncated:
                head = await self._status(stdoutOffset=0, stderrOffset=0)
                raise WorkspaceOutputLimitError(
                    _OUTPUT_LIMIT_MESSAGE,
                    limit=_MAX_OUTPUT_BYTES,
                    stdout=head.stdout[:_OUTPUT_PREVIEW_BYTES],
                    stderr=head.stderr[:_OUTPUT_PREVIEW_BYTES],
                )
        if exit_code == 125 and status.stdout == f'{self.marker}-cwd':
            self.missing_cwd = True
            return CommandResult(exit_code=exit_code, stdout='', stderr='')
        return CommandResult(exit_code=exit_code, stdout=status.stdout, stderr=status.stderr)

    async def _status(self, **params: int) -> _api.CommandStatus:
        try:
            async with _mapped_errors('Could not read the command result', sandbox=self._sandbox):
                return await _api.call(
                    self._sandbox.client,
                    'GET',
                    self._sandbox.path(f'/commands/{self._pid}'),
                    params=params,
                    model=_api.CommandStatus,
                    retry='idempotent',
                )
        except (httpx.HTTPError, _api.BoatAPIError) as error:
            raise WorkspaceError(
                f'Could not read the result of a command that may still be running: {error}'
            ) from error

    async def kill(self) -> None:
        """Stop the command and every process it started, unless it already exited."""
        if self._finished:
            return
        try:
            for pid in [self._pid] if self._pid is not None else await self._unconfirmed():
                try:
                    await _api.call(
                        self._sandbox.client,
                        'DELETE',
                        self._sandbox.path(f'/commands/{pid}'),
                        model=_api.Empty,
                        timeout=_KILL_TIMEOUT,
                    )
                except _api.BoatAPIError as error:
                    # 400: Boat no longer knows the command; 404: its sandbox is gone.
                    if error.status not in (400, 404):
                        raise
        except Exception:
            logger.warning('Could not stop a command in Boat sandbox %s; it may still be running.', self._sandbox.id)
            raise

    async def _unconfirmed(self) -> list[int]:
        """The ID of this command if Boat started it without confirming, found by its marker."""
        # The bracket keeps the pattern from matching the command line of this `pgrep` itself.
        result = await _exec(self._sandbox, ['pgrep', '-f', '--', f'[{self.marker[0]}]{self.marker[1:]}'], cwd=None)
        return [int(pid) for pid in result.stdout.split()]


class BoatSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem, SupportsRealpath):
    """A [Boat](https://boat.dev) sandbox as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands and file operations run inside a Boat sandbox, a cloud VM, so the host is never exposed.

    Building one does no I/O. The first operation creates or attaches to a sandbox. A new sandbox
    gets none of the Boat account's own credentials. The backend does not stop or delete the
    sandbox; that is the application's job.

    A command's complete output is returned once it exits. A shell string runs under `/bin/sh -c`;
    a timeout or cancellation stops the command and every process it started.

    A deleted sandbox, one Boat stopped (attaching again resumes it), rejected credentials, and an
    unknown snapshot raise `WorkspaceUnavailableError`. Path failures raise the built-in errors,
    such as `FileNotFoundError`.

    Args:
        client: An `httpx.AsyncClient` for Boat's API, with its `base_url` and an
            `Authorization: Bearer` header set, to create or attach with. The caller owns closing
            it. Without one, the backend opens its own from `BOAT_API_KEY` on first use and closes
            it in `aclose()`.
        ref: Identity of an existing sandbox to attach to on first use.
        machine_type: Boat machine size for a newly created sandbox, such as `'small'` or
            `'large'`; Boat's default when `None`.
        snapshot: Named Boat snapshot a newly created sandbox starts from; Boat's base image when `None`.
        auto_stop: Seconds after it starts, and after each resume, before Boat stops a newly created
            sandbox (its disk is kept); `0` disables it, and `None` keeps Boat's default of an hour.
            Command activity does not postpone it.
        working_dir: Absolute directory commands start in and relative paths resolve against; the
            sandbox's own default when `None`, discovered with `pwd -P` on first use.
        env: Environment variables every command gets; a command's own `env` is layered on top.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        ref: WorkspaceRef | None = None,
        machine_type: str | None = None,
        snapshot: str | None = None,
        auto_stop: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None and ref.provider != 'boat':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'boat'")
        if ref is not None and not ref.id.strip():
            raise ValueError('Boat workspace ref id cannot be empty')
        check_env(env)
        self._ref = ref
        self._client = client
        self._owns_client = client is None
        self._machine_type = machine_type
        self._snapshot = snapshot
        self._auto_stop = auto_stop
        self._env = dict(env or {})
        self._working_dir = absolute_path('working_dir', working_dir)
        # Sent with every creation attempt, so a retry after a lost reply gets the sandbox it created.
        self._idempotency_key = _marker()
        self._sandbox: _Sandbox | None = None
        # `working_dir()` must return a canonical absolute path, resolved once with `pwd -P`.
        self._resolved_working_dir: str | None = None
        # A sandbox this backend created still needs its `working_dir` made.
        self._working_dir_pending = False
        self._lock = anyio.Lock()
        # Set while nothing will call `aclose()`: the owned client then closes when the last
        # operation in flight ends.
        self._close_after_operation = False
        self._operations = 0

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the sandbox, or `None` before one has been created.

        The `id` is Boat's sandbox ID, such as `bx_h8mygsw2`.
        """
        return self._ref

    async def _connect(self) -> _Sandbox:
        """Create or attach to the sandbox on first use, and return it once it is ready.

        Attaching by `ref` to a sandbox that no longer exists raises `WorkspaceUnavailableError`; it
        does not create a replacement.
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

    async def _acquire_locked(self) -> _Sandbox:
        """Create or attach to the sandbox; the caller holds `_lock`."""
        if self._client is None:
            self._client = _new_client()
        client = self._client

        async def create() -> str:
            sandbox_id = await self._create(client)
            self._ref = WorkspaceRef(provider='boat', id=sandbox_id)
            self._working_dir_pending = self._working_dir is not None
            # Logged at creation, so the id can be found whatever ends the run before it is stored.
            logger.info('Created Boat sandbox %s', sandbox_id)
            return sandbox_id

        # Creation finishes even if the caller is cancelled, so a billed sandbox is never left unrecorded.
        sandbox_id = self._ref.id if self._ref is not None else await run_to_completion(create)
        sandbox = _Sandbox(client, sandbox_id)
        await self._wait_until_ready(sandbox)
        self._sandbox = sandbox
        return sandbox

    async def _create(self, client: httpx.AsyncClient) -> str:
        """Create a sandbox under an idempotency key kept across retries, so a lost reply is found rather than duplicated."""
        # Without `noEnv`, Boat gives every command the account owner's GitHub token and other
        # logins, which code an agent runs could read.
        body: dict[str, object] = {'noEnv': True}
        if self._machine_type is not None:
            body['type'] = self._machine_type
        if self._snapshot is not None:
            body['from'] = self._snapshot
        if self._auto_stop is not None:
            # Boat reads `0` as its default; `null` turns the auto-stop off.
            body['ttlSeconds'] = self._auto_stop or None
        with anyio.move_on_after(_CREATE_TIMEOUT):
            while True:
                try:
                    created = await _api.call(
                        client,
                        'POST',
                        '/sandboxes',
                        json=body,
                        headers={'Idempotency-Key': self._idempotency_key},
                        model=_api.SandboxResponse,
                    )
                    return created.sandbox.id
                except httpx.TransportError:
                    pass  # The sandbox may have been created before the reply was lost; the key finds it.
                except _api.BoatAPIError as error:
                    if error.status != 409 or error.code != 'idempotency_in_progress':
                        translated = _translate(error, 'Could not start Boat sandbox')
                        if translated is error:
                            raise
                        if type(translated) is WorkspaceError:
                            # Boat refused the request (an unknown snapshot, an invalid setting): no retry can fix it.
                            translated = WorkspaceUnavailableError(str(translated))
                        raise translated from error
                await anyio.sleep(_RETRY_DELAY)
        # A stalled control plane is transient, not a command deadline.
        raise TimeoutError(
            f'Boat sandbox creation did not complete within {_CREATE_TIMEOUT}s; the Boat control plane may be unreachable.'
        )

    async def _wait_until_ready(self, sandbox: _Sandbox) -> None:
        """Wait until Boat reports the sandbox ready, resuming it if Boat stopped it.

        Commands sent to a starting sandbox wait for it, but file calls do not: right after a
        resume, Boat's file API can report an existing file as missing.
        """
        with anyio.move_on_after(_CREATE_TIMEOUT):
            async with _mapped_errors(f'Could not attach to Boat sandbox {sandbox.id!r}'):
                while True:
                    try:
                        info = await _api.call(sandbox.client, 'GET', sandbox.path(), model=_api.SandboxResponse)
                        state = info.sandbox.state
                        if state in _STOPPED_STATES:
                            await _api.call(sandbox.client, 'POST', sandbox.path('/resume'), model=_api.Empty)
                            continue
                    except _api.BoatAPIError as error:
                        if error.status == 404:
                            raise WorkspaceUnavailableError(_not_found_message(sandbox.id)) from error
                        if error.status == 409:
                            # Boat cannot resume it, for example once its snapshot has expired.
                            raise WorkspaceUnavailableError(
                                f'Could not resume Boat sandbox {sandbox.id!r}: {error.message}'
                            ) from error
                        raise
                    if state in _READY_STATES:
                        return
                    if (failed := _state_error(sandbox.id, state)) is not None:
                        raise failed
                    await anyio.sleep(_READY_POLL)
        raise TimeoutError(f'Connecting to Boat sandbox {sandbox.id!r} did not complete within {_CREATE_TIMEOUT}s.')

    async def _make_working_dir(self, sandbox: _Sandbox) -> None:
        directory = self._working_dir
        assert directory is not None
        with anyio.CancelScope(shield=True):
            result = await _exec(sandbox, ['mkdir', '-p', '--', directory], cwd=None)
            if result.exit_code != 0:
                raise WorkspaceError(
                    f'Could not create working_dir {directory!r} in Boat sandbox {sandbox.id}: {result.stderr.strip()}'
                )
            self._working_dir_pending = False

    async def aclose(self) -> None:
        """Close the API client this backend opened for itself.

        A client passed as `client=` is left alone, and the sandbox keeps running. A later
        operation opens a new client and attaches by `ref`. `BoatSandbox` calls this when the run
        that used the backend ends.
        """
        if self._client is None or not self._owns_client or not running_on_asyncio():
            return
        if self._operations:
            # An operation in flight, such as a cancelled command being stopped, still needs the
            # client: the last one to finish closes it.
            self._close_after_operation = True
            return

        async def close() -> None:
            # Under the lock, so a sandbox still being created is recorded before its client closes.
            async with self._lock:
                client, self._client, self._sandbox, self._resolved_working_dir = self._client, None, None, None
            if client is not None:
                await _close_quietly(client)

        await run_to_completion(close)

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[None]:
        self._operations += 1
        try:
            yield
        finally:
            self._operations -= 1
            if self._operations == 0 and self._close_after_operation and self._owns_client and self._client is not None:
                # Detached before any await, so an operation starting meanwhile opens its own client.
                client, self._client, self._sandbox, self._resolved_working_dir = self._client, None, None, None
                await run_to_completion(lambda: _close_quietly(client))

    def _shell(self, sandbox: _Sandbox) -> _ShellFilesystem:
        return _ShellFilesystem(_SyncShell(sandbox, self.working_dir))

    async def working_dir(self) -> str:
        """Return the canonical directory commands start in, probed once per connection."""
        async with self._operation():
            if self._resolved_working_dir is None:
                sandbox = await self._connect()
                result = await _exec(sandbox, ['pwd', '-P'], cwd=self._working_dir)
                printed = result.stdout.removesuffix('\n')
                if result.exit_code != 0 or not posixpath.isabs(printed):
                    raise WorkspaceError(
                        f'Could not determine the working directory of Boat sandbox {sandbox.id}: {result.stderr}'
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
            check_env(env)
            # Acquiring the sandbox has its own bound; the timeout is the command's alone.
            sandbox = await self._connect()
            process = _BoatProcess(sandbox)
            line = _command_line(argv, marker=process.marker, cwd=self._working_dir, env={**self._env, **(env or {})})
            async with command_deadline(timeout, stop=process.kill, output=lambda: (process.stdout, process.stderr)):
                await process.start(line)
                result = await process.wait()
                if process.missing_cwd:
                    raise _missing_cwd(self._working_dir, sandbox.id)
            return result

    async def realpath(self, path: str) -> str:
        # Native, because the shell fallback would run several detached commands per call.
        async with self._operation():
            _check_path(path)
            sandbox = await self._connect()
            return await self._shell(sandbox).realpath(posixpath.join(await self.working_dir(), path))

    async def read_bytes(self, path: str) -> bytes:
        async with self._operation():
            _check_path(path)
            sandbox = await self._connect()
            async with _mapped_errors(f'Could not access {path!r} in the sandbox', sandbox=sandbox, path=path):
                try:
                    file = await _api.call(
                        sandbox.client,
                        'GET',
                        sandbox.path('/files'),
                        params={'path': path, 'encoding': 'base64'},
                        model=_api.FileRead,
                        retry='idempotent',
                    )
                except _api.BoatAPIError as error:
                    refusal = _file_refusal(error)
                    if refusal == 'outside':
                        return await self._shell(sandbox).read_bytes(path)
                    if refusal == 'not_a_file':
                        # A directory, or a special file such as a FIFO.
                        if (await self._shell(sandbox).stat(path)).is_dir:
                            raise IsADirectoryError(errno.EISDIR, 'Is a directory in the Boat sandbox', path) from error
                        raise OSError(f'Not a regular file in the Boat sandbox: {path!r}') from error
                    if refusal == 'too_large':
                        return await _download(sandbox, path)
                    raise
            return base64.b64decode(file.content)

    async def write_bytes(self, path: str, data: bytes) -> None:
        async with self._operation():
            _check_path(path)
            sandbox = await self._connect()
            # Boat creates missing parents, follows a symlink, and keeps an existing file's mode.
            async with _mapped_errors(f'Could not access {path!r} in the sandbox', sandbox=sandbox, path=path):
                try:
                    if len(data) <= _MAX_INLINE_FILE_BYTES:
                        await _api.call(
                            sandbox.client,
                            'PUT',
                            sandbox.path('/files'),
                            json={'path': path, 'content': base64.b64encode(data).decode(), 'encoding': 'base64'},
                            model=_api.Empty,
                            retry='idempotent',
                        )
                    else:
                        await _upload(sandbox, path, data)
                except _api.BoatAPIError as error:
                    if _file_refusal(error) != 'outside':
                        raise
                    await self._shell(sandbox).write_bytes(path, data)

    # The other file operations are the ones Pydantic AI derives from commands, run synchronously.
    async def stat(self, path: str) -> FileEntry:
        async with self._operation():
            _check_path(path)
            return await self._shell(await self._connect()).stat(path)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        async with self._operation():
            _check_path(path)
            return await self._shell(await self._connect()).list_dir(path)

    async def make_dir(self, path: str) -> None:
        async with self._operation():
            _check_path(path)
            await self._shell(await self._connect()).make_dir(path)

    async def remove(self, path: str) -> None:
        async with self._operation():
            _check_path(path)
            await self._shell(await self._connect()).remove(path)

    async def exists(self, path: str) -> bool:
        async with self._operation():
            _check_path(path)
            return await self._shell(await self._connect()).exists(path)


def _check_path(path: str) -> None:
    # Refused as the Daytona backend refuses them, so a path behaves the same in both cloud sandboxes.
    if '\n' in path or '\r' in path:
        raise ValueError('Boat file paths cannot contain a newline or carriage return')


def _file_refusal(error: _api.BoatAPIError) -> Literal['outside', 'not_a_file', 'too_large'] | None:
    """Why Boat's file API refused a path, when the refusal is not a POSIX error."""
    if error.status != 400:
        return None
    # Boat's file API reaches only paths that resolve under /home/user or /tmp.
    if error.code == 'invalid_path' or 'Path must resolve under' in error.message:
        return 'outside'
    if error.message == 'Path is not a file.':
        return 'not_a_file'
    if error.message.startswith('File is too large'):
        return 'too_large'
    return None


async def _download(sandbox: _Sandbox, path: str) -> bytes:
    url = await _api.call(
        sandbox.client,
        'POST',
        sandbox.path('/files/download-url'),
        json={'path': path},
        model=_api.TransferURL,
        retry='idempotent',
    )
    try:
        return await _api.transfer(sandbox.client, 'GET', url.url)
    except _api.BoatAPIError as error:
        # Boat issues the URL without checking the path, so the machine answers for a missing file.
        if error.status == 404:
            raise FileNotFoundError(errno.ENOENT, 'No such file or directory in the Boat sandbox', path) from error
        raise


async def _upload(sandbox: _Sandbox, path: str, data: bytes) -> None:
    url = await _api.call(
        sandbox.client,
        'POST',
        sandbox.path('/files/upload-url'),
        json={'path': path},
        model=_api.TransferURL,
        retry='idempotent',
    )
    await _api.transfer(sandbox.client, 'PUT', url.url, content=data)
