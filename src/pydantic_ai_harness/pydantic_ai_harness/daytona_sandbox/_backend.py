"""A Daytona sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

Daytona's session logs can deliver a block of stdout as stderr or the reverse, so each command writes
both streams to files that are downloaded once it exits.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import logging
import posixpath
import shlex
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

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
_OUTPUT_LIMIT_MESSAGE = (
    'Daytona command output exceeded the 10 MiB limit; redirect it to a file and read part of it instead'
)
_REQUEST_TIMEOUT = 30
# Bound sandbox creation and attachment so a wedged control plane cannot hang acquisition.
_CREATE_TIMEOUT = 120
# For create, start, and delete.
_LIFECYCLE_TIMEOUT = 60.0
_TEARDOWN_TIMEOUT = 30.0
# Between attempts to delete a command session.
_RETRY_DELAY = 1.0
# The most partial output a timeout reports per stream.
_PARTIAL_OUTPUT_LIMIT = 4096
# A command's whole output is downloaded into memory.
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_OUTPUT_PREVIEW_BYTES = 64 * 1024

logger = logging.getLogger(__name__)


def _require_asyncio() -> None:
    if not running_on_asyncio():
        raise UserError('Daytona needs the asyncio event loop: the Daytona SDK runs its calls on asyncio tasks.')


def _new_client() -> AsyncDaytona:
    """An `AsyncDaytona` configured from the environment: `DAYTONA_API_KEY`, or a JWT token and organization ID."""
    try:
        return daytona.AsyncDaytona()
    except daytona.DaytonaAuthenticationError as error:
        # The SDK reads the credentials here, so this is a missing key rather than a rejected one.
        raise WorkspaceUnavailableError(_MISSING_CREDENTIALS_MESSAGE) from error


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
            await _close_quietly(owned)


def _in_deleted_state(sandbox: AsyncSandbox) -> bool:
    return sandbox.state in (daytona.SandboxState.DESTROYING, daytona.SandboxState.DESTROYED)


def _unavailable_message(sandbox_id: str) -> str:
    return (
        f'The Daytona sandbox {sandbox_id!r} no longer exists (it was deleted). '
        "Pass `workspace='new'` to start a fresh sandbox."
    )


def _translate(error: Exception, context: str, *, path: str | None = None) -> Exception:
    """Map a Daytona SDK error onto the failures the workspace protocol promises.

    Returns `error` itself for what must propagate unchanged: connection failures, SDK timeouts,
    rate limits, server errors, and anything that is not a `DaytonaError`, which durable engines
    retry as transient. A not-found answer is a missing `path` for a path operation, and a refused
    request otherwise.
    """
    if path is not None and isinstance(error, daytona.DaytonaError):
        # Path failures carry the POSIX `strerror` text, as a 400 or, for a read, a 500.
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


async def _probe(sandbox: AsyncSandbox) -> WorkspaceUnavailableError | None:
    """`WorkspaceUnavailableError` if Daytona deleted or stopped `sandbox`; an inconclusive lookup counts as running."""
    try:
        with anyio.fail_after(_REQUEST_TIMEOUT):
            await sandbox.refresh_data(request_timeout=_REQUEST_TIMEOUT)
    except daytona.DaytonaNotFoundError:
        return WorkspaceUnavailableError(_unavailable_message(sandbox.id))
    except Exception:
        return None
    if _in_deleted_state(sandbox):
        return WorkspaceUnavailableError(_unavailable_message(sandbox.id))
    if sandbox.state in (daytona.SandboxState.STOPPED, daytona.SandboxState.ARCHIVED, daytona.SandboxState.PAUSED):
        return WorkspaceUnavailableError(
            f'The Daytona sandbox {sandbox.id!r} was stopped while idle; a later run that attaches to it '
            "starts it again. Pass `workspace='new'` to start a fresh sandbox."
        )
    return None


@asynccontextmanager
async def _mapped_errors(
    context: str, *, sandbox: AsyncSandbox | None = None, path: str | None = None
) -> AsyncGenerator[None]:
    """Raise the protocol's typed failure for a Daytona SDK error, and let the backend's own failures through.

    The toolbox answers a missing path and a deleted sandbox with the same not-found error, so with
    `sandbox`, a failure that does not already end the run is followed by one control-plane lookup.
    """
    try:
        yield
    except (WorkspaceError, IsADirectoryError):
        raise
    except Exception as error:
        translated = _translate(error, context, path=path)
        if sandbox is not None and not isinstance(translated, WorkspaceUnavailableError):
            if (gone := await _probe(sandbox)) is not None:
                raise gone from error
        if translated is error:
            raise
        raise translated from error


async def _close_client(client: AsyncDaytona) -> None:
    await client.close()
    # `close()` cancels the SDK's event connection task without waiting for it. On a loop that is not
    # drained afterwards, such as `Agent.run_sync`'s, the task then never closes its connection.
    connecting = getattr(getattr(client, '_event_dispatcher', None), '_connect_task', None)
    if connecting is not None:
        await asyncio.wait([connecting])


async def _close_quietly(client: AsyncDaytona) -> None:
    """Close an API client, bounded and shielded; a failure is logged, not raised."""
    with anyio.move_on_after(_TEARDOWN_TIMEOUT, shield=True) as deadline:
        try:
            await _close_client(client)
        except Exception as error:
            logger.warning('Could not close the Daytona API client: %r', error)
    if deadline.cancelled_caught:
        logger.warning('Closing the Daytona API client did not complete within %ss.', _TEARDOWN_TIMEOUT)


async def _delete_session(sandbox: AsyncSandbox, session_id: str) -> None:
    """Delete a command session, which kills its command, retrying until the caller's `stop_shielded` bound.

    A failure is logged, not raised, so it cannot replace the caller's outcome.
    """
    deleted = False
    try:
        while True:
            try:
                await sandbox.process.delete_session(session_id, request_timeout=_REQUEST_TIMEOUT)
                deleted = True
                return
            except daytona.DaytonaNotFoundError:
                deleted = True
                return
            except Exception:
                await anyio.sleep(_RETRY_DELAY)
    finally:
        if not deleted:
            logger.warning('Could not delete Daytona command session %s; its command may still be running.', session_id)


class _DaytonaProcess:
    """A command in its own Daytona process session, writing its stdout and stderr to files in `/tmp`."""

    def __init__(self, sandbox: AsyncSandbox) -> None:
        self._sandbox = sandbox
        self._process = sandbox.process
        self._session_id = f'pydantic-ai-{uuid.uuid4().hex}'
        self._marker = f'pydantic-ai-end-{uuid.uuid4().hex}'
        self._out, self._err = f'/tmp/{self._marker}.out', f'/tmp/{self._marker}.err'
        self._command_id: str | None = None
        self._finished = False
        self.missing_cwd = False
        # The partial output a timeout reports, read from the output files by `kill()`.
        self.stdout = ''
        self.stderr = ''

    def command_line(self, argv: Sequence[str], *, cwd: str | None, env: Mapping[str, str]) -> str:
        """The session command: `argv` with its stdout and stderr sent to this process's files.

        The `sh` exits with the argv's status, or 125 after writing `<marker>-cwd` when `cwd` is missing.
        """
        if env:
            # Inside the `sh`, since a `sh` such as dash drops names that are not shell identifiers.
            # `--` ends `env`'s options, so a name starting with `-` is not read as one.
            argv = ['env', '--', *(f'{name}={value}' for name, value in env.items()), *argv]
        missing_cwd = f'{{ printf %s {self._marker}-cwd >{self._out}; exit 125; }}'
        prefix = f'cd -- {shlex.quote(cwd)} || {missing_cwd}; ' if cwd is not None else ''
        script = f'{prefix}"$@" </dev/null >{self._out} 2>{self._err}'
        return shlex.join(['sh', '-c', script, 'sh', *argv])

    async def start(self, line: str) -> None:
        """Run `line` in a new session; a start that fails or is cancelled deletes the session."""
        try:
            with anyio.fail_after(_REQUEST_TIMEOUT):
                async with _mapped_errors('Could not start the command', sandbox=self._sandbox):
                    await self._process.create_session(self._session_id, request_timeout=_REQUEST_TIMEOUT)
                    response = await self._process.execute_session_command(
                        self._session_id,
                        daytona.SessionExecuteRequest(command=line, run_async=True),
                        timeout=_REQUEST_TIMEOUT,
                    )
        except BaseException as error:
            # Daytona may have accepted the command without acknowledging it; the session ID is the cleanup token.
            await stop_shielded(lambda: _delete_session(self._sandbox, self._session_id))
            if isinstance(error, TimeoutError):
                raise TimeoutError(
                    f'Daytona command session setup did not complete within {_REQUEST_TIMEOUT}s.'
                ) from error
            raise
        self._command_id = response.cmd_id

    async def wait(self) -> CommandResult:
        assert self._command_id is not None
        while True:
            async with _mapped_errors('Could not read the command result', sandbox=self._sandbox):
                command = await self._process.get_session_command(
                    self._session_id, self._command_id, request_timeout=_REQUEST_TIMEOUT
                )
            if command.exit_code is not None:
                break
            await anyio.sleep(0.2)
        self._finished = True
        return await self._result(command.exit_code)

    async def _result(self, exit_code: int) -> CommandResult:
        try:
            async with _mapped_errors('Could not read the command output', sandbox=self._sandbox):
                with anyio.fail_after(_REQUEST_TIMEOUT):
                    # Sizes first: a command can grow a file without writing it (`truncate`).
                    sizes = await self._process.exec(
                        f'for f in {self._out} {self._err}; do if [ -e "$f" ]; then wc -c < "$f"; else echo 0; fi; done',
                        timeout=_REQUEST_TIMEOUT,
                    )
                    try:
                        out_size, err_size = (int(size) for size in sizes.result.split())
                    except ValueError:
                        raise WorkspaceError(f'Could not read the command output sizes: {sizes.result!r}') from None
                    if out_size + err_size > _MAX_OUTPUT_BYTES:
                        stdout, stderr = await self._excerpts('head', _OUTPUT_PREVIEW_BYTES)
                        raise WorkspaceOutputLimitError(
                            _OUTPUT_LIMIT_MESSAGE, limit=_MAX_OUTPUT_BYTES, stdout=stdout, stderr=stderr
                        )
                    out = await self._sandbox.fs.download_file(self._out, _REQUEST_TIMEOUT) if out_size else b''
                    err = await self._sandbox.fs.download_file(self._err, _REQUEST_TIMEOUT) if err_size else b''
                    # A background child can keep writing after the sizes were read.
                    if len(out) + len(err) > _MAX_OUTPUT_BYTES:
                        raise WorkspaceOutputLimitError(
                            _OUTPUT_LIMIT_MESSAGE,
                            limit=_MAX_OUTPUT_BYTES,
                            stdout=out[:_OUTPUT_PREVIEW_BYTES].decode(errors='replace'),
                            stderr=err[:_OUTPUT_PREVIEW_BYTES].decode(errors='replace'),
                        )
        finally:
            await self._remove_output_files()
        stdout = out.decode(errors='replace')
        if f'{self._marker}-cwd' in stdout:
            self.missing_cwd = True
            return CommandResult(exit_code=exit_code, stdout='', stderr='')
        return CommandResult(exit_code=exit_code, stdout=stdout, stderr=err.decode(errors='replace'))

    async def _excerpts(self, end: Literal['head', 'tail'], limit: int) -> tuple[str, str]:
        """At most `limit` bytes from one end of each output file, cut in the sandbox before transfer."""
        paths = (self._out, self._err)
        cuts = '; '.join(f'{end} -c {limit} -- {path} > {path}.part 2>/dev/null || : > {path}.part' for path in paths)
        await self._process.exec(cuts, timeout=_REQUEST_TIMEOUT)
        out, err = [await self._sandbox.fs.download_file(f'{path}.part', _REQUEST_TIMEOUT) for path in paths]
        return out.decode(errors='replace'), err.decode(errors='replace')

    async def _remove_output_files(self) -> None:
        """Best effort: a file left behind only takes space in the sandbox's `/tmp`."""
        paths = f'{self._out} {self._err} {self._out}.part {self._err}.part'
        with anyio.move_on_after(_REQUEST_TIMEOUT, shield=True):
            try:
                await self._process.exec(f'rm -f -- {paths}', timeout=_REQUEST_TIMEOUT)
            except Exception:
                pass

    async def kill(self) -> None:
        """Delete the session, which kills its command, and keep the tail of its output for a timeout."""
        if self._command_id is None:
            return  # Never started: a failed start deletes its own session.
        await _delete_session(self._sandbox, self._session_id)
        if self._finished:
            return  # The result already read and removed the output files.
        with anyio.move_on_after(_REQUEST_TIMEOUT):
            try:
                self.stdout, self.stderr = await self._excerpts('tail', _PARTIAL_OUTPUT_LIMIT)
            except Exception:
                self.stdout, self.stderr = '', ''
        await self._remove_output_files()


class DaytonaSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem, SupportsRealpath):
    """A [Daytona](https://www.daytona.io) sandbox as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands and file operations run inside a Daytona sandbox, so the host is never exposed.

    Building one does no I/O. The first operation creates or attaches to a sandbox, and the typed
    `daytona.AsyncSandbox` is available through `get_sandbox()`. The backend does not stop or
    delete the sandbox; that is the application's job.

    A command's complete output is returned once it exits. A shell string runs under `/bin/sh -c`;
    a timeout or cancellation stops the command.

    A deleted sandbox, one Daytona stopped while idle (attaching again starts it), rejected
    credentials, and an unknown snapshot raise `WorkspaceUnavailableError`. Path failures raise the
    built-in errors, such as `FileNotFoundError`.

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

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the sandbox, or `None` before one has been created.

        The `id` is always Daytona's sandbox ID, also after attaching by a `ref` that names the sandbox.
        """
        return self._ref

    async def get_sandbox(self) -> AsyncSandbox:
        """Return the typed `daytona.AsyncSandbox`, creating or attaching to it on first use.

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

    async def _acquire_locked(self) -> AsyncSandbox:
        """Create or attach to the sandbox; the caller holds `_lock`."""
        if self._client is None:
            self._client = _new_client()
        client = self._client
        if self._ref is not None:
            sandbox = await self._attach(client, self._ref.id)
            self._record(sandbox)
            return sandbox

        async def create() -> AsyncSandbox:
            created = await self._create(client)
            self._record(created)
            self._working_dir_pending = self._working_dir is not None
            # Logged at creation, so the id can be found whatever ends the run before it is stored.
            logger.info('Created Daytona sandbox %s', created.id)
            return created

        # Creation finishes even if the caller is cancelled, so a billed sandbox is never left unrecorded.
        return await run_to_completion(create)

    def _record(self, sandbox: AsyncSandbox) -> None:
        self._sandbox = sandbox
        self._ref = WorkspaceRef(provider='daytona', id=sandbox.id)

    async def _create(self, client: AsyncDaytona) -> AsyncSandbox:
        """Create a sandbox under a name kept across retries, so a lost reply is found rather than duplicated."""
        if self._create_name is None:
            self._create_name = f'pydantic-ai-{uuid.uuid4().hex}'
        name = self._create_name
        if (existing := await self._find_created(client, name)) is not None:
            return existing
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
                with contextlib.suppress(daytona.DaytonaConnectionError, daytona.DaytonaTimeoutError):
                    if (created := await self._find_created(client, name)) is not None:
                        return created
                raise
            except Exception as error:
                translated = _translate(error, 'Could not start Daytona sandbox')
                if translated is error:
                    raise
                if type(translated) is WorkspaceError:
                    # Daytona refused the request (an unknown snapshot, an invalid setting): no retry can fix it.
                    translated = WorkspaceUnavailableError(str(translated))
                raise translated from error
        with contextlib.suppress(daytona.DaytonaConnectionError, daytona.DaytonaTimeoutError):
            if (created := await self._find_created(client, name)) is not None:
                return created
        # A stalled control plane is transient, not a command deadline.
        raise TimeoutError(
            f'Daytona sandbox creation did not complete within {_CREATE_TIMEOUT}s; '
            'the Daytona control plane may be unreachable.'
        )

    async def _find_created(self, client: AsyncDaytona, name: str) -> AsyncSandbox | None:
        """The sandbox an earlier create under `name` made, once it has started; `None` if Daytona has none."""
        async with _mapped_errors('Could not start Daytona sandbox'):
            try:
                sandbox = await client.get(name, request_timeout=_REQUEST_TIMEOUT)
            except daytona.DaytonaNotFoundError:
                return None
            if sandbox.state != daytona.SandboxState.STARTED:
                # The SDK's `create` stops waiting for a slow start once the sandbox exists.
                await sandbox.wait_for_sandbox_start(timeout=_LIFECYCLE_TIMEOUT)
        return sandbox

    async def _attach(self, client: AsyncDaytona, sandbox_id: str) -> AsyncSandbox:
        """Attach to a sandbox that already exists, starting it if it is stopped.

        A sandbox in the `destroying` state, which `delete()` leaves until it is gone, counts as gone.
        """
        with anyio.move_on_after(_CREATE_TIMEOUT):
            async with _mapped_errors(f'Could not attach to Daytona sandbox {sandbox_id!r}'):
                try:
                    sandbox = await client.get(sandbox_id, request_timeout=_REQUEST_TIMEOUT)
                    if not _in_deleted_state(sandbox):
                        await sandbox.start(timeout=_LIFECYCLE_TIMEOUT)
                except daytona.DaytonaNotFoundError as error:
                    raise WorkspaceUnavailableError(
                        f'The Daytona sandbox {sandbox_id!r} was not found (it was deleted, or never existed '
                        "in this Daytona organization). Pass `workspace='new'` to start a fresh sandbox."
                    ) from error
            if _in_deleted_state(sandbox):
                raise WorkspaceUnavailableError(_unavailable_message(sandbox.id))
            return sandbox
        raise TimeoutError(f'Connecting to Daytona sandbox {sandbox_id!r} did not complete within {_CREATE_TIMEOUT}s.')

    async def _make_working_dir(self, sandbox: AsyncSandbox) -> None:
        directory = self._working_dir
        assert directory is not None
        with anyio.CancelScope(shield=True):
            async with _mapped_errors(f'Could not create working_dir {directory!r}', sandbox=sandbox):
                result = await sandbox.process.exec(f'mkdir -p -- {shlex.quote(directory)}', timeout=_REQUEST_TIMEOUT)
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
        if self._operations:
            # An operation in flight, such as a cancelled command deleting its session, still needs
            # the client: the last one to finish closes it.
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

    async def working_dir(self) -> str:
        """Return the canonical directory commands start in, probed once per connection."""
        async with self._operation():
            if self._resolved_working_dir is None:
                sandbox = await self.get_sandbox()
                async with _mapped_errors('Could not determine the working directory', sandbox=sandbox):
                    result = await sandbox.process.exec('pwd -P', cwd=self._working_dir, timeout=_REQUEST_TIMEOUT)
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
            process = _DaytonaProcess(sandbox)
            line = process.command_line(argv, cwd=self._working_dir, env={**self._env, **(env or {})})
            async with command_deadline(timeout, stop=process.kill, output=lambda: (process.stdout, process.stderr)):
                await process.start(line)
                result = await process.wait()
                if process.missing_cwd:
                    raise WorkspaceError(
                        f'working_dir {self._working_dir!r} does not exist in Daytona sandbox {sandbox.id}. '
                        'Create it there, or pass a working_dir that exists.'
                    )
            # Delete the finished session before the operation ends and the client may close.
            await run_to_completion(lambda: stop_shielded(process.kill))
            return result

    async def realpath(self, path: str) -> str:
        # Native, because the shell fallback would cost a whole session command per call.
        async with self._operation():
            _check_path(path)
            sandbox = await self.get_sandbox()
            async with _mapped_errors(f'Could not resolve {path!r}', sandbox=sandbox, path=path):
                result = await sandbox.process.exec(
                    f'realpath -m -- {shlex.quote(path)}', cwd=self._working_dir, timeout=_REQUEST_TIMEOUT
                )
            if result.exit_code != 0:
                raise WorkspaceError(f'Could not resolve {path!r}: {result.result}')
            return result.result.strip()

    async def read_bytes(self, path: str) -> bytes:
        async with self._operation():
            _check_path(path)
            sandbox = await self.get_sandbox()
            async with _mapped_errors(f'Could not access {path!r} in the sandbox', sandbox=sandbox, path=path):
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
            # The upload creates missing parents, follows a symlink, and keeps an existing file's mode.
            async with _mapped_errors(f'Could not access {path!r} in the sandbox', sandbox=sandbox, path=path):
                await sandbox.fs.upload_file(data, path, timeout=_REQUEST_TIMEOUT)

    async def stat(self, path: str) -> FileEntry:
        async with self._operation():
            _check_path(path)
            sandbox = await self.get_sandbox()
            async with _mapped_errors(f'Could not access {path!r} in the sandbox', sandbox=sandbox, path=path):
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
            async with _mapped_errors(f'Could not access {path!r} in the sandbox', sandbox=sandbox, path=path):
                entries = await sandbox.fs.list_files(path, request_timeout=_REQUEST_TIMEOUT)
                # The toolbox omits looping symlinks from listings; `find -type l` lists them without following them.
                links = await sandbox.process.exec(
                    f'find {shlex.quote(path)} -mindepth 1 -maxdepth 1 -type l', timeout=_REQUEST_TIMEOUT
                )
            result = [
                FileEntry(
                    name=entry.name,
                    path=posixpath.join(path, entry.name),
                    is_dir=entry.is_dir,
                    size=None if entry.is_dir else entry.size,
                )
                for entry in entries
            ]
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
            async with _mapped_errors(f'Could not access {path!r} in the sandbox', sandbox=sandbox, path=path):
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
            async with _mapped_errors(f'Could not access {path!r} in the sandbox', sandbox=sandbox, path=path):
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
