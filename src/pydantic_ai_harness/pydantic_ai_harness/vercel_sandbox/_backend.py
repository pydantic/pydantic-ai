"""A Vercel Sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

External assumptions, verified 2026-10-27 against `vercel-sandbox` 0.10.0 (the package floor), a live
Vercel sandbox, and the Vercel docs. Re-check them before changing acquisition, command, or error handling:

- A sandbox is identified by a `name` unique within the credentials' project. The backend names each
  sandbox it creates, so a create whose reply was lost is found by that name rather than duplicated.
- Sandboxes are persistent by default: when a session stops, Vercel saves the filesystem, and a
  sandbox-level SDK call such as `Sandbox.create_process` resumes the sandbox in a new session. A
  `Process` stays bound to the session it started in.
  https://vercel.com/docs/sandbox/concepts/persistent-sandboxes
- A session runs for 5 minutes unless created with a longer `execution_time_limit`: at most 45 minutes
  on Hobby and 24 hours on Pro and Enterprise. https://vercel.com/docs/sandbox/pricing#runtime-limits
- `Sandbox.destroy()` keeps the sandbox's snapshots, which are billed storage, unless passed
  `delete_orphan_snapshots=True`.
- Every non-2xx answer is a `SandboxApiError` carrying `status_code`, with no subclass per status.
  Missing credentials raise `SandboxCredentialsError` at the first request, not when the client is
  built. Transport failures are `httpx2` errors the SDK does not wrap.
- `create_sandbox` polls until the session runs with no deadline of its own; `Process.wait()` and the
  process log stream have none either, so the backend bounds each.
- A process's stdout and stderr arrive as text records of one log stream that both `TextReader`s
  share: reading one buffers the other's records, so the backend reads both at once to keep its
  output cap bounded. 0.9.1 replays the whole stream when its two readers run at once (a reader
  waiting for the other to finish reopens the closed stream), hence the 0.10.0 floor.
  `Process.kill()` sends `SIGKILL` to the process.
- `Process.wait()` returns only once every process holding the command's stdout or stderr has
  closed it, so a background child that inherits them holds up the result. Deleting or stopping the
  sandbox ends a running command with SIGTERM, which an in-flight `wait()` reports as exit code 143.
- Commands run as uid 1000 (`ubuntu` on the default image) with every Linux capability, so file
  permissions do not restrict them.

The SDK source is `vercel/sandbox/_internal/` in the `vercel-sandbox` distribution.
"""

from __future__ import annotations

import asyncio
import logging
import math
import posixpath
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING

import anyio
import anyio.lowlevel
from anyio.abc import TaskGroup

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai.workspaces.workspace import _ShellFilesystem  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    check_env_types,
    check_timeout,
    command_argv,
    run_to_completion,
    running_on_asyncio,
    safe_credential_reason,
    stop_shielded,
)

if TYPE_CHECKING:
    from vercel.sandbox import Process, Sandbox, SandboxClient, TextReader

    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('VercelSandboxBackend',)

try:
    from vercel import sandbox as vercel
except ImportError as _import_error:
    raise ImportError('Install `pydantic-ai-harness[vercel]` to use VercelSandbox.') from _import_error

logger = logging.getLogger(__name__)

_CREDENTIALS_HELP = (
    'Set VERCEL_OIDC_TOKEN (`vercel env pull` writes one), or VERCEL_TOKEN with VERCEL_TEAM_ID and '
    'VERCEL_PROJECT_ID, or pass a configured `SandboxClient` as `client=`.'
)
_AUTH_MESSAGE = f'Vercel rejected the credentials. {_CREDENTIALS_HELP}'
_MISSING_CREDENTIALS_MESSAGE = f'No Vercel credentials found. {_CREDENTIALS_HELP}'

# The longest session Vercel's Hobby plan allows, so the default works on every plan.
DEFAULT_SANDBOX_TIMEOUT = 2_700

# Bound sandbox creation, which the SDK polls without a deadline, so a wedged control plane cannot hang acquisition.
_CREATE_TIMEOUT = 120
# For a lookup, and for the backend's own short commands.
_REQUEST_TIMEOUT = 30
_TEARDOWN_TIMEOUT = 30.0
# Vercel reports a command's exit only once its output has closed; should the log stream outlast it,
# read on only this long.
_OUTPUT_DRAIN_GRACE = 2.0
# Past a command's timeout, Vercel kills it itself, in case the client cannot.
_KILL_AFTER_GRACE = 10.0
_READ_CHUNK = 64 * 1024
# The local backend's cap and preview: a command's whole output is held in memory.
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_OUTPUT_PREVIEW_CHARS = 64 * 1024
# The only stderr, with exit code 125, of a command whose `working_dir` is missing.
_MISSING_WORKING_DIR = f'pydantic-ai-missing-working-dir-{uuid.uuid4().hex}'


def _require_asyncio() -> None:
    if not running_on_asyncio():
        raise UserError(
            'Vercel Sandbox needs the asyncio event loop: the backend creates a sandbox in an asyncio task, '
            'so a run cancelled meanwhile still records it.'
        )


def _new_client() -> SandboxClient:
    """A `SandboxClient` that reads its credentials from the environment on each request; building it does no I/O."""
    return vercel.SandboxClient.create()


def _translate(error: Exception, context: str) -> Exception:
    """Map a Vercel SDK error onto the failures the workspace protocol promises.

    Returns `error` itself for what must propagate unchanged, for durable engines to retry:
    transport failures, request timeouts, rate limits, server errors, and anything that is not a
    `SandboxError`.
    """
    if isinstance(error, vercel.SandboxCredentialsError):
        return WorkspaceUnavailableError(_MISSING_CREDENTIALS_MESSAGE)
    if isinstance(error, vercel.SandboxApiError):
        status = error.status_code
        if status in (401, 403):
            return WorkspaceUnavailableError(f'{safe_credential_reason(error)}. {_AUTH_MESSAGE}')
        if 400 <= status < 500 and status not in (408, 429):
            return WorkspaceError(f'{context}: {error}')
    if isinstance(error, vercel.SandboxStreamError):
        # An error record the service put in the process log stream.
        return WorkspaceError(f'{context}: {error}')
    return error


@asynccontextmanager
async def _mapped_errors(context: str) -> AsyncGenerator[None]:
    """Raise the protocol's typed failure for a Vercel SDK error, and let the backend's own failures through."""
    try:
        yield
    except WorkspaceError:
        raise
    except Exception as error:
        translated = _translate(error, context)
        if translated is error:
            raise
        raise translated from error


async def destroy_sandbox(client: SandboxClient | None, name: str) -> None:
    """Delete a sandbox by name, with the snapshots no other sandbox uses, without resuming it.

    A sandbox that no longer exists returns quietly; rejected credentials raise
    `WorkspaceUnavailableError`, like any other operation. Without `client`, one is opened from
    the environment and closed afterwards.

    Raises:
        UserError: The event loop is not asyncio.
    """
    _require_asyncio()
    if not name.strip():
        raise ValueError('Vercel workspace ref id cannot be empty')
    owned: SandboxClient | None = None
    if client is None:
        client = owned = _new_client()
    try:
        async with _mapped_errors(f'Could not delete Vercel sandbox {name!r}'):
            try:
                # Looking a sandbox up does not resume it.
                sandbox = await client.get_sandbox(name=name)
                # Its automatic snapshots are billed storage that would otherwise outlive it.
                await sandbox.destroy(delete_orphan_snapshots=True)
            except vercel.SandboxApiError as error:
                if error.status_code == 404:
                    return
                raise
    finally:
        if owned is not None:
            await _close_quietly(owned)


async def _close_quietly(client: SandboxClient) -> None:
    """Close an API client, bounded and shielded; a failure is logged, not raised."""
    with anyio.move_on_after(_TEARDOWN_TIMEOUT, shield=True) as deadline:
        try:
            await client.aclose()
        except Exception as error:
            logger.warning('Could not close the Vercel Sandbox API client: %r', error)
    if deadline.cancelled_caught:
        logger.warning('Closing the Vercel Sandbox API client did not complete within %ss.', _TEARDOWN_TIMEOUT)


class _Command:
    """One process: its output, bounded by `_MAX_OUTPUT_BYTES` across both streams, and its exit.

    The two readers and the wait run in one task group. Each records a failure and cancels the group
    rather than raise it, so the caller raises it as itself, not wrapped in an `ExceptionGroup`.
    """

    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.process: Process | None = None
        self.stdout: list[str] = []
        self.stderr: list[str] = []
        self.size = 0
        self.exit_code: int | None = None
        self.timed_out = False
        self.error: Exception | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.exit_code is None and not self.timed_out

    async def collect(self) -> None:
        process = self.process
        assert process is not None
        try:
            async with anyio.create_task_group() as group:
                group.start_soon(self._read, process.stdout, self.stdout, group)
                group.start_soon(self._read, process.stderr, self.stderr, group)
                group.start_soon(self._wait, process, group)
        finally:
            # Closing both readers closes the log stream they share; each close is shielded.
            for reader in (process.stdout, process.stderr):
                if reader is not None:  # pragma: no branch - both streams are piped
                    await reader.aclose()

    async def _read(self, reader: TextReader | None, chunks: list[str], group: TaskGroup) -> None:
        assert reader is not None
        try:
            while chunk := await reader.read(_READ_CHUNK):
                self._add(chunks, chunk)
        except Exception as error:
            self._fail(error, group)

    async def _wait(self, process: Process, group: TaskGroup) -> None:
        try:
            with anyio.CancelScope(deadline=self.deadline) as scope:
                self.exit_code = await process.wait()
            if scope.cancelled_caught:
                self.timed_out = True
                # Killed rather than abandoned, so the readers drain what it printed last; the grace
                # below bounds them if the log stream stays open.
                await stop_shielded(process.kill)
        except Exception as error:
            self._fail(error, group)
            return
        group.cancel_scope.deadline = anyio.current_time() + _OUTPUT_DRAIN_GRACE

    def _add(self, chunks: list[str], data: str) -> None:
        # A character is at least one byte, so a chunk longer than what is left is over the cap
        # without encoding a copy of it.
        self.size += len(data) if len(data) > _MAX_OUTPUT_BYTES - self.size else len(data.encode())
        if self.size > _MAX_OUTPUT_BYTES:
            chunks.append(data[:_OUTPUT_PREVIEW_CHARS])
            stdout, stderr = self.text(_OUTPUT_PREVIEW_CHARS)
            raise WorkspaceOutputLimitError(
                'Vercel command output exceeded the 10 MiB limit; redirect it to a file and read part of it instead',
                limit=_MAX_OUTPUT_BYTES,
                stdout=stdout,
                stderr=stderr,
            )
        chunks.append(data)

    def _fail(self, error: Exception, group: TaskGroup) -> None:
        if self.error is None:
            self.error = error
        group.cancel_scope.cancel()

    def text(self, chars: int | None = None) -> tuple[str, str]:
        return ''.join(self.stdout)[:chars], ''.join(self.stderr)[:chars]


class VercelSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem):
    """A [Vercel Sandbox](https://vercel.com/docs/vercel-sandbox) as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands run inside a Firecracker microVM, so the host is never exposed. File operations run
    through the sandbox's shell, the way `Workspace` derives them from commands.

    Building one does no I/O. The first operation creates or attaches to a sandbox, and the typed
    `vercel.sandbox.Sandbox` is available through `get_sandbox()`. The backend does not stop or
    delete the sandbox; that is the application's job.

    Sandboxes are persistent: when a session ends, Vercel saves the files, and the next operation
    resumes the sandbox in a new session. A command running when its session ends fails with
    `WorkspaceError`. A deleted sandbox and rejected credentials raise `WorkspaceUnavailableError`.

    A command's complete output is returned once it exits. A shell string runs under `/bin/sh -c`;
    a timeout or cancellation kills the command.

    Args:
        client: A `vercel.sandbox.SandboxClient` to create or attach with. The caller owns closing
            it. Without one, the backend opens its own on first use, reading the credentials from
            the environment, and closes it in `aclose()`.
        ref: Identity of an existing sandbox to attach to on first use; its `id` is the sandbox name.
        image: Image a newly created sandbox runs; Vercel's default, `vercel/sandbox/universal`, when `None`.
        allow_internet_access: Whether a newly created sandbox may reach the internet.
        sandbox_timeout: Seconds each session of a newly created sandbox runs before Vercel stops it
            and saves its files. The default, 2700, is the most Vercel's Hobby plan allows; Pro and
            Enterprise allow up to 86400.
        working_dir: Absolute directory commands start in and relative paths resolve against; the
            sandbox's own default when `None`, discovered with `pwd -P` on first use.
        env: Environment variables every command gets; a command's own `env` is layered on top.
            They are also set on a newly created sandbox.
    """

    def __init__(
        self,
        *,
        client: SandboxClient | None = None,
        ref: WorkspaceRef | None = None,
        image: str | None = None,
        allow_internet_access: bool = True,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None and ref.provider != 'vercel':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'vercel'")
        if ref is not None and not ref.id.strip():
            raise ValueError('Vercel workspace ref id cannot be empty')
        if type(sandbox_timeout) is not int or sandbox_timeout < 1:
            raise ValueError(f'sandbox_timeout must be an integer of at least 1, got {sandbox_timeout!r}.')
        check_env_types(env)
        self._ref = ref
        self._sandbox: Sandbox | None = None
        self._client = client
        self._owns_client = client is None
        self._image = image
        self._allow_internet_access = allow_internet_access
        self._sandbox_timeout = sandbox_timeout
        self._env = dict(env or {})
        self._working_dir = absolute_path('working_dir', working_dir)
        # `working_dir()` must return a canonical absolute path, resolved once with `pwd -P`.
        self._resolved_working_dir: str | None = None
        # A sandbox this backend created still needs its `working_dir` made.
        self._working_dir_pending = False
        # Kept across attempts, so a retried create finds a sandbox whose reply was lost.
        self._create_name: str | None = None
        self._lock = anyio.Lock()
        # Set while nothing will call `aclose()`: the owned client then closes when the last
        # operation in flight ends.
        self._close_after_operation = False
        self._operations = 0

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the sandbox, or `None` before one has been created; its `id` is the sandbox name."""
        return self._ref

    async def get_sandbox(self) -> Sandbox:
        """Return the typed `vercel.sandbox.Sandbox`, creating or attaching to it on first use.

        Attaching by `ref` to a sandbox that no longer exists raises `WorkspaceUnavailableError`; it
        does not create a replacement. The handle uses the backend's client, so it stops working when
        `aclose()` closes a client the backend opened.
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

    async def _acquire_locked(self) -> Sandbox:
        """Create or attach to the sandbox; the caller holds `_lock`."""
        if self._client is None:
            self._client = _new_client()
        client = self._client
        if self._ref is not None:
            self._sandbox = await self._attach(client, self._ref.id)
            return self._sandbox

        async def create() -> Sandbox:
            created = await self._create(client)
            self._sandbox = created
            self._ref = WorkspaceRef(provider='vercel', id=created.name)
            self._working_dir_pending = self._working_dir is not None
            # Logged at creation, so the name can be found whatever ends the run before it is stored.
            logger.info('Created Vercel sandbox %s', created.name)
            return created

        # Creation finishes even if the caller is cancelled, so a billed sandbox is never left unrecorded.
        return await run_to_completion(create)

    async def _create(self, client: SandboxClient) -> Sandbox:
        if self._create_name is None:
            self._create_name = f'pydantic-ai-{uuid.uuid4().hex}'
        elif (existing := await self._find_created(client, self._create_name)) is not None:
            return existing
        name = self._create_name
        context = 'Could not create a Vercel sandbox'
        with anyio.move_on_after(_CREATE_TIMEOUT):
            try:
                return await client.create_sandbox(
                    name=name,
                    image=self._image,
                    execution_time_limit=timedelta(seconds=self._sandbox_timeout),
                    # Vercel's default, set so a stored ref reaches the files after the session ends.
                    persistent=True,
                    network_policy=None if self._allow_internet_access else vercel.NetworkPolicy.deny_all(),
                    env=self._env or None,
                )
            except vercel.SandboxTerminalStateError as error:
                raise WorkspaceUnavailableError(
                    f'The Vercel sandbox {name!r} did not start: its session is {error.status}.'
                ) from error
            except Exception as error:
                translated = _translate(error, context)
                if translated is error:
                    # The sandbox may have been created before the reply was lost.
                    if (created := await self._find_created(client, name)) is not None:
                        return created
                    raise
                if type(translated) is WorkspaceError:
                    # Vercel refused the request (an unknown image, a timeout over the plan's limit): no retry can fix it.
                    translated = WorkspaceUnavailableError(str(translated))
                raise translated from error
        if (created := await self._find_created(client, name)) is not None:
            return created
        # A stalled control plane is transient, not a command deadline.
        raise TimeoutError(
            f'Vercel sandbox creation did not complete within {_CREATE_TIMEOUT}s; the Vercel API may be unreachable.'
        )

    async def _find_created(self, client: SandboxClient, name: str) -> Sandbox | None:
        """The sandbox an earlier create under `name` made; `None` when Vercel has none, or cannot say."""
        with anyio.move_on_after(_REQUEST_TIMEOUT):
            try:
                return await client.get_sandbox(name=name)
            except Exception:
                pass
        return None

    async def _attach(self, client: SandboxClient, name: str) -> Sandbox:
        """Look up a sandbox that already exists; a stopped one resumes on its first command."""
        with anyio.move_on_after(_REQUEST_TIMEOUT):
            async with _mapped_errors(f'Could not attach to Vercel sandbox {name!r}'):
                try:
                    return await client.get_sandbox(name=name)
                except vercel.SandboxApiError as error:
                    if error.status_code != 404:
                        raise
                    raise WorkspaceUnavailableError(
                        f'The Vercel sandbox {name!r} was not found (it was deleted, or never existed in this '
                        "Vercel project). Pass `workspace='new'` to start a fresh one."
                    ) from error
        raise TimeoutError(f'Connecting to Vercel sandbox {name!r} did not complete within {_REQUEST_TIMEOUT}s.')

    async def _make_working_dir(self, sandbox: Sandbox) -> None:
        directory = self._working_dir
        assert directory is not None
        with anyio.CancelScope(shield=True):
            result = await self._exec(sandbox, ['mkdir', '-p', '--', directory], {}, _REQUEST_TIMEOUT)
            if result.exit_code != 0:
                raise WorkspaceError(
                    f'Could not create working_dir {directory!r} in Vercel sandbox {sandbox.name!r}: '
                    f'{result.stderr.strip()}'
                )
            self._working_dir_pending = False

    async def aclose(self) -> None:
        """Close the `SandboxClient` this backend opened for itself.

        A client passed as `client=` is left alone, and the sandbox keeps running. A later
        operation opens a new client and attaches by `ref`. `VercelSandbox` calls this when the run
        that used the backend ends.
        """
        if self._client is None or not self._owns_client or not running_on_asyncio():
            return
        if self._operations:
            # An operation in flight, such as a cancelled command being killed, still needs the
            # client: the last one to finish closes it.
            self._close_after_operation = True
            return

        async def close() -> None:
            # Under the lock, so a sandbox still being created is recorded before its client closes.
            async with self._lock:
                client, self._client, self._sandbox = self._client, None, None
            if client is not None:  # pragma: no branch - only a concurrent `aclose` clears it first
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
                client, self._client, self._sandbox = self._client, None, None
                await run_to_completion(lambda: _close_quietly(client))

    async def working_dir(self) -> str:
        """Return the canonical directory commands start in, probed once."""
        async with self._operation():
            if self._resolved_working_dir is None:
                sandbox = await self.get_sandbox()
                result = await self._run_in_working_dir(sandbox, ['pwd', '-P'], self._env, _REQUEST_TIMEOUT)
                printed = result.stdout.removesuffix('\n')
                if result.exit_code != 0 or not posixpath.isabs(printed):
                    raise WorkspaceError(
                        f'Could not determine the working directory of Vercel sandbox {sandbox.name!r}: '
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
        async with self._operation():
            check_timeout(timeout)
            argv = command_argv(command, shell)
            # Acquiring the sandbox has its own bound; the timeout is the command's alone.
            sandbox = await self.get_sandbox()
            return await self._run_in_working_dir(sandbox, argv, {**self._env, **(env or {})}, timeout)

    # The file operations Pydantic AI derives from `run` for a command-only backend, each as one
    # operation, so its commands share one API client.
    async def read_bytes(self, path: str) -> bytes:
        async with self._operation():
            return await _ShellFilesystem(self).read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        async with self._operation():
            await _ShellFilesystem(self).write_bytes(path, data)

    async def stat(self, path: str) -> FileEntry:
        async with self._operation():
            return await _ShellFilesystem(self).stat(path)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        async with self._operation():
            return await _ShellFilesystem(self).list_dir(path)

    async def make_dir(self, path: str) -> None:
        async with self._operation():
            await _ShellFilesystem(self).make_dir(path)

    async def remove(self, path: str) -> None:
        async with self._operation():
            await _ShellFilesystem(self).remove(path)

    async def exists(self, path: str) -> bool:
        async with self._operation():
            return await _ShellFilesystem(self).exists(path)

    async def _run_in_working_dir(
        self, sandbox: Sandbox, argv: Sequence[str], env: Mapping[str, str], timeout: float | None
    ) -> CommandResult:
        """Run `argv` under `sh` in `working_dir`, so a missing program exits 127 as it would in a shell."""
        if self._working_dir is None:
            launcher = ['/bin/sh', '-c', 'exec "$@"', 'sh', *argv]
        else:
            script = (
                f'cd -- "$1" 2>/dev/null || {{ printf %s {_MISSING_WORKING_DIR} >&2; exit 125; }}; shift; exec "$@"'
            )
            launcher = ['/bin/sh', '-c', script, 'sh', self._working_dir, *argv]
        result = await self._exec(sandbox, launcher, env, timeout)
        if self._working_dir is not None and result.exit_code == 125 and result.stderr == _MISSING_WORKING_DIR:
            raise WorkspaceError(
                f'working_dir {self._working_dir!r} does not exist in Vercel sandbox {sandbox.name!r}. '
                'Create it there, or pass a working_dir that exists.'
            )
        return result

    async def _exec(
        self, sandbox: Sandbox, argv: Sequence[str], env: Mapping[str, str], timeout: float | None
    ) -> CommandResult:
        """Run `argv` as one Vercel process; a timeout, cancellation, or failure kills it."""
        command = _Command(math.inf if timeout is None else anyio.current_time() + timeout)
        try:
            with anyio.CancelScope(deadline=command.deadline):
                command.process = await sandbox.create_process(
                    argv[0],
                    argv[1:],
                    env=dict(env) or None,
                    kill_after=None if timeout is None else timeout + _KILL_AFTER_GRACE,
                )
            if command.process is not None:
                await command.collect()
                if command.error is not None:
                    raise command.error
                if command.exit_code is None and not command.timed_out:
                    # The task group absorbed a cancellation of this task: asyncio passes a cancelled
                    # `gather`'s message on to its children, where it reads as an AnyIO cancel scope's,
                    # and the group takes it for its own. Re-raise it.
                    raise asyncio.CancelledError
        except Exception as error:
            if command.running:
                assert command.process is not None
                await stop_shielded(command.process.kill)
            if isinstance(error, WorkspaceError):
                raise
            context = (
                'Command could not run in the Vercel sandbox'
                if command.process is None
                else 'Could not read the command result (the command may still be running)'
            )
            # Asking Vercel what became of the sandbox tells a deleted one or an ended session apart.
            translated = await self._ended(sandbox) or _translate(error, context)
            if translated is error:
                raise
            raise translated from error
        except BaseException:
            if command.running:
                assert command.process is not None
                await stop_shielded(command.process.kill)
            raise
        stdout, stderr = command.text()
        if command.process is None or command.timed_out:
            assert timeout is not None
            raise WorkspaceTimeoutError(f'Command timed out after {timeout:g} seconds', stdout=stdout, stderr=stderr)
        assert command.exit_code is not None
        # Deleting or stopping the sandbox ends a running command with SIGTERM, reported as 143 like
        # any other signal, so a signal exit is checked against the sandbox's state.
        if command.exit_code > 128 and (ended := await self._ended(sandbox)) is not None:
            raise ended
        return CommandResult(exit_code=command.exit_code, stdout=stdout, stderr=stderr)

    async def _ended(self, sandbox: Sandbox) -> WorkspaceError | None:
        """Why `sandbox` cannot run a command now: deleted, or its session ended; `None` if neither, or Vercel cannot say."""
        client = self._client
        if client is None:  # pragma: no cover - only an operation ending during a direct `get_sandbox()` clears it
            return None
        with anyio.move_on_after(_REQUEST_TIMEOUT):
            try:
                current = await client.get_sandbox(name=sandbox.name)
            except vercel.SandboxApiError as lookup:
                if lookup.status_code == 404:
                    return WorkspaceUnavailableError(
                        f'The Vercel sandbox {sandbox.name!r} no longer exists (it was deleted). '
                        "Pass `workspace='new'` to start a fresh one."
                    )
            except Exception:
                # Any other failure, transport errors included, leaves the command's own outcome to stand.
                pass
            else:
                ended = (
                    vercel.SandboxStatus.STOPPING,
                    vercel.SandboxStatus.STOPPED,
                    vercel.SandboxStatus.SNAPSHOTTING,
                    vercel.SandboxStatus.FAILED,
                    vercel.SandboxStatus.ABORTED,
                )
                if current.status in ended:
                    return WorkspaceError(
                        f'The session of Vercel sandbox {sandbox.name!r} ended while the command ran: its '
                        '`sandbox_timeout` ran out, or it was stopped. The sandbox keeps its files, and the next '
                        'operation resumes it.'
                    )
        return None
