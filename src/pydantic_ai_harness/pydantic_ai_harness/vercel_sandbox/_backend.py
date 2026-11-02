"""A [Vercel Sandbox](https://vercel.com/docs/vercel-sandbox) behind Pydantic AI's `WorkspaceBackend` protocol.

SDK behavior read from the source of `vercel-sandbox` 0.10.0 on 2026-11-02. Re-check these against
<https://github.com/vercel/sandbox> before changing acquisition, command, or error handling:

- A sandbox is identified by its `name`, unique in its Vercel project. `SandboxClient.get_or_create_sandbox`
  returns the sandbox of that name, or creates it and waits until it runs, so a create retried under the same
  name finds the first one instead of making a second.
- A persistent sandbox (Vercel's default) is snapshotted when its session ends, by `stop()` or its
  `execution_time_limit`, and the SDK resumes it on the next call. A sandbox created with `persistent=False`
  cannot be resumed once its session ends.
- `Sandbox.run_process(command, args)` runs an argv, not a shell line, writes each output chunk to the
  `stdout` and `stderr` objects as it arrives, and hands `on_start` a `Process` whose `kill()` sends SIGKILL.
  It resumes a stopped sandbox and replays the call only if the process never started.
- The SDK decodes output as UTF-8 with `errors='replace'`.
- `SandboxClient.create()` does no I/O; it reads `VERCEL_OIDC_TOKEN`, or `VERCEL_TOKEN` with `VERCEL_TEAM_ID`
  and `VERCEL_PROJECT_ID`, on its first request, raising `SandboxCredentialsError` when neither is set.
  The API answers refused credentials with a 401 or 403 `SandboxApiError` and a missing sandbox with a 404.
  A failure while a process runs is a `SandboxStreamError`.
- `get_or_create_sandbox` waits for a sandbox it creates to run, but returns one that already exists as it
  is, which can still be `pending`.
- A `Sandbox` handle works through the client that fetched it, and fails once that client is closed.

From Vercel's docs (<https://vercel.com/docs/vercel-sandbox>, read 2026-11-02): a new sandbox boots the
`vercel/sandbox/universal` image, works in `/vercel/sandbox`, and has a 5 minute session limit unless
`execution_time_limit` says otherwise; plans cap a session at 45 minutes (Hobby) or 24 hours (Pro and
Enterprise).
"""

from __future__ import annotations

import io
import logging
import posixpath
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import anyio
import anyio.lowlevel

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    CommandResult,
    SupportsCommands,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    check_env,
    check_timeout,
    command_argv,
    command_deadline,
    run_to_completion,
    running_on_asyncio,
)

if TYPE_CHECKING:
    from vercel.sandbox import Process, Sandbox, SandboxClient

    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('VercelSandboxBackend',)

try:
    from vercel import sandbox as vercel_sandbox
except ImportError as _import_error:
    raise ImportError('Install `pydantic-ai-harness[vercel]` to use VercelSandbox.') from _import_error

_AUTH_MESSAGE = (
    'Vercel did not accept the credentials. Set VERCEL_OIDC_TOKEN, or VERCEL_TOKEN with VERCEL_TEAM_ID and '
    'VERCEL_PROJECT_ID, or pass a configured `vercel.sandbox.SandboxClient` as `client=`.'
)
_OUTPUT_LIMIT_MESSAGE = (
    'Vercel Sandbox command output exceeded the 10 MiB limit; redirect it to a file and read part of it instead'
)
_NAME_PREFIX = 'pydantic-ai-'
# The longest session Vercel's Hobby plan allows, so the default works on every plan; Pro and Enterprise allow
# 86_400. Vercel's own default, 5 minutes, would end the session under most agent runs.
DEFAULT_EXECUTION_TIME_LIMIT = 2_700
# Bound sandbox creation, attachment, and deletion so a wedged control plane cannot hang a run.
_LIFECYCLE_TIMEOUT = 120.0
_REQUEST_TIMEOUT = 30.0
_TEARDOWN_TIMEOUT = 30.0
_POLL_INTERVAL = 0.5
# The local backend's cap and preview: output is held in memory until the command ends.
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_OUTPUT_PREVIEW_BYTES = 64 * 1024
# Runs a command as `exec "$@"` in the directory `$1` (none when empty), with stdin at EOF. `exec` replaces
# the shell, so a kill reaches the command itself, a program named like a builtin is looked up on `PATH`,
# and a missing one exits 127. A missing directory exits 125 with only the marker `$2` on stderr.
_WRAPPER = 'if [ -n "$1" ]; then cd -- "$1" 2>/dev/null || { printf %s "$2" >&2; exit 125; }; fi; shift 2; exec "$@" </dev/null'
# A sandbox in these states never runs another command.
_DEAD_STATUSES = frozenset({'failed', 'aborted'})
_STOPPED_STATUSES = frozenset({'stopping', 'stopped', 'snapshotting'})

logger = logging.getLogger(__name__)


def _require_asyncio() -> None:
    """Raise `UserError` unless the caller runs on asyncio, which the backend's cancellation handling needs."""
    if not running_on_asyncio():
        raise UserError('Vercel Sandbox needs the asyncio event loop: the backend runs its calls on asyncio tasks.')


def _check_ref(ref: WorkspaceRef) -> None:
    """Raise `ValueError` unless `ref` names a Vercel sandbox."""
    if ref.provider != 'vercel':
        raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'vercel'")
    if not ref.id.strip():
        raise ValueError('Vercel workspace ref id cannot be empty')


def _new_client() -> SandboxClient:
    """A `SandboxClient` configured from the environment; it reads credentials on its first request."""
    return vercel_sandbox.SandboxClient.create()


async def _close_quietly(client: SandboxClient) -> None:
    """Close an API client, bounded and shielded; a failure is logged, not raised."""
    with anyio.move_on_after(_TEARDOWN_TIMEOUT, shield=True) as deadline:
        try:
            await client.aclose()
        except Exception as error:
            logger.warning('Could not close the Vercel Sandbox client: %r', error)
    if deadline.cancelled_caught:
        logger.warning('Closing the Vercel Sandbox client did not complete within %ss.', _TEARDOWN_TIMEOUT)


def _gone_message(name: str) -> str:
    return f"The Vercel sandbox {name!r} no longer exists. Pass `workspace='new'` to start a fresh sandbox."


def _status(error: Exception) -> int | None:
    return error.status_code if isinstance(error, vercel_sandbox.SandboxApiError) else None


def _credentials_rejected(error: Exception) -> bool:
    """Whether `error` says Vercel found no credentials or refused them, rather than that it could not be asked."""
    return isinstance(error, vercel_sandbox.SandboxCredentialsError) or _status(error) in (401, 403)


def _translate(error: Exception, context: str) -> Exception:
    """Map a Vercel SDK error onto the failures the workspace protocol promises.

    Returns `error` itself for what must propagate unchanged: connection failures, rate limits, server
    errors, and anything that is not a `SandboxError`, which durable engines retry as transient.
    """
    if _credentials_rejected(error):
        return WorkspaceUnavailableError(_AUTH_MESSAGE)
    if _status(error) in (400, 409, 422):
        return WorkspaceError(f'{context}: {error}')
    return error


async def _probe(client: SandboxClient, name: str, *, running: bool) -> WorkspaceUnavailableError | None:
    """`WorkspaceUnavailableError` if the sandbox is gone or its session ended; an inconclusive lookup counts as running."""
    try:
        with anyio.fail_after(_REQUEST_TIMEOUT):
            sandbox = await client.get_sandbox(name=name)
    except Exception as error:
        return WorkspaceUnavailableError(_gone_message(name)) if _status(error) == 404 else None
    return _refusal(sandbox, running=running)


def _refusal(sandbox: Sandbox, *, running: bool) -> WorkspaceUnavailableError | None:
    """Why `sandbox` cannot run commands, or `None` when it can.

    With `running`, a command had started, so a stopped persistent sandbox also counts: its session ended
    under the command, which a later operation resumes from the snapshot. Before a command starts, the SDK
    resumes a stopped persistent sandbox itself, so a failure then is the failure to report.
    """
    status = sandbox.status
    if status in _DEAD_STATUSES:
        return WorkspaceUnavailableError(_gone_message(sandbox.name))
    if status in _STOPPED_STATUSES:
        if sandbox.persistent is False:  # `None` when the API leaves it out; Vercel's default is persistent
            return WorkspaceUnavailableError(
                f'The Vercel sandbox {sandbox.name!r} is not persistent and its session has ended, so it cannot '
                "be resumed. Pass `workspace='new'` to start a fresh sandbox."
            )
        if running:
            return WorkspaceUnavailableError(
                f'The session of the Vercel sandbox {sandbox.name!r} ended during the command; the next operation '
                'resumes the sandbox from its snapshot, without the processes it was running.'
            )
    return None


@asynccontextmanager
async def _mapped_errors(
    context: str,
    *,
    client: SandboxClient | None = None,
    name: str | None = None,
    started: Sequence[Process] = (),
) -> AsyncGenerator[None]:
    """Raise the protocol's typed failure for a Vercel SDK error, and let the backend's own failures through.

    A sandbox destroyed or stopped under an operation surfaces as a stream or API failure, so with `name`,
    a failure that does not already end the run is followed by one lookup of the sandbox. `started` holds
    the command's process once Vercel reports it running.
    """
    try:
        yield
    except WorkspaceError:
        raise
    except Exception as error:
        translated = _translate(error, context)
        if client is not None and name is not None and not isinstance(translated, WorkspaceUnavailableError):
            if (gone := await _probe(client, name, running=bool(started))) is not None:
                raise gone from error
        if translated is error:
            raise
        raise translated from error


async def destroy_sandbox(client: SandboxClient | None, name: str) -> None:
    """Destroy a sandbox by name, whatever state it is in, keeping its snapshots until they expire.

    A sandbox that no longer exists returns quietly; rejected credentials raise
    `WorkspaceUnavailableError`, like any other operation. Without `client`, one is opened from the
    environment and closed afterwards.

    Raises:
        UserError: The event loop is not asyncio.
        TimeoutError: Vercel did not answer in time.
    """
    _require_asyncio()
    if not name.strip():
        raise ValueError('Vercel workspace ref id cannot be empty')
    owned: SandboxClient | None = None
    if client is None:
        client = owned = _new_client()
    try:
        with anyio.fail_after(_LIFECYCLE_TIMEOUT):
            try:
                sandbox = await client.get_sandbox(name=name)
                await sandbox.destroy()
            except vercel_sandbox.SandboxError as error:
                if _status(error) == 404:
                    return
                if _credentials_rejected(error):
                    raise WorkspaceUnavailableError(_AUTH_MESSAGE) from error
                raise
    finally:
        if owned is not None:
            await _close_quietly(owned)


class _Output(io.StringIO):
    """One of a command's output streams, bounded with its sibling by `_MAX_OUTPUT_BYTES` across both."""

    def __init__(self, sizes: list[int]) -> None:
        super().__init__()
        # Shared with the sibling stream: the byte count of both.
        self._sizes = sizes

    def write(self, s: str, /) -> int:
        written = super().write(s)
        self._sizes[0] += len(s.encode(errors='replace'))
        if self._sizes[0] > _MAX_OUTPUT_BYTES:
            raise _OutputOverflow
        return written


def _preview(text: str) -> str:
    return text.encode(errors='replace')[:_OUTPUT_PREVIEW_BYTES].decode(errors='replace')


class _OutputOverflow(Exception):
    """Raised from an output stream's `write` to end a command whose output passed the limit."""


class VercelSandboxBackend(WorkspaceBackend, SupportsCommands):
    """A [Vercel Sandbox](https://vercel.com/docs/vercel-sandbox) as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands run inside a Vercel Firecracker microVM. File operations run as shell commands there,
    through [`Workspace`][pydantic_ai.workspaces.Workspace].

    Building one does no I/O. The first operation creates a sandbox named `pydantic-ai-` and 32 hex
    digits, or attaches to the one `ref` names; `get_sandbox()` returns its `vercel.sandbox.Sandbox`.
    The backend does not stop or destroy the sandbox; that is the application's job.

    A command's output streams back while it runs and is returned once it exits. A shell string runs
    under `/bin/sh -c`; a timeout or cancellation kills the command.

    A destroyed sandbox, one whose session ended and that is not persistent, and rejected credentials
    raise `WorkspaceUnavailableError`.

    Args:
        client: A `vercel.sandbox.SandboxClient` to create or attach with. The caller owns closing it.
            Without one, the backend opens its own from the environment on first use and closes it in
            `aclose()`.
        ref: Identity of an existing sandbox to attach to on first use.
        image: Image a newly created sandbox boots from; Vercel's `vercel/sandbox/universal` when `None`.
        vcpus: vCPUs of a newly created sandbox, with 2 GB of memory each; Vercel's default (2) when `None`.
        persistent: Whether a newly created sandbox is snapshotted when its session ends, so a later
            operation resumes it with its files. When `False`, the sandbox is gone once its session ends.
        execution_time_limit: Seconds a newly created sandbox's session runs before Vercel stops it. The
            default, 45 minutes, is the most Vercel's Hobby plan allows.
        working_dir: Absolute directory commands start in and relative paths resolve against; the
            sandbox's own default (`/vercel/sandbox`) when `None`.
        env: Environment variables every command gets; a command's own `env` is layered on top.
            Nothing is read from the host environment.
    """

    def __init__(
        self,
        *,
        client: SandboxClient | None = None,
        ref: WorkspaceRef | None = None,
        image: str | None = None,
        vcpus: int | None = None,
        persistent: bool = True,
        execution_time_limit: int = DEFAULT_EXECUTION_TIME_LIMIT,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None:
            _check_ref(ref)
        check_env(env)
        self._ref = ref
        self._client = client
        self._owns_client = client is None
        self._image = image
        self._vcpus = vcpus
        self._persistent = persistent
        self._execution_time_limit = execution_time_limit
        self._env = dict(env or {})
        self._working_dir = absolute_path('working_dir', working_dir)
        # Written to stderr, and nothing else, by a command whose working directory is missing.
        self._missing_cwd_marker = f'pydantic-ai-missing-working-dir-{uuid.uuid4().hex}'
        self._create_name: str | None = None
        self._sandbox: Sandbox | None = None
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

        The `id` is the sandbox's name, unique in its Vercel project.
        """
        return self._ref

    async def get_sandbox(self) -> Sandbox:
        """Return the sandbox's current `vercel.sandbox.Sandbox`, creating or attaching to it on first use.

        Use it for Vercel features the workspace API does not cover, such as ports, snapshots, and
        network policies. The handle works through this backend's client, so it stops working once
        `aclose()` closes that client; outside a run, call `get_sandbox(name=ref.id)` on your own
        `SandboxClient` instead. Attaching by `ref` to a sandbox that no longer exists raises
        `WorkspaceUnavailableError`; it does not create a replacement.
        """
        async with self._operation():
            client, sandbox = await self._acquire()
            async with _mapped_errors('Could not read the Vercel sandbox', client=client, name=sandbox.name):
                return await client.get_sandbox(name=sandbox.name)

    async def _acquire(self) -> tuple[SandboxClient, Sandbox]:
        """The API client and a sandbox that accepts commands, creating or attaching on first use."""
        _require_asyncio()
        async with self._lock:
            if self._client is None:
                self._client = _new_client()
            client = self._client
            if self._sandbox is None:
                if self._ref is None:
                    # Creation finishes even if the caller is cancelled, so a billed sandbox is never left unrecorded.
                    self._sandbox = await run_to_completion(lambda: self._create(client))
                else:
                    self._sandbox = await self._attach(client, self._ref.id)
            sandbox = self._sandbox
            if self._working_dir_pending:
                # A caller cancelled during creation stops here; the next operation makes the directory.
                await anyio.lowlevel.checkpoint_if_cancelled()
                await self._make_working_dir(client, sandbox)
        await anyio.lowlevel.checkpoint_if_cancelled()
        return client, sandbox

    async def _create(self, client: SandboxClient) -> Sandbox:
        """Create a sandbox under a name kept across retries, so a retried create finds it rather than making another."""
        if self._create_name is None:
            self._create_name = f'{_NAME_PREFIX}{uuid.uuid4().hex}'
        name = self._create_name
        resources = None if self._vcpus is None else vercel_sandbox.SandboxResources(vcpus=self._vcpus)
        # Logged before the request, so the name can be found whatever ends the run before it is stored.
        logger.info('Creating Vercel sandbox %s', name)
        with anyio.fail_after(_LIFECYCLE_TIMEOUT):
            try:
                sandbox, created = await client.get_or_create_sandbox(
                    name=name,
                    image=self._image,
                    resources=resources,
                    persistent=self._persistent,
                    execution_time_limit=self._execution_time_limit,
                )
            except vercel_sandbox.SandboxTerminalStateError as error:
                raise WorkspaceUnavailableError(f'Vercel could not start the sandbox: {error}') from error
            except vercel_sandbox.SandboxError as error:
                translated = _translate(error, 'Vercel refused to create the sandbox')
                if translated is error:
                    raise
                # An invalid image or size: no retry can fix it.
                raise WorkspaceUnavailableError(str(translated)) from error
        if not created:
            # A retry found the sandbox an earlier attempt made, which can still be starting.
            sandbox = await self._attach(client, name)
        self._ref = WorkspaceRef(provider='vercel', id=sandbox.name)
        self._working_dir_pending = self._working_dir is not None
        return sandbox

    async def _attach(self, client: SandboxClient, name: str) -> Sandbox:
        """The sandbox `name` names, once it has left `pending`; one that cannot run commands is refused."""
        with anyio.move_on_after(_LIFECYCLE_TIMEOUT):
            async with _mapped_errors(f'Could not attach to Vercel sandbox {name!r}'):
                while True:
                    try:
                        sandbox = await client.get_sandbox(name=name)
                    except vercel_sandbox.SandboxApiError as error:
                        if error.status_code != 404:
                            raise
                        raise WorkspaceUnavailableError(
                            f'The Vercel sandbox {name!r} was not found (it was destroyed, or never existed in '
                            "this project). Pass `workspace='new'` to start a fresh sandbox."
                        ) from error
                    if (refusal := _refusal(sandbox, running=False)) is not None:
                        raise refusal
                    if sandbox.status != 'pending':
                        # A stopped persistent sandbox is resumed by its first command.
                        return sandbox
                    await anyio.sleep(_POLL_INTERVAL)
        raise TimeoutError(f'Vercel sandbox {name!r} did not start within {_LIFECYCLE_TIMEOUT:g}s.')

    async def _make_working_dir(self, client: SandboxClient, sandbox: Sandbox) -> None:
        directory = self._working_dir
        assert directory is not None
        with anyio.CancelScope(shield=True):
            async with _mapped_errors(f'Could not create working_dir {directory!r}', client=client, name=sandbox.name):
                with anyio.fail_after(_REQUEST_TIMEOUT):
                    result = await sandbox.run_process('mkdir', ['-p', '--', directory], capture_output=True)
            if result.returncode != 0:
                raise WorkspaceError(
                    f'Could not create working_dir {directory!r} in Vercel sandbox {sandbox.name}: '
                    f'{(result.stderr or "").strip()}'
                )
            self._working_dir_pending = False

    async def aclose(self) -> None:
        """Close the `SandboxClient` this backend opened for itself.

        A client passed as `client=` is left alone, and the sandbox keeps running. A later operation
        opens a new client and attaches by `ref`. `VercelSandbox` calls this when the run that used the
        backend ends.
        """
        if self._client is None or not self._owns_client or not running_on_asyncio():
            return
        if self._operations:
            # An operation in flight, such as a cancelled command still being killed, needs the
            # client: the last one to finish closes it.
            self._close_after_operation = True
            return

        async def close() -> None:
            # Under the lock, so a sandbox still being created is recorded before its client closes.
            async with self._lock:
                client = self._detach_client()
            if client is not None:
                await _close_quietly(client)

        await run_to_completion(close)

    def _detach_client(self) -> SandboxClient | None:
        """Forget the owned client and the handles made through it; a later operation attaches by `ref` again."""
        client, self._client = self._client, None
        self._sandbox = None
        self._resolved_working_dir = None
        return client

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[None]:
        self._operations += 1
        try:
            yield
        finally:
            self._operations -= 1
            if self._operations == 0 and self._close_after_operation and self._owns_client and self._client is not None:
                # Detached before any await, so an operation starting meanwhile opens its own client.
                client = self._detach_client()
                assert client is not None
                await run_to_completion(lambda: _close_quietly(client))

    async def working_dir(self) -> str:
        """Return the canonical directory commands start in, probed once per connection."""
        async with self._operation():
            if self._resolved_working_dir is None:
                result = await self.run(['pwd', '-P'], timeout=_REQUEST_TIMEOUT)
                printed = result.stdout.removesuffix('\n')
                if result.exit_code != 0 or not posixpath.isabs(printed):
                    raise WorkspaceError(
                        'Could not determine the working directory of the Vercel sandbox: '
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
            check_env(env)
            # Acquiring the sandbox has its own bound; the timeout is the command's alone.
            client, sandbox = await self._acquire()
            sizes = [0]
            stdout, stderr = _Output(sizes), _Output(sizes)
            started: list[Process] = []

            async def kill() -> None:
                if started:
                    await started[0].kill()

            def output() -> tuple[str, str]:
                return stdout.getvalue(), stderr.getvalue()

            async with command_deadline(timeout, stop=kill, output=output):
                async with _mapped_errors(
                    'Could not run the command', client=client, name=sandbox.name, started=started
                ):
                    try:
                        completed = await sandbox.run_process(
                            '/bin/sh',
                            ['-c', _WRAPPER, 'sh', self._working_dir or '', self._missing_cwd_marker, *argv],
                            env={**self._env, **(env or {})},
                            stdout=stdout,
                            stderr=stderr,
                            on_start=started.append,
                            # Vercel kills the command itself if the client-side kill never reaches it, such
                            # as when the deadline passes before `on_start` reports the process.
                            kill_after=timeout,
                        )
                    except _OutputOverflow:
                        # Raising out of `command_deadline` kills the command.
                        raise WorkspaceOutputLimitError(
                            _OUTPUT_LIMIT_MESSAGE,
                            limit=_MAX_OUTPUT_BYTES,
                            stdout=_preview(stdout.getvalue()),
                            stderr=_preview(stderr.getvalue()),
                        ) from None
            out, err = output()
            if completed.returncode == 125 and not out and err == self._missing_cwd_marker:
                raise WorkspaceError(
                    f'working_dir {self._working_dir!r} does not exist in Vercel sandbox {sandbox.name}. '
                    'Create it there, or pass a working_dir that exists.'
                )
            return CommandResult(exit_code=completed.returncode, stdout=out, stderr=err)
