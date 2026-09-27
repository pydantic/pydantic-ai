"""An E2B sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

SDK assumptions verified 2026-09-09 against E2B 2.46.4 and 2.34.0: `connect` resumes
paused workspaces, with a 300-second lifetime unless it is given `timeout`; commands use
`/bin/bash -l -c`, the SDK command timeout does not kill the process, and command handles
accumulate output. `create(lifecycle=...)` exists from 2.48.0, the package floor. Re-check
https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_async/main.py and
https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_async/commands/command.py
before changing acquisition or deadline behavior.

Exception types verified 2026-09-25 against E2B 2.51.0
(https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/exceptions.py): a missing key or
a 401 is `AuthenticationException`; a gone sandbox is `SandboxNotFoundException` from the control
plane and a `TimeoutException` from envd (its proxy's 502); a 429 is `RateLimitException`, a
`SandboxException` subclass; a create the API refuses is a `SandboxException` carrying its 4xx
`status_code`; a 503 is `ServiceBusyException` (new in 2.48.0, hence the floor),
which is not. envd types only a missing path; its other path failures arrive as a 400
(`InvalidArgumentException`) or a 500 (`SandboxException`) whose message carries envd's wording
or Go's errno text, which `_path_error` reads.
"""

from __future__ import annotations

import asyncio
import math
import posixpath
import shlex
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import anyio
import anyio.lowlevel
import sniffio
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from pydantic_ai_harness._workspace_provider import absolute_path, command_argv, safe_credential_reason, stop_shielded

if TYPE_CHECKING:
    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('E2BSandboxBackend',)

try:
    import e2b
except ImportError as error:  # pragma: no cover - exercised by the isolated missing-extra test
    raise ImportError('Install `pydantic-ai-harness[e2b]` to use E2BSandbox.') from error

_AUTH_MESSAGE = 'E2B rejected the credentials. Set a valid E2B_API_KEY in the environment.'

# envd's own messages and Go's errno text, most specific first: making a directory where a file
# exists says both "already exists" and "not a directory", and "is not a directory" must not
# match "is a directory".
_PATH_ERRORS: tuple[tuple[str, type[OSError]], ...] = (
    ('already exists', FileExistsError),
    ('permission denied', PermissionError),
    ('not a directory', NotADirectoryError),
    ('path is a file', NotADirectoryError),
    ('is a directory', IsADirectoryError),
)

# The most E2B's Hobby plan allows, so the default works on every plan; Pro plans allow 86_400.
DEFAULT_SANDBOX_TIMEOUT = 3_600

# Bound the sandbox-create call so a wedged control plane cannot hang acquisition.
_CREATE_TIMEOUT = 120

# Bounds the internal `pwd` probe behind `working_dir()` and the best-effort kills.
_INTERNAL_EXEC_TIMEOUT = 10

# E2B's own command `timeout` bounds the event stream and leaves the command running, so it is
# switched off (0 is the SDK's "no limit") and the deadline is enforced client-side instead,
# with a kill at expiry. See `E2BSandboxBackend.run`.
_SDK_STREAM_UNBOUNDED = 0


def _path_error(error: Exception, path: str) -> OSError | None:
    """The builtin path error an untyped envd failure describes, or `None` if it is not one."""
    message = str(error).lower()
    for marker, builtin in _PATH_ERRORS:
        if marker in message:
            return builtin(f'{error} (in the E2B sandbox: {path!r})')
    return None


async def _file_entry(sandbox: e2b.AsyncSandbox, entry: e2b.EntryInfo) -> FileEntry:
    """The protocol entry for `entry`, with `is_dir` and `size` following a symlink.

    envd describes a symlink with the link's own size and, as `symlink_target`, the path it
    resolves to, so a symlink entry is completed by stat-ing that target. A dangling link, which
    envd reports with itself as the target, reads as a file with no size.
    """
    target: e2b.EntryInfo | None = entry
    if entry.symlink_target is not None:
        try:
            target = await sandbox.files.get_info(posixpath.join(posixpath.dirname(entry.path), entry.symlink_target))
        except e2b.FileNotFoundException:
            target = None
        if target is not None and target.symlink_target is not None:
            target = None
    is_dir = target is not None and target.type is e2b.FileType.DIR
    # A directory's reported size is an implementation detail of the underlying filesystem
    # rather than a content length, so report none for it, like the built-in backends.
    size = None if target is None or is_dir else target.size
    return FileEntry(name=entry.name, path=entry.path, is_dir=is_dir, size=size)


def _unavailable_message(sandbox_id: str) -> str:
    return (
        f'The E2B sandbox {sandbox_id!r} is no longer running: it was killed, or it was paused when its '
        '`sandbox_timeout` ran out (a later run that attaches to it resumes it). '
        "Pass `workspace='new'` to start a fresh sandbox."
    )


def _is_lifetime_refusal(error: e2b.SandboxException) -> bool:
    """Whether E2B refused the requested `sandbox_timeout`, e.g. `400: Timeout cannot be greater than 1 hours`."""
    status = error.status_code
    # E2B can also return HTTP 400 for a transport i/o timeout during creation.
    return status is not None and 400 <= status < 500 and 'timeout cannot be greater than' in str(error).lower()


def _refused_message(context: str, error: e2b.SandboxException) -> str:
    message = f'{context}: {error}'
    if _is_lifetime_refusal(error):
        message += ' Hobby plans allow at most 3600 seconds; pass `E2BSandbox(sandbox_timeout=3600)`.'
    return message


class E2BSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem):
    """An [E2B](https://e2b.dev) sandbox as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands and file operations run inside an E2B microVM, so the host is never exposed.

    Building one does no I/O. The first operation creates or attaches to a workspace, and the
    typed `e2b.AsyncSandbox` is available through `get_sandbox()`. The backend does not kill the
    sandbox; killing it is the application's job.

    Commands run as one-shot operations, with complete output returned after they finish.

    Every command runs through `/bin/bash -l -c`, so an argv sequence is quoted into a single
    shell word string first and login startup files run before the command does; a
    `shell=True` string runs under `/bin/sh -c` inside that login shell. E2B's own
    command `timeout` abandons the output stream and leaves the command running, so the
    deadline is enforced client-side instead. On timeout or cancellation (including while
    starting), a separate command signals the process group when `setsid` is available.
    Custom templates without `setsid` fall back to stopping the command leader only.

    Args:
        sandbox: A live `e2b.AsyncSandbox` you already have. Whoever created it owns killing it.
        ref: Identity of an existing sandbox to attach to on first use.
        template: E2B template name or ID a newly created sandbox runs; E2B's default when `None`.
            An unknown template raises `WorkspaceUnavailableError` on first use. Custom templates
            need `/bin/bash` and `/bin/sh`; without `setsid`, stop targets only the leader.
        sandbox_timeout: Total lifetime of the sandbox in seconds, applied when it is created and
            again when attaching to it. When it runs out, E2B pauses the sandbox rather than
            killing it, and attaching resumes it. The default, 3600, is the most E2B's Hobby plan
            allows; Pro plans allow up to 86400.
        working_dir: Absolute directory commands start in and relative paths resolve against.
            E2B has no create-time working directory, so this is applied per command, including
            on an attached sandbox; `None` uses the sandbox's own default, discovered with
            `pwd -P` on first use.
        env: Environment variables every command gets, on a created or an attached sandbox;
            per-command `env` is layered on top. Nothing is read from the host environment.
        allow_internet_access: Whether a newly created sandbox may reach the internet.
    """

    def __init__(
        self,
        *,
        sandbox: e2b.AsyncSandbox | None = None,
        ref: WorkspaceRef | None = None,
        template: str | None = None,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        allow_internet_access: bool = True,
    ) -> None:
        if ref is not None and ref.provider != 'e2b':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'e2b'")
        if sandbox is not None and ref is not None:
            raise ValueError('pass either `sandbox` or `ref`, not both')
        self._ref = ref if sandbox is None else WorkspaceRef(provider='e2b', id=sandbox.sandbox_id)
        self._sandbox = sandbox
        self._working_dir = absolute_path('working_dir', working_dir)
        # `working_dir()` must return a canonical absolute path: the configured one, or the
        # sandbox's default, resolved once with `pwd -P`.
        self._resolved_working_dir: str | None = None
        self._lock = anyio.Lock()
        self._acquisition: asyncio.Task[e2b.AsyncSandbox] | None = None
        self._template = template
        self._sandbox_timeout = sandbox_timeout
        self._env = dict(env) if env is not None else None
        self._allow_internet_access = allow_internet_access
        self._probe_setsid = template is not None or sandbox is not None or ref is not None
        self._setsid: bool | None = None
        self._setsid_lock = anyio.Lock()

    async def get_sandbox(self) -> e2b.AsyncSandbox:
        """Return the typed `e2b.AsyncSandbox`, creating or attaching to it on first use.

        Every operation takes the handle from here, so none reaches an unacquired one. The lock
        serializes concurrent first uses -- two callers each creating a sandbox would leave the
        loser billed and unreferenced. A caller cancelled while E2B creates the sandbox still records
        it before the cancellation propagates, so `ref` names it and a retry reuses it.
        Attaching by `ref` to a sandbox that no longer exists raises
        `WorkspaceUnavailableError`; it does not create a replacement.
        """
        if self._ref is None and sniffio.current_async_library() == 'asyncio':
            # An AnyIO shield cannot stop native Task.cancel(); the backend owns this task
            # until it has recorded the paid sandbox, even if every caller leaves.
            task = self._acquisition
            if task is None:
                task = asyncio.create_task(self._acquire_detached(), name='e2b-sandbox-acquisition')
                self._acquisition = task
            return await asyncio.shield(task)
        return await self._acquire()

    async def _acquire_detached(self) -> e2b.AsyncSandbox:
        try:
            return await self._acquire()
        finally:
            self._acquisition = None

    async def _acquire(self) -> e2b.AsyncSandbox:
        async with self._lock:
            if (sandbox := self._sandbox) is not None:
                return sandbox
            ref = self._ref
            sandbox = await self._attach(ref.id) if ref is not None else await self._create()
            self._sandbox = sandbox
            self._ref = WorkspaceRef(provider='e2b', id=sandbox.sandbox_id)
            if ref is None and self._working_dir is not None:
                # Record the new sandbox before setup, so a failed mkdir still leaves
                # a ref the caller can use to clean up the billed sandbox.
                with anyio.CancelScope(shield=True):
                    async with self._sdk_errors(sandbox.sandbox_id, 'Could not create working_dir', self._working_dir):
                        await sandbox.files.make_dir(self._working_dir)
        await anyio.lowlevel.checkpoint_if_cancelled()
        return sandbox

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the sandbox, or `None` before one has been created."""
        return self._ref

    @asynccontextmanager
    async def _sdk_errors(self, sandbox_id: str | None, context: str, path: str | None = None) -> AsyncGenerator[None]:
        """Raise E2B's exceptions as the protocol's typed failures; see `_translate`."""
        try:
            yield
        except Exception as error:
            translated = await self._translate(error, context, path, sandbox_id)
            if translated is error:
                raise
            raise translated from error

    async def _translate(self, error: Exception, context: str, path: str | None, sandbox_id: str | None) -> Exception:
        """Map one E2B exception onto the protocol's typed failures, or return it unchanged.

        Rejected credentials and a gone sandbox end the run. E2B types an unanswered envd request
        as `TimeoutException` (the SDK's command timeout is disabled); after a liveness probe, a
        gone sandbox is unavailable, anything else is transient. It is not a
        `WorkspaceTimeoutError`, which is reserved for a command's own `timeout=`. Other SDK
        errors mean the operation failed. Rate limits, a busy service, transport failures, and
        anything E2B does not type come back unchanged to propagate as transient, for durable
        engines to retry.
        """
        if isinstance(error, e2b.InvalidArgumentException) and 'cwd ' in str(error) and 'does not exist' in str(error):
            return FileNotFoundError(str(error))
        if isinstance(error, e2b.AuthenticationException):
            return WorkspaceUnavailableError(f'{_AUTH_MESSAGE} {safe_credential_reason(error)}.')
        if isinstance(error, e2b.SandboxNotFoundException) and sandbox_id is not None:
            return WorkspaceUnavailableError(_unavailable_message(sandbox_id))
        if isinstance(error, e2b.FileNotFoundException):
            return FileNotFoundError(f'No such file or directory in the E2B sandbox: {path!r}')
        if (
            path is not None
            and type(error) in (e2b.SandboxException, e2b.InvalidArgumentException)
            and (path_error := _path_error(error, path)) is not None
        ):
            return path_error
        if isinstance(error, e2b.TimeoutException):
            sandbox = self._sandbox
            if sandbox is not None and not await _is_running(sandbox):
                return WorkspaceUnavailableError(_unavailable_message(sandbox.sandbox_id))
            return error
        if isinstance(error, e2b.SandboxException) and not isinstance(error, e2b.RateLimitException):
            return WorkspaceError(f'{context}: {error}')
        return error

    async def read_bytes(self, path: str) -> bytes:
        sandbox = await self.get_sandbox()
        async with self._sdk_errors(sandbox.sandbox_id, f'Could not read {path!r}', path):
            # envd reports FIFOs as files and its read blocks indefinitely without a writer.
            # Probe via the shell before asking envd to open the path (also follows links).
            probe = await self.run(f'test -p {shlex.quote(path)}', shell=True, timeout=_INTERNAL_EXEC_TIMEOUT)
            if probe.exit_code == 0:
                raise OSError(f'Could not read {path!r}: FIFO reads are not supported')
            return bytes(await sandbox.files.read(path, 'bytes'))

    async def write_bytes(self, path: str, data: bytes) -> None:
        sandbox = await self.get_sandbox()
        async with self._sdk_errors(sandbox.sandbox_id, f'Could not write {path!r}', path):
            # The SDK's default 60s request bound can interrupt a valid large upload.
            await sandbox.files.write(path, data, request_timeout=0)  # pyright: ignore[reportUnknownMemberType]

    async def stat(self, path: str) -> FileEntry:
        sandbox = await self.get_sandbox()
        async with self._sdk_errors(sandbox.sandbox_id, f'Could not stat {path!r}', path):
            return await _file_entry(sandbox, await sandbox.files.get_info(path))

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        sandbox = await self.get_sandbox()
        async with self._sdk_errors(sandbox.sandbox_id, f'Could not list {path!r}', path):
            # `depth=1` is E2B's non-recursive listing, as the protocol asks.
            entries = await sandbox.files.list(path, depth=1)
            resolved: list[FileEntry | None] = [None] * len(entries)
            failures: list[Exception | None] = [None] * len(entries)
            limit = anyio.Semaphore(16)

            async def resolve(index: int, entry: e2b.EntryInfo) -> None:
                # Symlinks need an envd round trip; cap concurrent lookups without
                # changing the order returned by the listing.
                async with limit:
                    try:
                        resolved[index] = await _file_entry(sandbox, entry)
                    except Exception as error:
                        # Keep each entry's path and the original SDK exception out of
                        # ExceptionGroup; transient failures must retain their identity.
                        failures[index] = error

            async with anyio.create_task_group() as group:
                for index, entry in enumerate(entries):
                    group.start_soon(resolve, index, entry)
            for index, error in enumerate(failures):
                if error is not None:
                    translated = await self._translate(
                        error, f'Could not list {path!r}', entries[index].path, sandbox.sandbox_id
                    )
                    if translated is error:
                        raise error
                    raise translated from error
            return [entry for entry in resolved if entry is not None]

    async def make_dir(self, path: str) -> None:
        sandbox = await self.get_sandbox()
        async with self._sdk_errors(sandbox.sandbox_id, f'Could not create directory {path!r}', path):
            await sandbox.files.make_dir(path)

    async def remove(self, path: str) -> None:
        sandbox = await self.get_sandbox()
        async with self._sdk_errors(sandbox.sandbox_id, f'Could not remove {path!r}', path):
            # envd removes with `os.RemoveAll`, which succeeds on a missing path; the protocol
            # reports that as `FileNotFoundError`.
            if not await sandbox.files.exists(path):
                raise e2b.FileNotFoundException(path)
            await sandbox.files.remove(path)

    async def exists(self, path: str) -> bool:
        sandbox = await self.get_sandbox()
        async with self._sdk_errors(sandbox.sandbox_id, f'Could not check {path!r}', path):
            return await sandbox.files.exists(path)

    async def _create(self) -> e2b.AsyncSandbox:
        """Provision a fresh E2B sandbox.

        E2B may create the sandbox before its response arrives. The asyncio acquisition task
        survives caller cancellation, and the Trio path shields this call; `_CREATE_TIMEOUT`
        bounds the SDK request, though a response lost after remote creation cannot be recovered. A request E2B refuses (an
        unknown template, a lifetime over the plan's limit) is `WorkspaceUnavailableError`:
        retrying it cannot succeed.
        """
        with anyio.move_on_after(_CREATE_TIMEOUT, shield=True):
            async with self._sdk_errors(None, 'Could not start E2B sandbox'):
                try:
                    return await e2b.AsyncSandbox.create(
                        template=self._template,
                        timeout=self._sandbox_timeout,
                        envs=dict(self._env) if self._env is not None else None,
                        secure=True,
                        allow_internet_access=self._allow_internet_access,
                        # Pause at the end of the lifetime instead of killing, so the files survive.
                        lifecycle={'on_timeout': 'pause'},
                    )
                except e2b.SandboxException as error:
                    refused = error.status_code is not None and 400 <= error.status_code < 500
                    if not refused or isinstance(error, e2b.RateLimitException):
                        raise
                    raise WorkspaceUnavailableError(_refused_message('Could not start E2B sandbox', error)) from error
        # A transient failure like any unreachable service: it propagates for durable engines to
        # retry, with a message that says what did not answer.
        raise TimeoutError(
            f'E2B sandbox creation did not complete within {_CREATE_TIMEOUT}s; '
            'the E2B control plane may be unreachable.'
        )

    async def _attach(self, sandbox_id: str) -> e2b.AsyncSandbox:
        """Attach to an E2B sandbox that already exists, without taking over its lifecycle.

        E2B resumes a paused sandbox on connect, so attaching to one that was paused restarts
        it; a sandbox that is gone raises `WorkspaceUnavailableError` rather than resolving to
        a dead environment. Nothing is recreated in its place; a run that expected files there
        must be told they are gone, not handed an empty workspace. A lifetime over the plan's
        limit is refused like on create, as `WorkspaceUnavailableError`.
        """
        context = f'Could not connect to E2B sandbox {sandbox_id!r}'
        async with self._sdk_errors(sandbox_id, context):
            try:
                # Without `timeout`, a resumed sandbox gets E2B's 300 seconds; a running one keeps
                # the longer of its current and the given lifetime.
                return await e2b.AsyncSandbox.connect(sandbox_id, timeout=self._sandbox_timeout)
            except e2b.SandboxException as error:
                if type(error) is not e2b.SandboxException or not _is_lifetime_refusal(error):
                    raise
                raise WorkspaceUnavailableError(_refused_message(context, error)) from error

    async def working_dir(self) -> str:
        """The canonical absolute directory commands start in."""
        if self._resolved_working_dir is None:
            result = await self.run(['pwd', '-P'], timeout=_INTERNAL_EXEC_TIMEOUT)
            printed = result.stdout.removesuffix('\n')
            if result.exit_code != 0 or not posixpath.isabs(printed):
                sandbox = await self.get_sandbox()
                raise WorkspaceError(
                    f'Could not determine the working directory of E2B sandbox {sandbox.sandbox_id}: '
                    f'`pwd -P` exited {result.exit_code} and printed {result.stdout!r}. Use absolute paths.'
                )
            self._resolved_working_dir = printed
        return self._resolved_working_dir

    async def _has_setsid(self, sandbox: e2b.AsyncSandbox) -> bool:
        async with self._setsid_lock:
            if self._setsid is None:
                async with self._sdk_errors(sandbox.sandbox_id, 'Could not probe E2B command launcher'):
                    probe = await sandbox.commands.run('command -v setsid >/dev/null 2>&1', background=True, timeout=10)
                    try:
                        result = await probe.wait()
                        self._setsid = result.exit_code == 0
                    except e2b.CommandExitException:
                        self._setsid = False
            return self._setsid

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run a command, killing it on timeout, cancellation, or a failed result read."""
        # `commands.run` takes only a string, which E2B hands to `/bin/bash -l -c`; `shlex.join`
        # keeps each argv element one word, and a `shell=True` string runs under `/bin/sh -c`.
        line = shlex.join(command_argv(command, shell))
        cwd = absolute_path('cwd', cwd)
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
        # Acquiring the sandbox has its own bound; the timeout is the command's alone.
        sandbox = await self.get_sandbox()
        handle: e2b.AsyncCommandHandle | None = None
        result: e2b.CommandResult | None = None
        # The token is allocated before the start RPC: a lost ACK must not make the
        # remote group undiscoverable. setsid isolates this invocation from other jobs.
        pgid_file = f'/tmp/pydantic-e2b-pgid-{uuid.uuid4().hex}'
        claim = f'{pgid_file}.claim'
        # The launcher and canceller race on one atomic mkdir. A cancelled late launcher
        # exits before user code; if launch wins, the canceller waits for its registration.
        # Keep a cancellation claim when the start ACK is lost so a late RPC stays fenced.
        launch = f'mkdir {claim} 2>/dev/null || exit 143; echo "$$:$(ps -o lstart= -p $$)" > {pgid_file}; exec {line}'
        # Probe only custom/attached images: default E2B images include util-linux.
        isolated = await self._has_setsid(sandbox) if self._probe_setsid else True
        launch = f'{"setsid " if isolated else ""}sh -c {shlex.quote(launch)}'

        async def stop() -> None:
            # A new SDK command can run even when the original output stream is blocked.
            script = (
                f'if mkdir {claim} 2>/dev/null; then exit 0; fi; '
                f'i=0; while [ ! -s {pgid_file} ] && [ "$i" -lt 100 ]; do '
                'sleep 0.1; i=$((i+1)); done; '
                f'if [ -s {pgid_file} ]; then record=$(cat {pgid_file}); p=${{record%%:*}}; '
                # Match creation time before signalling: a reused PID belongs to another run.
                '[ "$(ps -o lstart= -p "$p")" = "${record#*:}" ] || exit 0; '
                + ('[ "$(ps -o pgid= -p "$p" 2>/dev/null | tr -d " ")" = "$p" ] || exit 0; ' if isolated else '')
                + f'kill -TERM {"-" if isolated else ""}"$p" 2>/dev/null || true; sleep 0.1; '
                + f'kill -KILL {"-" if isolated else ""}"$p" 2>/dev/null || true; fi'
            )
            try:
                stopper = await sandbox.commands.run(
                    f'sh -c {shlex.quote(script)}', background=True, timeout=_SDK_STREAM_UNBOUNDED
                )
                await stopper.wait()
            except Exception:
                pass  # A failed control-plane stop must not mask the original failure.

        try:
            with anyio.move_on_after(timeout):
                # A lost start ACK must not hide a launched command from the side-channel stop.
                handle = await sandbox.commands.run(
                    launch,
                    background=True,
                    # Explicit caller settings take precedence over the portable UTF-8 default.
                    envs={'LC_ALL': 'C.UTF-8', **(self._env or {}), **(env or {})},
                    cwd=cwd if cwd is not None else self._working_dir,
                    timeout=_SDK_STREAM_UNBOUNDED,
                )
                assert handle is not None
                result = await handle.wait()
            if result is None:
                assert timeout is not None
                raise WorkspaceTimeoutError(
                    f'Command timed out after {timeout:g} seconds',
                    stdout=handle.stdout if handle is not None else '',
                    stderr=handle.stderr if handle is not None else '',
                )
            return CommandResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)
        except e2b.CommandExitException as error:
            return CommandResult(exit_code=error.exit_code, stdout=error.stdout, stderr=error.stderr)
        except BaseException as error:
            # A child task shields the side-channel stop from repeated task cancellation.
            # Allow the winning launcher to publish its PID before bounding a lost start ACK.
            await stop_shielded(stop, grace=12)
            if isinstance(error, Exception):
                context = (
                    'Command could not run in the E2B sandbox'
                    if handle is None
                    else 'Could not read the command result (the command may still be running)'
                )
                translated = await self._translate(error, context, None, sandbox.sandbox_id)
                if translated is not error:
                    raise translated from error
            raise
        finally:
            # Only a completed start and result proves no late RPC can use this claim.
            if result is not None:
                with anyio.move_on_after(0.5, shield=True):
                    try:
                        await sandbox.files.remove(pgid_file)
                        await sandbox.files.remove(claim)
                    except Exception:
                        pass


async def _is_running(sandbox: e2b.AsyncSandbox) -> bool:
    """Ask E2B's health probe whether the sandbox runs; a probe that fails counts as running.

    Probing only after an error keeps the extra round trip off successful operations, and a
    failed probe leaves the original error to propagate rather than aborting the run on a guess.
    """
    try:
        return await sandbox.is_running()
    except Exception:
        return True
