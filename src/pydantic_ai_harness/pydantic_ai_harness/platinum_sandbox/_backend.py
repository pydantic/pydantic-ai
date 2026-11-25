"""A Platinum sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

Platinum's exec API runs one process per request, and one request can wait about 85 seconds. A command whose
timeout is at most 80 seconds is one exec request, and the sandbox kills the command's process group at its
deadline. A longer timeout, or none, runs the command as a managed process whose output is streamed until it
exits. File operations use Platinum's filesystem API.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import math
import os
import posixpath
import re
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, NoReturn, TypeVar

import anyio
import anyio.lowlevel

from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    SupportsRealpath,
    Workspace,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._typeguards import is_mapping, is_object_list
from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    check_env,
    check_provider,
    check_timeout,
    command_argv,
    require_asyncio,
    run_to_completion,
    running_on_asyncio,
    safe_credential_reason,
    stop_shielded,
)

if TYPE_CHECKING:
    from platinum import ProcessEvent
    from platinum.aio import AsyncPlatinum, AsyncProcess, AsyncSandbox

    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('PlatinumSandboxBackend',)

try:
    import platinum
except ImportError as error:
    raise ImportError('Install `pydantic-ai-harness[platinum]` to use PlatinumSandbox.') from error

_T = TypeVar('_T')

_PROVIDER = 'platinum'
DEFAULT_WORKING_DIR = '/workspace'
"""Where commands start, and relative paths resolve, unless `working_dir` says otherwise."""

_ASYNCIO_ONLY = 'Platinum needs the asyncio event loop: the Platinum SDK runs its requests on asyncio.'
_AUTH_MESSAGE = 'Platinum refused the API key. Set PT_TOKEN, or pass a configured `AsyncPlatinum` as `client=`.'
_MISSING_CREDENTIALS_MESSAGE = (
    'No Platinum API key found. Set PT_TOKEN, or pass a configured `AsyncPlatinum` as `client=`.'
)
_OUTPUT_LIMIT_MESSAGE = (
    'The command output exceeded the 16 MiB limit; redirect it to a file and read part of it instead'
)

# The longest timeout one exec request serves; a longer one, or none, runs as a managed process.
_EXEC_MAX_TIMEOUT = 80.0
# Client-side slack past a managed process's deadline, which the sandbox enforces first.
_PROCESS_GRACE = 15.0
# The longest one output stream connection is asked to stay open; the SDK reconnects past it.
_STREAM_MAX_SECONDS = 1800
# The backend's own short commands: `pwd -P`, realpath, the FIFO probe, joining parts, listings.
_INTERNAL_TIMEOUT = 30.0
# How long stopping a cancelled or timed-out command may take.
_STOP_GRACE = 10.0
# How long a created or started sandbox may take to be running.
_START_TIMEOUT = 120.0
# How long one filesystem request may take.
_FILE_TIMEOUT = 600.0
# A read that has not answered by then may be a FIFO, which the sandbox would wait on for a writer.
_FIFO_PROBE_AFTER = 2.0
# Symlinks of one listing stat-ed at once, to follow them.
_LIST_STAT_CONCURRENCY = 16

# Exec answers `413 output_too_large` past 16 MiB of output, and managed processes keep the same cap.
_MAX_OUTPUT_BYTES = 16 * 1024 * 1024
_OUTPUT_PREVIEW_CHARS = 64 * 1024
# A managed process keeps the last 256 KiB of each stream for a reader that falls behind.
_PROCESS_RING_BYTES = 256 * 1024
# A larger write goes in parts of this size, well below Platinum's per-request limit; a `413 payload_too_large`
# that states a lower limit shrinks them.
_UPLOAD_PART_BYTES = 8 * 1024 * 1024
# A file one read can't return (`413 file_too_large`) is copied out in parts of this size.
_READ_PART_MIB = 64
# Upload parts, stop claims, read parts and oversized listings go here.
_SCRATCH_DIR = '/tmp'
# `/files/list` describes entries with `lstat`, so a symlink reads as a non-directory with mode 0777.
_SYMLINK_MODE = 0o777

_STARTING_STATES = frozenset({'pending', 'provisioning', 'starting', 'resuming', 'restoring', 'unarchiving'})
_REF_ID = re.compile(r'[A-Za-z0-9_-]{1,128}')

# Every command runs as `sh -c _LAUNCH sh <working dir> <stop claim> <argv...>`. Platinum's exec has no working
# directory, so the script enters it, creating it if missing, then runs the argv with stdin at EOF. A program named
# like an option (`-x`) is resolved to its path first, because `exec` in dash and bash would read it as one. With a
# claim (an exec request), the script records its pid, which is also the command's process group, so a cancelled
# call can kill the group. The claim is a directory made with one `mkdir`, so a stop that arrives before the launch
# takes it first and the command never runs (exit 143).
_NO_WORKING_DIR = 'pydantic-ai: cannot enter or create the working directory'
_NO_WORKING_DIR_EXIT = 125
_LAUNCH = (
    'd=$1 c=$2; shift 2; '
    'if [ -n "$d" ]; then { cd -- "$d" 2>/dev/null || { mkdir -p -- "$d" && cd -- "$d"; }; } '
    f'|| {{ printf "%s\\n" "{_NO_WORKING_DIR}" >&2; exit {_NO_WORKING_DIR_EXIT}; }}; fi; '
    'case $1 in '
    '-*/*) p=./$1 ;; '
    '-*) p=$(command -v -- "$1") || { printf "sh: %s: not found\\n" "$1" >&2; exit 127; } ;; '
    '*) p=$1 ;; '
    'esac; shift; '
    'if [ -z "$c" ]; then exec "$p" "$@" </dev/null; fi; '
    'if mkdir -- "$c" 2>/dev/null; then echo $$ > "$c/pid"; '
    'elif [ -d "$c" ]; then exit 143; '
    'else exec "$p" "$@" </dev/null; fi; '
    '"$p" "$@" </dev/null; s=$?; rm -rf -- "$c"; exit "$s"'
)
# Stops the command behind a claim: takes the claim first (the command then never starts), or kills the process
# group the launch recorded. dash's `kill` takes no `--`.
_STOP = (
    'c=$1; if mkdir -- "$c" 2>/dev/null; then exit 0; fi; '
    'i=0; while [ ! -s "$c/pid" ] && [ "$i" -lt 40 ]; do sleep 0.05; i=$((i+1)); done; '
    'p=$(cat -- "$c/pid" 2>/dev/null); rm -rf -- "$c"; '
    'case $p in ""|*[!0-9]*) exit 0 ;; esac; '
    'kill -9 "-$p" 2>/dev/null; exit 0'
)
# Joins the parts of a write too large for one request. 120: a parent is not a directory; 121: the target is one.
_JOIN_PARTS = (
    't=$1 d=$2; [ -d "$t" ] && exit 121; '
    'mkdir -p -- "$(dirname -- "$t")" 2>/dev/null || exit 120; '
    'cat -- "$d"/part-* > "$t"'
)
# Copies one part of a file too large for one read: `<part>` gets `<count>` MiB from MiB `<skip>` on.
_COPY_PART = 'dd if="$1" of="$2" bs=1048576 skip="$3" count="$4" 2>/dev/null'
# A listing too large for one reply, as NUL-separated `<d|f><path>` records in a file. 102: no such path; 120: not
# a directory.
_LIST_TO_FILE = (
    'p=$1 f=$2; '
    'if ! [ -d "$p" ]; then [ -e "$p" ] && exit 120; exit 102; fi; '
    'find -H "$p" -mindepth 1 -maxdepth 1 -exec sh -c '
    '\'for e do if [ -d "$e" ]; then printf "d%s\\000" "$e"; else printf "f%s\\000" "$e"; fi; done\' sh {} + > "$f"'
)

logger = logging.getLogger(__name__)


def _new_client() -> AsyncPlatinum:
    """An `AsyncPlatinum` from `PT_TOKEN` and `PT_API_URL`. Building one does no I/O."""
    try:
        return platinum.AsyncPlatinum()
    except ValueError as error:
        raise WorkspaceUnavailableError(_MISSING_CREDENTIALS_MESSAGE) from error


async def _close_quietly(client: AsyncPlatinum) -> None:
    """Close an API client, bounded and shielded; a failure is logged, not raised."""
    with anyio.move_on_after(_INTERNAL_TIMEOUT, shield=True):
        try:
            await client.aclose()
        except Exception as error:
            logger.warning('Could not close the Platinum API client: %r', error)


def check_ref(ref: WorkspaceRef) -> None:
    """Raise `ValueError` unless `ref` names a Platinum sandbox by an id that fits in one URL segment.

    A ref can come from stored or untrusted message history.
    """
    check_provider(ref, _PROVIDER)
    if _REF_ID.fullmatch(ref.id) is None:
        raise ValueError(f'invalid Platinum workspace ref id {ref.id!r}')


async def destroy_sandbox(client: AsyncPlatinum | None, sandbox_id: str) -> None:
    """Delete a sandbox by id without starting it. One that no longer exists returns quietly.

    Rejected credentials raise `WorkspaceUnavailableError`. Without `client`, one is opened from the environment
    and closed afterwards.

    Raises:
        UserError: The event loop is not asyncio.
    """
    require_asyncio(_ASYNCIO_ONLY)
    owned: AsyncPlatinum | None = None
    if client is None:
        client = owned = _new_client()
    try:
        await client.sandboxes.delete(sandbox_id)
    except platinum.NotFoundError:
        return
    except (platinum.AuthenticationError, platinum.ForbiddenError) as error:
        raise WorkspaceUnavailableError(f'{safe_credential_reason(error)}. {_AUTH_MESSAGE}') from error
    finally:
        if owned is not None:
            await _close_quietly(owned)


def _is_gone(state: object) -> bool:
    return isinstance(state, str) and (state in ('deleted', 'lost') or state.startswith('failed'))


def _gone_message(sandbox_id: str, state: object = None) -> str:
    why = f'it is {state}' if isinstance(state, str) and state else 'it was deleted, or never existed in this org'
    return (
        f'The Platinum sandbox {sandbox_id!r} is no longer available: {why}. '
        "Pass `workspace='new'` to start a fresh sandbox."
    )


def _body_field(error: platinum.PlatinumError, key: str) -> object:
    body: object = error.body
    return body.get(key) if is_mapping(body) else None


def _error_text(error: platinum.PlatinumError) -> str:
    detail = _body_field(error, 'error')
    return (detail if isinstance(detail, str) and detail else str(error)).lower()


def _raise_acquire_error(error: platinum.PlatinumError, context: str) -> NoReturn:
    """Raise for a create, attach or start failure.

    Refused credentials and requests end the run; transient failures (429, 5xx, transport) propagate unchanged for
    a durable engine to retry.
    """
    if isinstance(error, (platinum.AuthenticationError, platinum.ForbiddenError)):
        raise WorkspaceUnavailableError(f'{context}: {safe_credential_reason(error)}. {_AUTH_MESSAGE}') from error
    if 400 <= error.status < 500 and error.status not in (408, 409, 429):
        raise WorkspaceUnavailableError(f'{context}: {error}') from error
    raise error


def _path_error(error: platinum.PlatinumError, path: str) -> OSError | None:
    """The built-in error a filesystem refusal describes, or `None` if it describes none.

    The API names some failures with a `code` and passes the sandbox's `strerror` text through for the rest.
    """
    code = error.code or ''
    text = _error_text(error)

    def oserror(kind: type[OSError], number: int) -> OSError:
        return kind(number, f'{os.strerror(number)} in the Platinum sandbox', path)

    if code == 'not_a_directory' or 'not a directory' in text:
        return oserror(NotADirectoryError, errno.ENOTDIR)
    if code == 'path_not_found' or 'no such file or directory' in text:
        return oserror(FileNotFoundError, errno.ENOENT)
    if code == 'not_a_file' or 'is a directory' in text:
        return oserror(IsADirectoryError, errno.EISDIR)
    if code == 'permission_denied' or 'permission denied' in text or 'operation not permitted' in text:
        return oserror(PermissionError, errno.EACCES)
    if 'file exists' in text:
        return oserror(FileExistsError, errno.EEXIST)
    if 'too many levels of symbolic links' in text:
        return oserror(OSError, errno.ELOOP)
    if code == 'disk_full':
        return oserror(OSError, errno.ENOSPC)
    if code == 'directory_not_empty':
        return oserror(OSError, errno.ENOTEMPTY)
    return None


def _translate(error: platinum.PlatinumError, sandbox_id: str, path: str | None) -> Exception:
    """Map an operation's SDK error onto the failures the workspace protocol promises.

    Returns `error` itself for what must propagate unchanged: transport failures, rate limits and server errors,
    which durable engines retry as transient.
    """
    if error.code == 'sandbox_not_found':
        return WorkspaceUnavailableError(_gone_message(sandbox_id))
    if isinstance(error, (platinum.AuthenticationError, platinum.ForbiddenError)):
        return WorkspaceUnavailableError(f'{safe_credential_reason(error)}. {_AUTH_MESSAGE}')
    if path is not None and _names_path(error, path):
        if (path_error := _path_error(error, path)) is not None:
            return path_error
        if error.status == 400:
            return ValueError(f'{path!r}: {error}')
    if 400 <= error.status < 500 and error.status not in (408, 429):
        return WorkspaceError(f'Platinum sandbox {sandbox_id}: {error}')
    return error


def _names_path(error: platinum.PlatinumError, path: str) -> bool:
    """Whether `error` is about `path` rather than the request: any refusal, or a server error with the system call's text.

    A write the sandbox refuses, such as one without permission, is a `500` whose text names the path.
    """
    return 400 <= error.status < 500 or (error.status >= 500 and path.lower() in _error_text(error))


def _write_limit(error: platinum.PlatinumError) -> int | None:
    """The per-write byte limit a `413 payload_too_large` states, if any."""
    limit = _body_field(error, 'max_bytes') if error.status == 413 else None
    return limit if type(limit) is int and limit > 0 else None


def _refused_before_running(error: platinum.PlatinumError) -> bool:
    """Whether the command never started: a 4xx other than a request timeout, or a connection that never opened."""
    if isinstance(error, (platinum.PlatinumConnectionError, platinum.PlatinumTimeoutError)):
        return type(error.__cause__).__name__ in ('ConnectError', 'ConnectTimeout')
    return 400 <= error.status < 500 and error.status != 408


def _check_path(path: str) -> str:
    """`path` normalized as text, as `Workspace.resolve` does; the filesystem API refuses `..` segments."""
    if '\0' in path:
        raise ValueError('path must not contain NUL bytes')
    if not path.startswith('/'):
        raise ValueError(f'path must be absolute, got {path!r}')
    return '/' + posixpath.normpath(path).lstrip('/')  # `normpath` keeps a leading `//`


def _decode(data: bytes | bytearray) -> str:
    return bytes(data).decode(errors='replace')


def _with_env(argv: list[str], env: Mapping[str, str]) -> list[str]:
    """`argv` run under `env`; exec takes no environment of its own. `--` ends `env`'s options."""
    if not env:
        return argv
    return ['env', '--', *(f'{name}={value}' for name, value in env.items()), *argv]


def _scratch(kind: str) -> str:
    return f'{_SCRATCH_DIR}/.pydantic-ai-{kind}-{uuid.uuid4().hex}'


class _Capture:
    """The stdout and stderr of a managed process, capped together at `_MAX_OUTPUT_BYTES`."""

    def __init__(self) -> None:
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.dropped = 0

    def add(self, event: ProcessEvent) -> bool:
        """Keep a chunk; `False` once the output is over the cap."""
        self.dropped += event.dropped
        data = event.data if isinstance(event.data, bytes) else event.data.encode()
        (self.stdout if event.type == 'stdout' else self.stderr).extend(data)
        return len(self.stdout) + len(self.stderr) <= _MAX_OUTPUT_BYTES

    def texts(self) -> tuple[str, str]:
        return _decode(self.stdout), _decode(self.stderr)

    def previews(self) -> tuple[str, str]:
        stdout, stderr = self.texts()
        return stdout[:_OUTPUT_PREVIEW_CHARS], stderr[:_OUTPUT_PREVIEW_CHARS]


class _BoundedCommands:
    """The backend's commands with a deadline, for core's shell `realpath`.

    Without one, each would be a managed process instead of one exec request.
    """

    def __init__(self, backend: PlatinumSandboxBackend) -> None:
        self._backend = backend

    @property
    def ref(self) -> WorkspaceRef | None:  # pragma: no cover - a protocol member core's shell realpath doesn't use
        return self._backend.ref

    async def working_dir(self) -> str:  # pragma: no cover - the backend passes realpath an absolute path
        return await self._backend.working_dir()

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        return await self._backend.run(command, shell=shell, env=env, timeout=timeout or _INTERNAL_TIMEOUT)


class PlatinumSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem, SupportsRealpath):
    """A [Platinum](https://www.platinum.dev) sandbox as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands and file operations run inside a Platinum sandbox, a Cloud Hypervisor microVM, so the host is never
    exposed.

    Building one does no I/O. The first operation creates a sandbox, or attaches to the one `ref` names, and the
    `platinum.AsyncSandbox` is available through `get_sandbox()`. The backend does not stop or delete the sandbox;
    that is the application's job.

    A command's complete output is returned once it exits. A shell string runs under `/bin/sh -c`; a timeout or
    cancellation stops the command. A sandbox that stopped while idle is started again, and the operation that
    found it stopped runs once more.

    A deleted sandbox, rejected credentials, and a refused create (an unknown template, too few credits) raise
    `WorkspaceUnavailableError`. Path failures raise the built-in errors, such as `FileNotFoundError`.

    Args:
        sandbox: A `platinum.AsyncSandbox` you already have. Whoever created it owns deleting it.
        client: A `platinum.AsyncPlatinum` API client to create or attach with. The caller owns closing it.
            Without one, the backend opens its own from `PT_TOKEN` and `PT_API_URL` on first use and closes it in
            `aclose()`.
        ref: Identity of an existing sandbox to attach to on first use.
        template: Platinum template a newly created sandbox boots from; Platinum's default, `pt-base`, when `None`.
        cpu: vCPUs of a newly created sandbox; the template's default when `None`.
        ram_mb: Memory of a newly created sandbox, in MiB; the template's default when `None`.
        disk_gb: Disk of a newly created sandbox, in GB; the template's default when `None`.
        auto_stop_minutes: Idle minutes before Platinum stops a newly created sandbox; `None` keeps Platinum's
            default, `0`, which never stops it.
        allow_internet_access: Whether a newly created sandbox may reach the internet; without it, the hosts of
            its `secrets` stay reachable.
        secrets: Names of organization secrets a newly created sandbox may spend. Platinum's egress proxy writes
            each one into the request headers of its allowed hosts, so the sandbox's commands use it without
            holding it.
        working_dir: Absolute directory commands start in and relative paths resolve against, created if it is
            missing; `None` keeps the template's own.
        env: Environment variables every command gets; a command's own `env` is layered on top. They are also set
            on a newly created sandbox.
    """

    def __init__(
        self,
        *,
        sandbox: AsyncSandbox | None = None,
        client: AsyncPlatinum | None = None,
        ref: WorkspaceRef | None = None,
        template: str | None = None,
        cpu: int | None = None,
        ram_mb: int | None = None,
        disk_gb: int | None = None,
        auto_stop_minutes: int | None = None,
        allow_internet_access: bool = True,
        secrets: Sequence[str] | None = None,
        working_dir: str | None = DEFAULT_WORKING_DIR,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if sandbox is not None and ref is not None:
            raise ValueError('pass either `sandbox` or `ref`, not both')
        if ref is not None:
            check_ref(ref)
        if isinstance(secrets, str):
            raise TypeError(f'`secrets` takes a list of secret names, such as [{secrets!r}], not a string')
        check_env(env)
        self._ref = ref if sandbox is None else WorkspaceRef(provider=_PROVIDER, id=sandbox.id)
        self._sandbox = sandbox
        self._client = client
        self._owns_client = client is None and sandbox is None
        self._template = template
        self._cpu = cpu
        self._ram_mb = ram_mb
        self._disk_gb = disk_gb
        self._auto_stop_minutes = auto_stop_minutes
        self._allow_internet_access = allow_internet_access
        self._secrets = list(secrets) if secrets is not None else None
        self._env = dict(env or {})
        working_dir = absolute_path('working_dir', working_dir)
        self._working_dir = posixpath.normpath(working_dir) if working_dir is not None else None
        self._create_name: str | None = None
        # `working_dir()` must return a canonical absolute path, resolved once with `pwd -P`.
        self._resolved_working_dir: str | None = None
        self._lock = anyio.Lock()
        # Set while nothing will call `aclose()`: the owned client then closes when the last operation in flight
        # ends.
        self._close_after_operation = False
        self._operations = 0

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the sandbox, or `None` before one has been created."""
        return self._ref

    async def get_sandbox(self) -> AsyncSandbox:
        """Return the `platinum.AsyncSandbox`, creating or attaching to it on first use.

        Attaching by `ref` to a sandbox that no longer exists raises `WorkspaceUnavailableError`; it does not
        create a replacement.
        """
        require_asyncio(_ASYNCIO_ONLY)
        async with self._lock:
            if (sandbox := self._sandbox) is None:
                sandbox = await self._acquire_locked()
        await anyio.lowlevel.checkpoint_if_cancelled()
        return sandbox

    async def _acquire_locked(self) -> AsyncSandbox:
        """Create or attach to the sandbox; the caller holds `_lock`."""
        if self._client is None:
            self._client = _new_client()
        client = self._client
        if self._ref is not None:
            sandbox = await self._attach(client, self._ref.id)
            self._sandbox = sandbox
            return sandbox

        async def create() -> AsyncSandbox:
            created = await self._create(client)
            self._sandbox = created
            self._ref = WorkspaceRef(provider=_PROVIDER, id=created.id)
            # Logged at creation, so the id can be found whatever ends the run before it is stored.
            logger.info('Created Platinum sandbox %s (%s)', created.id, self._create_name)
            return created

        # Creation finishes even if the caller is cancelled, so a billed sandbox is never left unrecorded.
        return await run_to_completion(create)

    async def _create(self, client: AsyncPlatinum) -> AsyncSandbox:
        """Create a sandbox under a name kept across retries, so a lost reply is found rather than duplicated."""
        if self._create_name is None:
            self._create_name = f'pydantic-ai-{uuid.uuid4().hex[:12]}'
        name = self._create_name
        try:
            return await client.sandboxes.create(
                name=name,
                template=self._template,
                cpu=self._cpu,
                ram_mb=self._ram_mb,
                disk_gb=self._disk_gb,
                env=self._env or None,
                auto_stop_minutes=self._auto_stop_minutes,
                allow_internet_access=None if self._allow_internet_access else False,
                secrets=list(self._secrets) if self._secrets else None,
                wait_for_running=True,
                wait_timeout_ms=int(_START_TIMEOUT * 1000),
            )
        except platinum.PlatinumError as error:
            if (created := _body_field(error, 'id')) and isinstance(created, str) and error.status == 0:
                # Created, but it was not running in time: record it so it can be found and deleted.
                self._ref = WorkspaceRef(provider=_PROVIDER, id=created)
                if error.code == 'terminal_state':
                    raise WorkspaceUnavailableError(
                        f'The Platinum sandbox {created} failed to start ({error}). '
                        'Delete it with `PlatinumSandbox().destroy(ref)`.'
                    ) from error
                raise TimeoutError(
                    f'The Platinum sandbox {created} was not running after {_START_TIMEOUT:g}s.'
                ) from error
            if isinstance(error, (platinum.PlatinumTimeoutError, platinum.PlatinumConnectionError)) or (
                error.code == 'name_taken'
            ):
                # The reply was lost, or an earlier attempt's sandbox holds the name.
                if (found := await self._find_created(client, name)) is not None:
                    return found
            _raise_acquire_error(error, 'Could not create a Platinum sandbox')

    async def _find_created(self, client: AsyncPlatinum, name: str) -> AsyncSandbox | None:
        """The sandbox an earlier create under `name` made, once it is running; `None` if there is none."""
        try:
            page = await client.sandboxes.list(limit=100)
        except platinum.PlatinumError:
            return None
        rows: object = page.get('rows')
        for row in rows if is_object_list(rows) else ():
            if is_mapping(row) and row.get('name') == name and isinstance(sandbox_id := row.get('id'), str):
                if not _is_gone(row.get('state')):
                    logger.info('Found Platinum sandbox %s (%s) after its create reply was lost', sandbox_id, name)
                    self._ref = WorkspaceRef(provider=_PROVIDER, id=sandbox_id)
                    return await self._attach(client, sandbox_id)
        return None

    async def _attach(self, client: AsyncPlatinum, sandbox_id: str) -> AsyncSandbox:
        """Attach to a sandbox that exists, starting it if it is stopped or archived."""
        try:
            sandbox = await client.sandboxes.connect(sandbox_id)
            state: object = (await sandbox.refresh()).get('state')
        except platinum.NotFoundError as error:
            raise WorkspaceUnavailableError(_gone_message(sandbox_id)) from error
        except platinum.PlatinumError as error:
            _raise_acquire_error(error, f'Could not attach to Platinum sandbox {sandbox_id!r}')
        if _is_gone(state):
            raise WorkspaceUnavailableError(_gone_message(sandbox_id, state))
        await self._ensure_running(sandbox, state)
        return sandbox

    async def _ensure_running(self, sandbox: AsyncSandbox, state: object) -> None:
        if state == 'running':
            return
        try:
            if state in _STARTING_STATES and await _became_running(sandbox):
                return
            logger.info('Platinum sandbox %s is %s; starting it', sandbox.id, state)
            await sandbox.start(wait_for_running=True, wait_timeout_ms=int(_START_TIMEOUT * 1000))
        except platinum.PlatinumError as error:
            if error.code == 'terminal_state':
                raise WorkspaceUnavailableError(_gone_message(sandbox.id, _body_field(error, 'state'))) from error
            if isinstance(error, platinum.PlatinumTimeoutError) and error.status == 0:
                raise TimeoutError(
                    f'The Platinum sandbox {sandbox.id} was not running after {_START_TIMEOUT:g}s.'
                ) from error
            _raise_acquire_error(error, f'Could not start Platinum sandbox {sandbox.id!r}')

    async def aclose(self) -> None:
        """Close the `AsyncPlatinum` API client this backend opened for itself.

        A client passed as `client=` is left alone, and the sandbox keeps running. A later operation opens a new
        client and attaches by `ref`. `PlatinumSandbox` calls this when the run that used the backend ends.
        """
        if self._client is None or not self._owns_client or not running_on_asyncio():
            return
        if self._operations:
            # An operation in flight, such as a cancelled command being stopped, still needs the client: the last
            # one to finish closes it.
            self._close_after_operation = True
            return

        async def close() -> None:
            # Under the lock, so a sandbox still being created is recorded before its client closes.
            async with self._lock:
                client = self._client
                self._detach_client()
            if client is not None:  # another `aclose()` may have closed it meanwhile
                await _close_quietly(client)

        await run_to_completion(close)

    def _detach_client(self) -> None:
        # The sandbox handle belongs to the client; the next operation attaches by ref with a new one.
        self._client, self._sandbox = None, None

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[None]:
        self._operations += 1
        try:
            yield
        finally:
            self._operations -= 1
            client = self._client
            if self._operations == 0 and self._close_after_operation and self._owns_client and client is not None:
                # Detached before any await, so an operation starting meanwhile opens its own client.
                self._detach_client()
                await run_to_completion(lambda: _close_quietly(client))

    async def _call(self, operation: Callable[[AsyncSandbox], Awaitable[_T]], *, path: str | None = None) -> _T:
        """Run `operation` on the sandbox, mapping its errors.

        A sandbox that is not running (stopped while idle, or from elsewhere) is started, and the operation runs
        once more: that refusal means nothing ran.
        """
        async with self._operation():
            sandbox = await self.get_sandbox()
            async with self._mapped_errors(sandbox, path=path):
                try:
                    return await operation(sandbox)
                except platinum.ConflictError as error:
                    if error.code != 'sandbox_not_running':
                        raise
                await self._ensure_running(sandbox, 'stopped')
                return await operation(sandbox)

    @asynccontextmanager
    async def _mapped_errors(self, sandbox: AsyncSandbox, *, path: str | None = None) -> AsyncGenerator[None]:
        """Raise the protocol's typed failure for an SDK error, and let the backend's own failures through.

        A transport failure or server error is followed by one lookup, since the sandbox may have been deleted
        under the operation.
        """
        try:
            yield
        except platinum.PlatinumError as error:
            translated = _translate(error, sandbox.id, path)
            if translated is error and (error.status == 0 or error.status >= 500):
                if (gone := await self._probe(sandbox)) is not None:
                    raise gone from error
            if translated is error:
                raise
            raise translated from error

    async def _probe(self, sandbox: AsyncSandbox) -> WorkspaceUnavailableError | None:
        """`WorkspaceUnavailableError` if the sandbox can never run again; an inconclusive lookup counts as alive."""
        try:
            with anyio.fail_after(_INTERNAL_TIMEOUT):
                state: object = (await sandbox.refresh()).get('state')
        except platinum.NotFoundError:
            return WorkspaceUnavailableError(_gone_message(sandbox.id))
        except Exception:
            return None
        return WorkspaceUnavailableError(_gone_message(sandbox.id, state)) if _is_gone(state) else None

    async def working_dir(self) -> str:
        """Return the canonical directory commands start in, probed once."""
        async with self._operation():
            if self._resolved_working_dir is None:
                result = await self.run(['pwd', '-P'], timeout=_INTERNAL_TIMEOUT)
                printed = result.stdout.removesuffix('\n')
                if result.exit_code != 0 or not posixpath.isabs(printed):
                    raise WorkspaceError(
                        'Could not determine the working directory of the Platinum sandbox: '
                        f'`pwd -P` exited {result.exit_code}: {result.stderr.strip()}'
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
        argv = command_argv(command, shell)
        check_timeout(timeout)
        check_env(env)
        merged = {**self._env, **(env or {})}
        if timeout is not None and timeout <= _EXEC_MAX_TIMEOUT:
            exec_timeout = timeout
            return await self._call(lambda sandbox: self._exec(sandbox, _with_env(argv, merged), exec_timeout))
        return await self._call(lambda sandbox: self._process(sandbox, argv, merged, timeout))

    def _launch(self, argv: list[str], claim: str) -> list[str]:
        return ['sh', '-c', _LAUNCH, 'sh', self._working_dir or '', claim, *argv]

    def _result(self, exit_code: int, stdout: str, stderr: str) -> CommandResult:
        if exit_code == _NO_WORKING_DIR_EXIT and not stdout and stderr.endswith(_NO_WORKING_DIR + '\n'):
            raise WorkspaceError(
                f'working_dir {self._working_dir!r} cannot be entered or created in the Platinum sandbox: '
                f'{stderr.strip()}'
            )
        return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    async def _exec(self, sandbox: AsyncSandbox, argv: list[str], timeout: float) -> CommandResult:
        """One exec request; the sandbox kills the command's process group at `timeout`."""
        claim = _scratch('run')
        try:
            result = await sandbox.exec(self._launch(argv, claim), timeout_ms=max(1, math.ceil(timeout * 1000)))
        except platinum.PlatinumError as error:
            if error.code == 'exec_failed':
                raise await self._exec_failure(sandbox, error, claim, timeout) from error
            if error.status == 413 and error.code == 'output_too_large':
                # The command ran to the end; its output was over the reply limit and is gone.
                raise WorkspaceOutputLimitError(_OUTPUT_LIMIT_MESSAGE, limit=_MAX_OUTPUT_BYTES) from error
            if _refused_before_running(error):
                raise
            # No answer: the command may still be running.
            await stop_shielded(lambda: self._stop(sandbox, claim), grace=_STOP_GRACE)
            if isinstance(error, platinum.PlatinumTimeoutError):
                raise WorkspaceTimeoutError(
                    f'Command timed out after {timeout:g} seconds (the sandbox did not answer)'
                ) from error
            raise
        except BaseException:
            await stop_shielded(lambda: self._stop(sandbox, claim), grace=_STOP_GRACE)
            raise
        return self._result(result.exit_code, result.stdout, result.stderr)

    async def _exec_failure(
        self, sandbox: AsyncSandbox, error: platinum.PlatinumError, claim: str, timeout: float
    ) -> WorkspaceError:
        """The failure an exec request reported.

        A timeout carries the output printed before it; otherwise the sandbox could not start the command.
        """
        result = _body_field(error, 'result')
        if is_mapping(result) and result.get('timed_out') is True:
            await stop_shielded(lambda: self._stop(sandbox, claim), grace=_STOP_GRACE)  # removes the claim
            stdout, stderr = result.get('stdout'), result.get('stderr')
            return WorkspaceTimeoutError(
                f'Command timed out after {timeout:g} seconds',
                stdout=stdout if isinstance(stdout, str) else '',
                stderr=stderr if isinstance(stderr, str) else '',
            )
        return WorkspaceError(f'The Platinum sandbox could not start the command: {error}')

    async def _stop(self, sandbox: AsyncSandbox, claim: str) -> None:
        await sandbox.exec(['sh', '-c', _STOP, 'sh', claim], timeout_ms=int(_STOP_GRACE * 1000))

    async def _process(
        self, sandbox: AsyncSandbox, argv: list[str], env: Mapping[str, str], timeout: float | None
    ) -> CommandResult:
        """Run a managed process, streaming its output until it exits.

        The sandbox enforces `timeout`, with a client deadline behind it.
        """
        process = await sandbox.process.start(
            self._launch(argv, ''),
            env=dict(env) if env else None,
            timeout_ms=max(1, math.ceil(timeout * 1000)) if timeout is not None else None,
        )
        capture = _Capture()
        exit_event: ProcessEvent | None = None
        try:
            with anyio.move_on_after(None if timeout is None else timeout + _PROCESS_GRACE) as deadline:
                exit_event = await _follow(process, capture, timeout)
        except BaseException:
            # Cancelled, or the stream failed: don't leave the command running.
            await stop_shielded(process.kill, grace=_STOP_GRACE)
            raise
        if exit_event is None:
            await stop_shielded(process.kill, grace=_STOP_GRACE)
            if not deadline.cancelled_caught:
                stdout, stderr = capture.previews()
                raise WorkspaceOutputLimitError(
                    _OUTPUT_LIMIT_MESSAGE, limit=_MAX_OUTPUT_BYTES, stdout=stdout, stderr=stderr
                )
        stdout, stderr = capture.texts()
        if exit_event is None or exit_event.timed_out:
            raise WorkspaceTimeoutError(f'Command timed out after {timeout:g} seconds', stdout=stdout, stderr=stderr)
        if capture.dropped:
            raise WorkspaceOutputLimitError(
                f'{capture.dropped} bytes of the command output were lost before they could be read (a managed '
                'process keeps the last 256 KiB of each stream); redirect it to a file instead',
                limit=_PROCESS_RING_BYTES,
                stdout=stdout[:_OUTPUT_PREVIEW_CHARS],
                stderr=stderr[:_OUTPUT_PREVIEW_CHARS],
            )
        return self._result(_exit_code(exit_event), stdout, stderr)

    async def realpath(self, path: str) -> str:
        """Follow every symlink in `path` like `os.path.realpath(strict=False)`, with core's shell resolution."""
        if '\0' in path:
            raise ValueError('path must not contain NUL bytes')
        # Not normalized: `link/..` climbs from the link's target.
        return await Workspace(_BoundedCommands(self)).realpath(path)

    async def read_bytes(self, path: str) -> bytes:
        path = _check_path(path)
        return await self._call(lambda sandbox: self._read_guarded(sandbox, path), path=path)

    async def _read_guarded(self, sandbox: AsyncSandbox, path: str) -> bytes:
        """Read `path`, unless it is a FIFO.

        The sandbox's open of a FIFO waits for a writer that may never come, so a read that hasn't answered soon
        is checked with `test -p`.
        """
        task = asyncio.ensure_future(self._read(sandbox, path))
        try:
            done, _ = await asyncio.wait([task], timeout=_FIFO_PROBE_AFTER)
            if not done and await self._is_fifo(path):
                raise OSError(errno.EINVAL, 'Reading a FIFO is not supported in the Platinum sandbox', path)
            return await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.wait([task])
            if not task.cancelled():
                task.exception()  # retrieved, so asyncio does not report it as never retrieved

    async def _is_fifo(self, path: str) -> bool:
        try:
            return (await self.run(['test', '-p', path], timeout=_INTERNAL_TIMEOUT)).exit_code == 0
        except Exception as error:  # the read itself reports a sandbox that is gone
            logger.debug('Platinum sandbox: probing %s for a FIFO failed: %r', path, error)
            return False

    async def _read(self, sandbox: AsyncSandbox, path: str) -> bytes:
        try:
            return await sandbox.files.read(path, timeout=_FILE_TIMEOUT)
        except platinum.PlatinumError as error:
            if not (error.status == 413 and error.code == 'file_too_large'):
                raise
        out = bytearray()
        part = _scratch('read')
        try:
            while True:
                skip = len(out) // (1024 * 1024)
                copied = await self._internal(
                    ['sh', '-c', _COPY_PART, 'sh', path, part, str(skip), str(_READ_PART_MIB)]
                )
                if copied.exit_code != 0:
                    raise WorkspaceError(f'Could not read {path!r} in parts: {copied.stderr.strip()[:500]}')
                chunk = await sandbox.files.read(part, timeout=_FILE_TIMEOUT)
                out.extend(chunk)
                if len(chunk) < _READ_PART_MIB * 1024 * 1024:
                    return bytes(out)
        finally:
            await self._delete_scratch(sandbox, part)

    async def _internal(self, argv: list[str]) -> CommandResult:
        return await self.run(argv, timeout=_EXEC_MAX_TIMEOUT)

    async def _delete_scratch(self, sandbox: AsyncSandbox, path: str) -> None:
        with anyio.move_on_after(_INTERNAL_TIMEOUT, shield=True):
            try:
                await sandbox.files.delete(path, recurse=True)
            except platinum.PlatinumError as error:
                logger.debug('Platinum sandbox %s: removing %s failed: %r', sandbox.id, path, error)

    async def write_bytes(self, path: str, data: bytes) -> None:
        path = _check_path(path)
        payload = bytes(data)
        await self._call(lambda sandbox: self._write(sandbox, path, payload), path=path)

    async def _write(self, sandbox: AsyncSandbox, path: str, data: bytes) -> None:
        """Write in one request, which creates missing parents and writes through a symlink.

        Past one request's limit, the data goes in parts that the sandbox joins.
        """
        part_size = _UPLOAD_PART_BYTES
        if len(data) <= part_size:
            try:
                return await sandbox.files.write(path, data, timeout=_FILE_TIMEOUT)
            except platinum.PlatinumError as error:
                limit = _write_limit(error)
                if limit is None or limit >= len(data):
                    raise
                part_size = limit
        parts = _scratch('upload')
        try:
            await self._write_parts(sandbox, parts, data, part_size)
            result = await self._internal(['sh', '-c', _JOIN_PARTS, 'sh', path, parts])
        finally:
            await self._delete_scratch(sandbox, parts)
        if result.exit_code == 121:
            raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)
        if result.exit_code == 120:
            raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), path)
        if result.exit_code != 0:
            raise WorkspaceError(f'Joining the uploaded parts of {path!r} failed: {result.stderr.strip()[:500]}')

    @staticmethod
    async def _write_parts(sandbox: AsyncSandbox, parts: str, data: bytes, part_size: int) -> None:
        offset = index = 0
        while offset < len(data):
            chunk = data[offset : offset + part_size]
            try:
                await sandbox.files.write(f'{parts}/part-{index:06d}', chunk, timeout=_FILE_TIMEOUT)
            except platinum.PlatinumError as error:
                limit = _write_limit(error)  # a deployment with a lower per-write limit says so
                if limit is None or limit >= len(chunk):
                    raise
                part_size = limit
                continue
            offset += len(chunk)
            index += 1

    async def stat(self, path: str) -> FileEntry:
        path = _check_path(path)
        info = await self._call(lambda sandbox: sandbox.files.stat(path), path=path)
        is_dir = bool(info.get('is_dir'))
        size: object = info.get('size')
        return FileEntry(
            name=posixpath.basename(path),
            path=path,
            is_dir=is_dir,
            size=size if type(size) is int and not is_dir else None,
        )

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        path = _check_path(path)
        return await self._call(lambda sandbox: self._list(sandbox, path), path=path)

    async def _list(self, sandbox: AsyncSandbox, path: str) -> list[FileEntry]:
        try:
            raw = await sandbox.files.list(path)
        except platinum.PlatinumError as error:
            if error.status == 413:  # `list_too_large`: over one reply's limit
                return await self._list_large(sandbox, path)
            raise
        entries: list[FileEntry] = []
        limit = anyio.Semaphore(_LIST_STAT_CONCURRENCY)

        async def add(item: Mapping[str, object]) -> None:
            if (entry := await _list_entry(sandbox, path, item, limit)) is not None:
                entries.append(entry)

        async with anyio.create_task_group() as group:
            for item in raw:
                group.start_soon(add, item)
        return sorted(entries, key=lambda entry: entry.name)

    async def _list_large(self, sandbox: AsyncSandbox, path: str) -> list[FileEntry]:
        listing = _scratch('list')
        try:
            result = await self._internal(['sh', '-c', _LIST_TO_FILE, 'sh', path, listing])
            if result.exit_code == 102:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
            if result.exit_code == 120:
                raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), path)
            if result.exit_code != 0:
                raise WorkspaceError(f'Listing {path!r} failed: {result.stderr.strip()[:500]}')
            data = await self._read(sandbox, listing)
        finally:
            await self._delete_scratch(sandbox, listing)
        records = data.decode(errors='surrogateescape').split('\0')
        return sorted(
            (
                FileEntry(name=posixpath.basename(record[1:]), path=record[1:], is_dir=record[0] == 'd', size=None)
                for record in records
                if len(record) > 1
            ),
            key=lambda entry: entry.name,
        )

    async def make_dir(self, path: str) -> None:
        path = _check_path(path)
        await self._call(lambda sandbox: _make_dir(sandbox, path), path=path)

    async def remove(self, path: str) -> None:
        """Remove a file, a symlink itself, or a directory and everything in it.

        The working directory and its ancestors are refused with `ValueError`, like the built-in backends do.
        """
        path = _check_path(path)
        async with self._operation():
            root = await self.working_dir()
            # The leaf stays unresolved: removing a link removes only the link.
            parent = await self.realpath(posixpath.dirname(path))
            target = posixpath.normpath(posixpath.join(parent, posixpath.basename(path)))
            if root == target or root.startswith(target.rstrip('/') + '/'):
                raise ValueError('cannot remove the workspace root or its ancestor')
            await self._call(lambda sandbox: _remove(sandbox, path), path=path)

    async def exists(self, path: str) -> bool:
        try:
            await self.stat(path)
        except (FileNotFoundError, NotADirectoryError):
            return False
        except OSError as error:
            if error.errno == errno.ELOOP:
                return False
            raise
        return True


async def _became_running(sandbox: AsyncSandbox) -> bool:
    """Wait for a sandbox that is starting; `False` if it settled as stopped or archived instead."""
    try:
        await sandbox.wait_state('running', timeout=_START_TIMEOUT)
    except platinum.PlatinumError as error:
        if error.code != 'not_running':
            raise
        return False
    return True


async def _follow(process: AsyncProcess, capture: _Capture, timeout: float | None) -> ProcessEvent | None:
    """Stream `process`'s output into `capture` until its `exit` event; `None` once the output is over the cap."""
    max_seconds = _STREAM_MAX_SECONDS if timeout is None else min(_STREAM_MAX_SECONDS, int(timeout + _PROCESS_GRACE))
    stream = process.stream(encoding='base64', max_seconds=max(1, max_seconds))
    try:
        async for event in stream:
            if event.type == 'exit':
                return event
            if event.type in ('stdout', 'stderr') and not capture.add(event):
                return None
    finally:
        if isinstance(stream, AsyncGenerator):
            await stream.aclose()
    # The SDK's stream ends only after an exit event, or raises.
    raise WorkspaceError(
        f'The output stream of Platinum process {process.handle} ended without an exit'
    )  # pragma: no cover


def _exit_code(event: ProcessEvent) -> int:
    """The exit code `sh` would report: 128 plus the signal for a command a signal ended."""
    if event.exit_signal:
        return 128 + event.exit_signal
    return event.exit_code if event.exit_code is not None else -1


async def _list_entry(
    sandbox: AsyncSandbox, directory: str, item: Mapping[str, object], limit: anyio.Semaphore
) -> FileEntry | None:
    """The entry `/files/list` describes, following a symlink to its target as the protocol asks."""
    name = item.get('name')
    if not isinstance(name, str) or not name:
        return None
    path = posixpath.join(directory, name)
    is_dir = bool(item.get('is_dir'))
    size = item.get('size')
    if not is_dir and item.get('mode') == _SYMLINK_MODE:
        # Perhaps a symlink: one that leads nowhere (dangling, a loop) reads as a file of unknown size.
        async with limit:
            try:
                target = await sandbox.files.stat(path)
            except platinum.PlatinumError:
                target = {}
        is_dir = bool(target.get('is_dir'))
        size = target.get('size')
    return FileEntry(name=name, path=path, is_dir=is_dir, size=size if type(size) is int and not is_dir else None)


async def _make_dir(sandbox: AsyncSandbox, path: str) -> None:
    try:
        await sandbox.files.mkdir(path)
    except platinum.PlatinumError as error:
        if error.code != 'not_a_directory':
            raise
        # The API says the same for a file at the path and for one among its parents.
        try:
            info = await sandbox.files.stat(path)
        except platinum.PlatinumError:
            raise error from None
        if not info.get('is_dir'):
            raise FileExistsError(errno.EEXIST, f'{os.strerror(errno.EEXIST)} in the Platinum sandbox', path) from error
        raise


async def _remove(sandbox: AsyncSandbox, path: str) -> None:
    # Not recursive at first: a recursive delete succeeds on a missing path, which must be `FileNotFoundError`.
    try:
        await sandbox.files.delete(path)
    except platinum.PlatinumError as error:
        if error.code != 'directory_not_empty':
            raise
        await sandbox.files.delete(path, recurse=True)
