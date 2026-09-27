"""A native Modal sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

SDK assumptions verified 2026-09-09 against Modal Python SDK 1.5.2:

* `.aio`, `from_id`, `poll`, `terminate`, and `detach` follow the native sandbox lifecycle:
  https://modal.com/docs/guide/sandboxes
* `exec` uses whole-second deadlines and has no per-command kill operation:
  https://modal.com/docs/guide/sandbox-spawn
* the native filesystem API supplies the operations used here:
  https://modal.com/docs/sdk/py/latest/Sandbox

Re-check these SDK methods before changing the protocol integration.
"""

from __future__ import annotations

import asyncio
import codecs
import importlib
import logging
import math
import posixpath
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from uuid import uuid4

import anyio
from pydantic_ai.exceptions import UserError
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

from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    command_argv,
    command_deadline,
    safe_credential_reason,
)

if TYPE_CHECKING:
    import modal
    import modal.container_process
    import modal.io_streams
    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('ModalSandboxBackend',)

DEFAULT_APP_NAME = 'pydantic-ai-harness'
# Modal's maximum sandbox lifetime (24 hours). The framework never terminates a sandbox, so a
# conversation can continue in it for as long as Modal allows.
DEFAULT_SANDBOX_TIMEOUT = 86_400

_MISSING_MODAL = (
    'The \'modal\' package is required for ModalSandbox. Install it with `uv add "pydantic-ai-harness[modal]"`.'
)

_AUTH_MESSAGE = 'Modal rejected the credentials. Set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or run `modal token new`.'

# Bound the workspace-create RPCs so a wedged control plane cannot hang acquisition.
_CREATE_TIMEOUT = 600


_INTERNAL_EXEC_TIMEOUT = 10

_CLIENT_DEADLINE_EXIT = -1
_SIGKILL_EXIT = 137

_PARTIAL_OUTPUT_LIMIT = 65_536
logger = logging.getLogger(__name__)


def _translate(error: Exception, *, context: str, unavailable: str, path: str | None = None) -> Exception | None:
    """Map a Modal SDK exception onto the workspace protocol's typed failures.

    Returns `None` for an exception that must propagate unchanged: Modal's connection, rate-limit,
    and internal-service errors, and anything unrecognized, are transient infrastructure failures
    that a durable engine retries. `unavailable` is the message for a sandbox that no longer
    exists; `path` is set for a filesystem operation, whose path-level errors become the builtin
    ones.
    """
    import modal

    exc = modal.exception
    if isinstance(error, (exc.AuthError, exc.PermissionDeniedError)):
        return WorkspaceUnavailableError(f'{safe_credential_reason(error)}. {_AUTH_MESSAGE}')
    # `SandboxTimeoutError` is the sandbox reaching its lifetime (`sandbox_timeout`), not a
    # command timing out; a command's own deadline is handled in `run()`.
    # For about 30 seconds after `terminate()`, Modal still polls the sandbox as running while
    # refusing exec with a `ConflictError`; only its message separates it from a transient conflict.
    if isinstance(error, (exc.NotFoundError, exc.SandboxTerminatedError, exc.SandboxTimeoutError)) or (
        isinstance(error, exc.ConflictError) and 'shutting down' in str(error).lower()
    ):
        return WorkspaceUnavailableError(unavailable)
    if path is not None:
        path_errors: tuple[tuple[type[Exception], type[OSError], str], ...] = (
            (exc.SandboxFilesystemNotFoundError, FileNotFoundError, 'No such file or directory'),
            (exc.SandboxFilesystemIsADirectoryError, IsADirectoryError, 'Is a directory'),
            (exc.SandboxFilesystemNotADirectoryError, NotADirectoryError, 'Not a directory'),
            (exc.SandboxFilesystemPermissionError, PermissionError, 'Permission denied'),
            (exc.SandboxFilesystemPathAlreadyExistsError, FileExistsError, 'File exists'),
        )
        for sdk_type, builtin, description in path_errors:
            if isinstance(error, sdk_type):
                return builtin(f'{description} in the Modal sandbox: {path!r}')
    if isinstance(
        error,
        (
            exc.InvalidError,  # `ConflictError` subclasses it; the caller probes whether the sandbox is gone
            exc.AlreadyExistsError,
            exc.ExecutionError,
            exc.RequestSizeError,
            exc.SandboxFilesystemError,
            exc.FilesystemExecutionError,
        ),
    ):
        return WorkspaceError(f'{context}: {error}')
    return None


def _unwrap_filesystem_error(error: Exception) -> Exception:
    """The SDK failure Modal's filesystem layer replaced, or `error` itself.

    Modal 1.5.2 runs each filesystem operation as an exec and re-raises what that exec raised
    `from None`: connection and service failures as a `NotFoundError` ("the Sandbox is
    unavailable"), every other SDK failure -- rate limits and internal errors included -- as a
    generic `SandboxFilesystemError`. Classifying those replacements would turn a retryable
    transport failure into a terminal one, so the original, still in `__context__`, is used.
    """
    import modal

    exc = modal.exception
    original = error.__context__
    if (
        type(error) in (exc.NotFoundError, exc.SandboxFilesystemError)
        and error.__suppress_context__
        and isinstance(original, exc.Error)
    ):
        return original
    return error


# Linux's limit on symlinks followed while resolving one path; a longer chain is a loop.
_MAX_SYMLINK_HOPS = 40


async def _follow_symlinks(
    sandbox: modal.Sandbox, path: str, entry: modal.types.FileInfo | None
) -> tuple[str, modal.types.FileInfo | None]:
    """The path a symlink at `path` finally leads to, and its entry: `None` if it doesn't exist.

    Modal's `stat` and `list_files` describe a symlink itself, so a link is resolved by stat-ing its
    target, hop by hop. A dangling or looping link has no entry.
    """
    import modal

    link, hops = path, 0
    visited = {posixpath.normpath(path)}
    while entry is not None and entry.is_symlink():
        hops += 1
        if hops > _MAX_SYMLINK_HOPS or entry.symlink_target is None:
            return link, None
        # A relative target is relative to the directory holding the link.
        link = posixpath.normpath(posixpath.join(posixpath.dirname(link), entry.symlink_target))
        if link in visited:
            # Modal does not detect all cycles in its symlink metadata; skip repeated RPCs.
            entry = None
            continue
        visited.add(link)
        try:
            entry = await sandbox.filesystem.stat.aio(link)
        except (
            modal.exception.SandboxFilesystemNotFoundError,
            modal.exception.SandboxFilesystemNotADirectoryError,
        ):
            entry = None
    return link, entry


async def _file_entry(sandbox: modal.Sandbox, entry: modal.types.FileInfo, path: str) -> FileEntry:
    """The protocol entry for `entry` at `path`, with `is_dir` and `size` following a symlink.

    A dangling or looping link is reported as a file with no size.
    """
    _, target = await _follow_symlinks(sandbox, path, entry)
    is_dir = target is not None and target.is_dir()
    # A directory's reported size is an implementation detail of the underlying filesystem
    # rather than a content length, so report none for it, like the built-in backends.
    size = None if target is None or is_dir else target.size
    return FileEntry(name=entry.name, path=path, is_dir=is_dir, size=size)


class ModalSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem):
    """A Modal sandbox implementing Pydantic AI's `WorkspaceBackend` protocol.

    Construction performs no I/O. The first operation creates or attaches to a sandbox, and the
    typed `modal.Sandbox` is available through `get_sandbox()`. The backend does not terminate the
    sandbox; terminating it is the application's job.

    Commands run in isolated process groups. Cancellation and command deadlines attempt to stop
    the foreground group without terminating the sandbox shared by other commands.

    The protocol is structural, but subclassing it here makes a signature drift fail the type
    check on this class instead of at a distant `WorkspaceBackend` call.

    Args:
        sandbox: A live `modal.Sandbox` you already have. Whoever created it owns terminating it.
        ref: Identity of an existing sandbox to attach to on first use.
        image: Registry tag, or a `modal.Image`, a newly created sandbox runs. `None` (the default)
            is Debian slim with Python 3.12, `git`, and `ripgrep`.
        app_name: Modal app a newly created sandbox belongs to.
        create_app_if_missing: Create the Modal app when it does not exist yet.
        sandbox_timeout: Total lifetime of a newly created sandbox, in seconds (Modal's `timeout`).
            Defaults to Modal's maximum, 24 hours.
        idle_timeout: Seconds without activity after which Modal terminates a newly created
            sandbox; `None` (the default) never terminates it for being idle.
        working_dir: Absolute directory commands start in and relative paths resolve against,
            applied to every command, including in an attached sandbox; the image's when `None`.
        env: Environment variables every command gets; a command's own `env` is layered on top.
    """

    def __init__(
        self,
        *,
        sandbox: modal.Sandbox | None = None,
        ref: WorkspaceRef | None = None,
        image: str | modal.Image | None = None,
        app_name: str = DEFAULT_APP_NAME,
        create_app_if_missing: bool = True,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        idle_timeout: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None and ref.provider != 'modal':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'modal'")
        if sandbox is not None and ref is not None:
            raise ValueError('pass either `sandbox` or `ref`, not both')
        if type(sandbox_timeout) is not int or not 10 <= sandbox_timeout <= 86_400:
            raise ValueError('sandbox_timeout must be an integer between 10 and 86400 seconds')
        if image is not None and not isinstance(image, str):
            modal_sdk = importlib.import_module('modal')
            if not isinstance(image, modal_sdk.Image):
                raise TypeError('image must be a registry tag or modal.Image')
        if env is not None and any(type(key) is not str or type(value) is not str for key, value in env.items()):
            raise TypeError('env keys and values must be strings')
        self._ref = ref if sandbox is None else WorkspaceRef(provider='modal', id=sandbox.object_id)
        self._sandbox: modal.Sandbox | None = sandbox
        self._image = image
        self._app_name = app_name
        self._create_app_if_missing = create_app_if_missing
        self._sandbox_timeout = sandbox_timeout
        self._idle_timeout = idle_timeout
        self._working_dir = absolute_path('working_dir', working_dir)
        self._env = dict(env) if env is not None else {}
        # `_working_dir` is what was configured (`None` for the image's); the protocol needs the
        # canonical absolute path, which only `pwd -P` in the sandbox can give.
        self._resolved_working_dir: str | None = None
        self._lock = anyio.Lock()
        # Keep the detached acquisition alive after a caller's native Task.cancel().
        self._acquisition: asyncio.Task[modal.Sandbox] | None = None
        self._create_name: str | None = None
        # The default Debian image includes setsid; custom and attached images need a probe.
        self._setsid_available: bool | None = True if image is None and ref is None and sandbox is None else None
        self._setsid_lock = anyio.Lock()

    async def get_sandbox(self) -> modal.Sandbox:
        """Return the typed `modal.Sandbox`, creating or attaching to it on first use.

        The lock serializes concurrent first uses -- two callers each creating a sandbox would
        leave the loser billed and unreferenced. Attaching by `ref` to a sandbox that no longer
        exists raises `WorkspaceUnavailableError`; it does not create a replacement.

        Raises:
            UserError: The `modal` package is not installed.
        """
        if (sandbox := self._sandbox) is not None:
            return sandbox
        task = self._acquisition
        if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
            # Native Task.cancel() pierces AnyIO shields. The child owns recording the ref,
            # so a cancelled caller cannot strand a committed sandbox without an identity.
            task = asyncio.create_task(self._acquire())
            self._acquisition = task
        return await asyncio.shield(task)

    async def _acquire(self) -> modal.Sandbox:
        async with self._lock:
            if (sandbox := self._sandbox) is not None:
                return sandbox
            try:
                importlib.import_module('modal')
            except ImportError as e:
                raise UserError(_MISSING_MODAL) from e
            ref = self._ref
            sandbox = await self._attach(ref.id) if ref is not None else await self._create()
            self._sandbox = sandbox
            self._ref = WorkspaceRef(provider='modal', id=sandbox.object_id)
            return sandbox

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the sandbox, or `None` before one has been created."""
        return self._ref

    @asynccontextmanager
    async def _mapped_errors(
        self, sandbox: modal.Sandbox, context: str, path: str | None = None
    ) -> AsyncGenerator[None]:
        """Raise the protocol's typed failure for a Modal exception, and let anything else through."""
        try:
            yield
        except Exception as wrapped:
            error = _unwrap_filesystem_error(wrapped) if path is not None else wrapped
            mapped = await _failure(sandbox, error, context, path)
            if mapped is None:
                raise error
            raise mapped from error

    async def read_bytes(self, path: str) -> bytes:
        absolute_path('path', path)
        sandbox = await self.get_sandbox()
        # Modal's filesystem read opens FIFOs in blocking mode and its FileInfo omits the
        # file kind. Probe through the sandbox shell before entering that unbounded SDK read.
        if (await self.run(['test', '-p', path], timeout=_INTERNAL_EXEC_TIMEOUT)).exit_code == 0:
            raise OSError(f'Cannot read FIFO in the Modal sandbox: {path!r}')
        async with self._mapped_errors(sandbox, f'Could not read {path!r}', path):
            return await sandbox.filesystem.read_bytes.aio(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        absolute_path('path', path)
        # Modal takes the data first, creates missing parents, and replaces existing contents.
        import modal

        sandbox = await self.get_sandbox()
        async with self._mapped_errors(sandbox, f'Could not write {path!r}', path):
            # Modal replaces a symlink on write, unlike open(2). Resolve the leaf
            # explicitly so writes through links update the same file the shell sees.
            target = path
            visited: set[str] = set()
            for _ in range(_MAX_SYMLINK_HOPS):
                normalized = posixpath.normpath(target)
                if normalized in visited:
                    raise OSError(f'Symlink loop in the Modal sandbox: {path!r}')
                visited.add(normalized)
                try:
                    info = await sandbox.filesystem.stat.aio(target)
                except modal.exception.SandboxFilesystemNotFoundError:
                    break
                if not info.is_symlink() or info.symlink_target is None:
                    break
                target = posixpath.normpath(posixpath.join(posixpath.dirname(target), info.symlink_target))
            else:
                raise OSError(f'Too many symlinks in the Modal sandbox: {path!r}')
            await sandbox.filesystem.write_bytes.aio(data, target)

    async def stat(self, path: str) -> FileEntry:
        absolute_path('path', path)
        sandbox = await self.get_sandbox()
        async with self._mapped_errors(sandbox, f'Could not stat {path!r}', path):
            info = await sandbox.filesystem.stat.aio(path)
            entry = await _file_entry(sandbox, info, path)
            # Listings retain broken links, but stat follows them like the local backend.
            if info.is_symlink() and entry.size is None and not entry.is_dir:
                raise FileNotFoundError(path)
            return entry

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        absolute_path('path', path)
        sandbox = await self.get_sandbox()
        async with self._mapped_errors(sandbox, f'Could not list {path!r}', path):
            entries = await sandbox.filesystem.list_files.aio(path)
            # Limit simultaneous SDK requests without making large link-heavy listings serial.
            limit = anyio.Semaphore(8)
            results: list[FileEntry | None] = [None] * len(entries)

            async def resolve(index: int, entry: modal.types.FileInfo) -> None:
                async with limit:
                    results[index] = await _file_entry(sandbox, entry, posixpath.join(path, entry.name))

            async with anyio.create_task_group() as group:
                for index, entry in enumerate(entries):
                    group.start_soon(resolve, index, entry)
            return [result for result in results if result is not None]

    async def make_dir(self, path: str) -> None:
        absolute_path('path', path)
        sandbox = await self.get_sandbox()
        async with self._mapped_errors(sandbox, f'Could not create directory {path!r}', path):
            await sandbox.filesystem.make_directory.aio(path)

    async def remove(self, path: str) -> None:
        absolute_path('path', path)
        sandbox = await self.get_sandbox()
        async with self._mapped_errors(sandbox, f'Could not remove {path!r}', path):
            await sandbox.filesystem.remove.aio(path, recursive=True)

    async def exists(self, path: str) -> bool:
        try:
            await self.stat(path)
        except (FileNotFoundError, NotADirectoryError):
            # Modal splits "there is nothing at that path" in two: a missing entry, and a
            # non-leaf component that is a file.
            return False
        return True

    async def _create(self) -> modal.Sandbox:
        """Provision a fresh Modal sandbox."""
        import modal

        sandbox: modal.Sandbox | None = None
        create_attempted = False
        if self._create_name is None:
            self._create_name = f'pydantic-ai-{uuid4().hex}'
        name = self._create_name
        try:
            # Shielded so that a caller cancelled mid-create still gets the sandbox Modal made:
            # `get_sandbox` records it before the cancellation is delivered, so `ref` names it and
            # a retry reuses it. Only the local deadline interrupts the call; a sandbox created
            # after it fires is reaped at its `sandbox_timeout`.
            with anyio.CancelScope(shield=True), anyio.move_on_after(_CREATE_TIMEOUT):
                app = await modal.App.lookup.aio(self._app_name, create_if_missing=self._create_app_if_missing)
                if self._image is None:
                    # Built on create, not at import: Modal caches it per workspace after the first build.
                    built = modal.Image.debian_slim(python_version='3.12').apt_install('git', 'ripgrep')  # pyright: ignore[reportUnknownMemberType]
                elif isinstance(self._image, str):
                    built = modal.Image.from_registry(self._image)  # pyright: ignore[reportUnknownMemberType]
                else:
                    built = self._image
                variables: dict[str, str | None] | None = dict(self._env) if self._env else None
                create_attempted = True
                sandbox = await modal.Sandbox.create.aio(  # pyright: ignore[reportUnknownMemberType]
                    app=app,
                    image=built,
                    name=name,
                    workdir=self._working_dir,
                    env=variables,
                    timeout=self._sandbox_timeout,
                    idle_timeout=self._idle_timeout,
                )
        except Exception as error:
            if create_attempted and (recovered := await self._recover_create(name)) is not None:
                return recovered
            message = f'Could not start Modal sandbox: {error}'
            mapped = _translate(error, context=message, unavailable=message)
            # SDK releases disagree on whether image build errors live in modal.exception.
            if type(error).__name__ == 'ImageBuildError':
                raise WorkspaceUnavailableError(message) from error
            if mapped is None:
                raise
            # Modal refused the request itself: an unknown app or image, or an invalid argument
            # such as a `sandbox_timeout` above its limit. Retrying cannot fix that, so it ends
            # the run rather than going back to the model.
            raise (
                mapped if isinstance(mapped, WorkspaceUnavailableError) else WorkspaceUnavailableError(message)
            ) from error
        if sandbox is None:
            if create_attempted and (recovered := await self._recover_create(name)) is not None:
                return recovered
            # Keep the name across retries, so a late commit can still be recovered.
            raise TimeoutError(
                f'Modal sandbox creation did not complete within {_CREATE_TIMEOUT}s; '
                'an image build or pull may still be running. Check for an existing sandbox before retrying.'
            )
        return sandbox

    async def _recover_create(self, name: str) -> modal.Sandbox | None:
        """Reattach a sandbox whose create RPC committed but whose response was lost."""
        import modal

        try:
            sandbox = await modal.Sandbox.from_name.aio(self._app_name, name)
            return sandbox if await sandbox.poll.aio() is None else None
        except modal.exception.NotFoundError:
            return None
        except Exception:
            # Recovery is best effort; retain the original failure for retry policy.
            logger.warning('Could not check whether named Modal sandbox creation completed')
            return None

    async def _attach(self, sandbox_id: str) -> modal.Sandbox:
        """Attach to a Modal sandbox that already exists.

        Modal hands back a handle for a sandbox it still knows about even after that sandbox
        has terminated, so this polls: a `WorkspaceRef` must not resolve to a dead environment.
        Nothing is recreated in its place -- a run that expected files there must be told they
        are gone, not handed an empty sandbox.
        """
        import modal

        try:
            sandbox = await modal.Sandbox.from_id.aio(sandbox_id)
            finished = await sandbox.poll.aio()
        except Exception as error:
            mapped = _translate(
                error,
                context=f'Could not connect to Modal sandbox {sandbox_id!r}',
                unavailable=_unavailable_message(sandbox_id),
            )
            if mapped is None:
                raise
            # A malformed stored ref cannot become valid by retrying an attach.
            if isinstance(error, modal.exception.InvalidError):
                raise WorkspaceUnavailableError(_unavailable_message(sandbox_id)) from error
            raise mapped from error
        if finished is not None:
            raise WorkspaceUnavailableError(_unavailable_message(sandbox_id))
        return sandbox

    async def working_dir(self) -> str:
        """The sandbox's working directory (absolute POSIX path)."""
        # Modal exposes no API for a running sandbox's working directory -- it is the image's,
        # or the configured `working_dir` every command is given -- so ask the sandbox itself,
        # which also canonicalizes a configured path. It cannot change, so the probe is an
        # idempotent read: overlapping first calls may each run their own `pwd`, get the same
        # answer, and the cache converges. No lock needed.
        if self._resolved_working_dir is None:
            sandbox = await self.get_sandbox()
            result = await self.run(['pwd', '-P'], timeout=_INTERNAL_EXEC_TIMEOUT)
            printed = result.stdout.removesuffix('\n')
            # Only an absolute path is an answer. Caching whatever else the sandbox printed
            # would hand every later `resolve()` a working directory that is not one,
            # mis-resolving relative paths with no error.
            if result.exit_code != 0 or not posixpath.isabs(printed):
                raise WorkspaceError(
                    f'Could not determine the working directory of Modal sandbox {sandbox.object_id!r}: '
                    f'`pwd` exited {result.exit_code} and printed {result.stdout!r}. Use absolute paths.'
                )
            self._resolved_working_dir = printed
        return self._resolved_working_dir

    async def _has_setsid(self, sandbox: modal.Sandbox) -> bool:
        # Cache the probe across concurrent runs on custom or attached images.
        async with self._setsid_lock:
            if self._setsid_available is None:
                async with self._mapped_errors(sandbox, 'Could not check Modal process isolation'):
                    probe = await sandbox.exec.aio(
                        'sh',
                        '-c',
                        'command -v setsid >/dev/null 2>&1 && setsid -w true',
                        timeout=_INTERNAL_EXEC_TIMEOUT,
                        workdir='/',
                        text=False,
                    )
                    self._setsid_available = await probe.wait.aio() == 0
            return self._setsid_available

    async def run(  # noqa: C901 - deadline, cancellation and Modal exit classification share the same process state
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Execute a command and wait for it to complete.

        A shell command runs as `/bin/sh -c <command>`. Cancellation and command deadlines
        stop its foreground process group, without terminating the shared sandbox.
        """
        # Modal executes argv and never a shell string, so shell interpretation is requested
        # explicitly through `/bin/sh -c`, the one shell every sandbox image carries.
        argv = command_argv(command, shell)
        if not shell:
            # Through `sh`, so a program that can't start exits 127 or 126 as it does in `sh`.
            argv = ['/bin/sh', '-c', 'exec "$@"', 'sh', *argv]
        # Given per command, not only at creation, so an attached sandbox honors it too.
        workdir = absolute_path('cwd', cwd) or self._working_dir
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
        variables: dict[str, str | None] | None = {**self._env, **(env or {})} or None
        # Acquiring the sandbox has its own bound; the timeout is the command's alone.
        sandbox = await self.get_sandbox()
        if cwd is not None:
            await self.stat(cwd)
        started_at = time.monotonic()
        # Modal takes whole seconds and reads 0 as no deadline, so round up. The client
        # deadline below includes exec-start, collection, and stream drain.
        deadline = None if timeout is None else max(1, math.ceil(timeout))
        timed_out = f'Command timed out after {timeout} seconds.'
        token = uuid4().hex
        pid_file = f'/tmp/.pydantic-modal-{token}.pid'
        cancel_file = f'/tmp/.pydantic-modal-{token}.cancel'
        # Without setsid, only the wrapper leader can be signalled, not its descendants.
        isolated = await self._has_setsid(sandbox)
        # Register a killable group before running user code; a tombstone prevents a late
        # exec-start reply from launching work after its caller has been cancelled.
        # `setsid -w` waits for its child; without -w Modal reports success after the fork.
        # Keep the group leader alive while the child runs, so it can remove its marker
        # after completion. A cancelled late start still sees the tombstone before user code.
        start_script = (
            'test ! -e "$1" || exit 143; pid_file=$2; echo $$ > "$pid_file"; '
            'if test -e "$1"; then rm -f "$pid_file"; exit 143; fi; '
            'shift 2; "$@" </dev/null; status=$?; rm -f "$pid_file"; exit "$status"'
        )
        wrapped = [
            *(['setsid', '-w'] if isolated else []),
            'sh',
            '-c',
            start_script,
            'modal-command',
            cancel_file,
            pid_file,
            *argv,
        ]
        # dash's builtin kill rejects `--`; the negative PID addresses the group. Give TERM
        # a short grace period, then KILL survivors (including TERM-ignoring descendants).
        target = '-"$pid"' if isolated else '"$pid"'
        stop_script = (
            'touch "$1"; if test -f "$2"; then '
            f'pid=$(cat "$2"); kill -TERM {target} 2>/dev/null || true; '
            f'i=0; while kill -0 {target} 2>/dev/null && test "$i" -lt 5; do '
            'sleep 0.2; i=$((i+1)); done; '
            f'if kill -0 {target} 2>/dev/null; then kill -KILL {target} 2>/dev/null || true; fi; '
            'fi; rm -f "$2"'
        )

        async def stop() -> None:
            try:
                stopper = await sandbox.exec.aio(
                    'sh',
                    '-c',
                    stop_script,
                    'modal-stop',
                    cancel_file,
                    pid_file,
                    timeout=2,
                    # The user's cwd may have been deleted by the command itself.
                    workdir='/',
                    text=False,
                )
                await _check_stop(stopper, sandbox.object_id)
            except Exception:
                # Stop can fail when the sandbox is already gone; preserve the original
                # cancellation/error and leave its ref available for explicit cleanup.
                logger.warning('Could not stop Modal command in sandbox %s', sandbox.object_id)

        snapshots = ['', '']

        def captured() -> tuple[str, str]:
            return snapshots[0], snapshots[1]

        exit_event = anyio.Event()
        done_event = anyio.Event()
        results: list[tuple[str, str, int, float]] = []
        errors: list[Exception] = []
        async with command_deadline(timeout, stop=stop, output=captured):
            async with self._mapped_errors(sandbox, 'Command could not run in the workspace'):
                process = await sandbox.exec.aio(*wrapped, timeout=deadline, workdir=workdir, env=variables, text=False)
        command_error: WorkspaceTimeoutError | None = None
        async with anyio.create_task_group() as group:

            async def collect() -> None:
                try:
                    results.append(await _collect_output(process, snapshots, exit_event))
                except Exception as error:
                    errors.append(error)
                    exit_event.set()
                finally:
                    done_event.set()

            group.start_soon(collect)
            try:
                async with command_deadline(
                    None if timeout is None else max(0.001, timeout - (time.monotonic() - started_at)),
                    stop=stop,
                    output=captured,
                ):
                    await exit_event.wait()
            except WorkspaceTimeoutError as error:
                command_error = error
                group.cancel_scope.cancel()
            else:
                # Once the process exits, output can drain past its command deadline.
                with anyio.move_on_after(None if timeout is None else max(2, timeout)) as drain_scope:
                    await done_event.wait()
                if drain_scope.cancelled_caught:
                    group.cancel_scope.cancel()
                    stdout, stderr = captured()
                    command_error = WorkspaceTimeoutError(timed_out, stdout=stdout, stderr=stderr)
        if command_error is not None:
            raise command_error
        if errors:
            mapped = await _failure(sandbox, errors[0], 'Could not read the command result')
            if mapped is not None:
                raise mapped from errors[0]
            raise errors[0]
        stdout, stderr, exit_code, exited_at = results[0]

        if deadline is not None and (
            exit_code == _CLIENT_DEADLINE_EXIT or (exit_code == _SIGKILL_EXIT and exited_at - started_at >= deadline)
        ):
            raise WorkspaceTimeoutError(timed_out, stdout=stdout, stderr=stderr)
        if exit_code == _SIGKILL_EXIT:
            # 137 can also be a user's SIGKILL; only a finished sandbox proves the workspace died.
            # Keep a stalled control-plane probe from delaying an otherwise valid command result.
            try:
                with anyio.fail_after(2):
                    finished = await sandbox.poll.aio()
            except Exception:
                pass
            else:
                if finished is not None:
                    raise WorkspaceUnavailableError(_unavailable_message(sandbox.object_id))
        return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)


async def _collect_output(
    process: modal.container_process.ContainerProcess[bytes], snapshots: list[str], exit_event: anyio.Event
) -> tuple[str, str, int, float]:
    """Drain both streams and wait as one owned operation."""

    async def read(reader: modal.io_streams.StreamReader[bytes], index: int) -> str:
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        pieces: list[str] = []
        async for chunk in reader:
            text = decoder.decode(chunk)
            pieces.append(text)
            # Keep a bounded snapshot even when Modal interrupts a streaming read.
            snapshots[index] = (snapshots[index] + text)[-_PARTIAL_OUTPUT_LIMIT:]
        tail = decoder.decode(b'', final=True)
        pieces.append(tail)
        snapshots[index] = (snapshots[index] + tail)[-_PARTIAL_OUTPUT_LIMIT:]
        return ''.join(pieces)

    stdout, stderr, exit_code = '', '', 0
    exited_at = math.inf
    read_error: Exception | None = None

    async with anyio.create_task_group() as group:

        async def collect(reader: modal.io_streams.StreamReader[bytes], index: int) -> None:
            nonlocal stdout, stderr, read_error
            try:
                result = await read(reader, index)
                if index == 0:
                    stdout = result
                else:
                    stderr = result
            except Exception as error:
                # Cancel the other reader and wait together; preserve the SDK error
                # without wrapping a single failure in an ExceptionGroup.
                read_error = error
                group.cancel_scope.cancel()

        async def wait() -> None:
            nonlocal exit_code, exited_at, read_error
            try:
                exit_code = await process.wait.aio()
                # Output can drain later; classify a deadline kill by the process exit time.
                exited_at = time.monotonic()
                exit_event.set()
            except Exception as error:
                read_error = error
                group.cancel_scope.cancel()

        group.start_soon(collect, process.stdout, 0)
        group.start_soon(collect, process.stderr, 1)
        group.start_soon(wait)
    if read_error is not None:
        raise read_error

    return stdout, stderr, exit_code, exited_at


async def _check_stop(process: modal.container_process.ContainerProcess[bytes], sandbox_id: str) -> None:
    if await process.wait.aio() != 0:
        logger.warning('Modal command stop exited nonzero in sandbox %s', sandbox_id)


def _unavailable_message(sandbox_id: str) -> str:
    return (
        f'The Modal sandbox {sandbox_id!r} is no longer running: it was terminated, or it reached its '
        "`sandbox_timeout` or `idle_timeout`. Pass `workspace='new'` to start a fresh sandbox."
    )


async def _failure(sandbox: modal.Sandbox, error: Exception, context: str, path: str | None = None) -> Exception | None:
    """Translate an SDK failure on `sandbox`, or `None` to let it propagate.

    Two Modal errors do not say whether the sandbox is gone: exec on a dead sandbox raises
    `ConflictError`, which Modal also uses for transient aborts, and the filesystem layer reports a
    dead sandbox as a generic `SandboxFilesystemError`. For those, the sandbox is probed before the
    error is reported as an ordinary operation failure.
    """
    import modal

    mapped = _translate(error, context=context, unavailable=_unavailable_message(sandbox.object_id), path=path)
    if type(mapped) is WorkspaceError and isinstance(
        error, (modal.exception.ConflictError, modal.exception.SandboxFilesystemError)
    ):
        return await _probe(sandbox, error) or mapped
    return mapped


async def _probe(sandbox: modal.Sandbox, error: Exception) -> WorkspaceUnavailableError | None:
    """`WorkspaceUnavailableError` if `sandbox` has stopped running, else `None`."""
    # Probing only after an error keeps the extra round trip off successful operations.
    import modal

    unavailable = _unavailable_message(sandbox.object_id)
    try:
        finished = await sandbox.poll.aio()
        if finished is None and isinstance(error, modal.exception.SandboxFilesystemError):
            # A terminated sandbox that is still shutting down polls as running and fails
            # filesystem calls with a generic error; only exec names the state.
            with anyio.fail_after(_INTERNAL_EXEC_TIMEOUT):
                await sandbox.exec.aio('true', timeout=_INTERNAL_EXEC_TIMEOUT)
    except Exception as probe_error:
        # A probe failing for any other reason, a transport error included, leaves the
        # original error standing rather than replacing it.
        mapped = _translate(probe_error, context='', unavailable=unavailable)
        return mapped if isinstance(mapped, WorkspaceUnavailableError) else None
    if finished is not None:
        return WorkspaceUnavailableError(unavailable)
    return None
