"""Lifecycle management for a Modal sandbox."""

from __future__ import annotations

import math
import posixpath
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
import anyio.lowlevel
from typing_extensions import Self

if TYPE_CHECKING:
    import modal
    import modal.io_streams

# Defaults shared by `ModalSandboxSession` and `ModalSandbox` (which imports them), so the
# two public constructors cannot drift: a setting is "left at its default" iff it equals
# the constant here.
DEFAULT_IMAGE = 'python:3.12-slim'
DEFAULT_APP_NAME = 'pydantic-ai-harness'
DEFAULT_SANDBOX_TIMEOUT = 300


class ModalSandboxError(RuntimeError):
    """Base class for failures reported by the Modal sandbox integration.

    The toolset turns direct instances into `ModelRetry`. Terminal subclasses
    propagate because retrying cannot restore a missing sandbox or credentials.
    """


class ModalSandboxTerminalError(ModalSandboxError):
    """A sandbox failure that retrying cannot fix, so the run should end, not loop.

    The toolset lets this propagate out of the tool (ending the run) instead of
    turning it into a `ModelRetry`: re-issuing the command would hit the same wall.
    Raised as `ModalSandboxUnavailableError` for a sandbox that no longer exists
    and `ModalSandboxAuthError` for rejected credentials.
    """


class ModalSandboxUnavailableError(ModalSandboxTerminalError):
    """The sandbox no longer exists: terminated, or expired at its `sandbox_timeout`.

    Every later command against it would fail the same way, so it is terminal. In
    owned mode this is what a run outliving the sandbox lifetime looks like; raise
    `sandbox_timeout` (or shorten the work) if runs legitimately need longer.
    """


class ModalSandboxAuthError(ModalSandboxTerminalError):
    """Modal rejected the credentials, so no sandbox operation can succeed.

    Fixing this is an operator action (configure credentials), not something a
    retry or a new run can do, which is why it is terminal.
    """


@dataclass(frozen=True, kw_only=True)
class ModalSandboxExecResult:
    """The outcome of running a command in the sandbox."""

    stdout: str
    """The command's standard output, tail-truncated when `max_output_bytes` was set."""
    stderr: str
    """The command's standard error, tail-truncated when `max_output_bytes` was set."""
    returncode: int
    """The exit status: 0-255 for a real exit (128+n for signal n), or Modal's `-1` deadline sentinel."""
    stdout_truncated: bool = False
    """True when `max_output_bytes` dropped earlier stdout bytes; `stdout` is the retained tail."""
    stderr_truncated: bool = False
    """True when `max_output_bytes` dropped earlier stderr bytes; `stderr` is the retained tail."""
    timed_out: bool = False
    """True when a deadline was applied and the command was killed by it.

    Modal reports a client-side deadline kill as returncode `-1`; a server-side kill at
    the same deadline surfaces as the plain SIGKILL exit (137), so that exit is also
    read as a timeout when the command consumed its whole deadline window.
    """
    applied_timeout: int | None = None
    """The whole-second deadline Modal enforced for this command, or None if unbounded.

    This is the quantized value actually sent to Modal, not the (possibly fractional)
    timeout the caller requested, so the caller can report the exact deadline.
    """


_MISSING_MODAL = (
    'The \'modal\' package is required for ModalSandbox. Install it with `uv add "pydantic-ai-harness[modal]"`.'
)

_AUTH_MESSAGE = 'Modal rejected the credentials. Set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or run `modal token new`.'

# Bound the sandbox-create RPCs (app lookup + create) so a wedged control plane cannot make
# the enter uncancellable. Creation is shielded so a normal cancellation cannot orphan a
# just-created sandbox (see `__aenter__`), but a shield with no deadline would hang forever if
# the RPC never returns. Generous, since its only job is to break a true hang: a cold start is
# well under this. If it fires after Modal already provisioned the sandbox, that sandbox is
# reaped server-side by its own `sandbox_timeout` -- the same backstop as any create leak.
_CREATE_TIMEOUT = 120


def _unavailable_sandbox_exc_types() -> tuple[type[BaseException], ...]:
    """Modal exception types that mean the sandbox itself no longer exists -- a terminal condition.

    A missing *file* is a different, recoverable error (`SandboxFilesystemNotFoundError`);
    these are the ones that say the whole sandbox is unusable.
    """
    import modal

    return (
        modal.exception.NotFoundError,
        modal.exception.SandboxTerminatedError,
        modal.exception.SandboxTimeoutError,
    )


# Modal does not currently expose a per-exec kill: a command is reaped by its own
# server-side timeout (or by the whole sandbox being terminated). So every command we run
# carries a deadline, even
# internal ones like the `pwd` used for path resolution, so a cancelled or abandoned run
# cannot leave a command billing indefinitely. This bounds that internal probe.
_INTERNAL_EXEC_TIMEOUT = 10

# The exit status of a process killed by SIGKILL (128 + 9): what a server-side deadline
# kill looks like when it beats Modal's client-side `-1` sentinel (see `exec`).
_SIGKILL_EXIT = 137

# Teardown runs shielded from cancellation, so an unreachable Modal control plane could
# otherwise hang the caller forever on exit. Bound each teardown RPC so a stalled
# terminate/detach gives up rather than wedging the process; the owned sandbox is still
# reaped server-side by its own `sandbox_timeout`.
_TEARDOWN_TIMEOUT = 30


# This is the mechanism layer: every Modal-specific operation (create/attach,
# exec, file access, path resolution, lifecycle) is contained here, behind a small
# byte-oriented method surface that the toolset depends on. Keeping it isolated from
# the presentation in `_toolset.py` is what lets the sandbox internals change without
# touching the tools or the capability.
class ModalSandboxSession:
    """Async context manager that owns or attaches to a Modal sandbox.

    In *owned* mode (the default) it creates a fresh sandbox from `image` on
    enter. On exit it requests termination and waits for a bounded period;
    `sandbox_timeout` is the server-side cleanup backstop. In *attach* mode
    (`sandbox_id` set) it looks
    up an existing sandbox and leaves it running on exit, so a sandbox you manage
    elsewhere can be reused across runs.

    Modal's SDK is asyncio-native, so this session drives its `.aio` coroutine API
    directly and requires an asyncio event loop. It authenticates the way the Modal
    CLI and SDK do: from the config written by `modal token new`, or from the
    `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` environment variables (which take
    precedence).

    ```python
    from pydantic_ai_harness.modal_sandbox import ModalSandboxSession

    async with ModalSandboxSession(image='python:3.12-slim') as session:
        result = await session.exec(['echo', 'hello'])
    ```
    """

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        sandbox_id: str | None = None,
        app_name: str = DEFAULT_APP_NAME,
        create_app_if_missing: bool = True,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if type(sandbox_timeout) is not int or sandbox_timeout <= 0:
            raise ValueError(f'sandbox_timeout must be a positive integer, got {sandbox_timeout!r}.')
        if sandbox_id is not None:
            conflicts = [
                name
                for name, value, default in (
                    ('image', image, DEFAULT_IMAGE),
                    ('app_name', app_name, DEFAULT_APP_NAME),
                    ('create_app_if_missing', create_app_if_missing, True),
                    ('sandbox_timeout', sandbox_timeout, DEFAULT_SANDBOX_TIMEOUT),
                    ('workdir', workdir, None),
                    ('env', env, None),
                )
                if value != default
            ]
            if conflicts:
                raise ValueError(
                    f'{", ".join(conflicts)} only apply when creating a sandbox, but `sandbox_id` attaches '
                    'to an existing one. Remove them, or drop `sandbox_id` to create a sandbox.'
                )
        self._image = image
        self._sandbox_id = sandbox_id
        self._app_name = app_name
        self._create_app_if_missing = create_app_if_missing
        self._sandbox_timeout = sandbox_timeout
        self._workdir = workdir
        self._env = dict(env) if env is not None else None
        self._sandbox: modal.Sandbox | None = None
        self._cwd: str | None = None
        # Serializes the one-time `pwd` probe so a batch of concurrent tool calls resolving
        # relative paths fires a single probe, not one per call (see `_resolve`).
        self._cwd_lock = anyio.Lock()

    @property
    def sandbox_id(self) -> str | None:
        """The id of the running sandbox, or None when it is not running."""
        if self._sandbox is None:
            return None
        return self._sandbox.object_id

    async def __aenter__(self) -> Self:
        """Create or attach to the sandbox."""
        if self._sandbox is not None:
            # Without this guard a second enter would overwrite the handle and orphan the
            # first owned sandbox (billed until its `sandbox_timeout`) with no error.
            raise ModalSandboxError(
                'The session is already open; exit it before entering again. '
                'Use a separate session per concurrent context.'
            )
        # Clear any cwd cached from a prior entry: a reused session must resolve relative
        # paths against the new sandbox's tree, not the previous one's.
        self._cwd = None
        try:
            import modal
        except ImportError as e:
            raise ModalSandboxError(_MISSING_MODAL) from e
        try:
            # Shield creation so a cancellation arriving mid-create cannot drop the sandbox
            # handle before we store it. Without this, an owned sandbox created server-side
            # would be orphaned (reaped only by its own `sandbox_timeout`) because `__aexit__`
            # would see no handle to terminate. The cold-start wait is brief, and we honor the
            # cancellation at the checkpoint just below. The inner deadline bounds the shielded
            # RPC so a wedged control plane cannot make this uncancellable (see `_CREATE_TIMEOUT`).
            # The shield holds for anyio-scope cancellation; a raw `asyncio.Task.cancel()` can
            # still interrupt it, in which case the server-side `sandbox_timeout` is the backstop.
            with anyio.CancelScope(shield=True):
                with anyio.move_on_after(_CREATE_TIMEOUT):
                    self._sandbox = await self._open_sandbox()
        except modal.exception.Error as e:
            raise self._open_error(e) from e
        if self._sandbox is None:
            # The deadline fired: the create RPC never returned. Fail here rather than proceed
            # with no sandbox. Any sandbox Modal provisioned before the hang is reaped by its
            # own `sandbox_timeout`, the same backstop as a create leak.
            raise ModalSandboxError(
                f'Modal sandbox creation did not complete within {_CREATE_TIMEOUT}s; '
                'the Modal control plane may be unreachable.'
            )
        try:
            # If the run was cancelled during the shielded create, this raises; tear the
            # just-created sandbox down here rather than leaving it for `sandbox_timeout`.
            await anyio.lowlevel.checkpoint()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def _open_sandbox(self) -> modal.Sandbox:
        """Create an owned sandbox or attach to an existing one."""
        import modal

        if self._sandbox_id is not None:
            sandbox = await modal.Sandbox.from_id.aio(self._sandbox_id)
            if await sandbox.poll.aio() is not None:
                raise ModalSandboxUnavailableError(
                    f'Could not attach to Modal sandbox {self._sandbox_id!r}: it does not exist or has terminated.'
                )
            return sandbox
        app = await modal.App.lookup.aio(self._app_name, create_if_missing=self._create_app_if_missing)
        # `from_registry` builds the image spec locally (no network), so it has no `.aio` variant.
        # Its typing uses an untyped `**kwargs`, so pyright flags the access.
        image = modal.Image.from_registry(self._image)  # pyright: ignore[reportUnknownMemberType]
        # Modal types env values as `str | None` (None unsets); widen our `dict[str, str]` to
        # match, since dict is invariant in its value type.
        env: dict[str, str | None] | None = (
            {key: value for key, value in self._env.items()} if self._env is not None else None
        )
        # `create.aio` is typed with a partially-`Any` coroutine return, so pyright flags the call.
        return await modal.Sandbox.create.aio(  # pyright: ignore[reportUnknownMemberType]
            app=app, image=image, timeout=self._sandbox_timeout, workdir=self._workdir, env=env
        )

    async def __aexit__(self, *args: object) -> None:
        """Request termination when owned, then attempt to detach within bounded waits."""
        sandbox = self._sandbox
        self._sandbox = None
        self._cwd = None
        if sandbox is None:
            return
        owned = self._sandbox_id is None
        # Shield cleanup so cancellation does not interrupt the termination request. Stop a
        # sandbox we created; an attached one keeps running. Attempt detach in `finally` --
        # Modal's recommended cleanup -- even if terminating the owned sandbox fails.
        # (As in `__aenter__`, the shield holds for anyio-scope cancellation, not a raw
        # `asyncio.Task.cancel()`; the server-side `sandbox_timeout` covers that case.)
        with anyio.CancelScope(shield=True):
            try:
                if owned:
                    # Bound each RPC independently so a stalled terminate still lets detach run;
                    # a single shared deadline would cancel the detach the moment terminate hung.
                    with anyio.move_on_after(_TEARDOWN_TIMEOUT):
                        try:
                            await sandbox.terminate.aio(wait=True)
                        except Exception:
                            # Termination is best-effort. A sandbox that no longer exists is
                            # success, not an error: an owned run that outlived its
                            # `sandbox_timeout` self-terminates. Any other failure (control
                            # plane, transport) must not replace the exception unwinding
                            # through the `async with` body (an exception from `__aexit__`
                            # would mask it), and the server-side `sandbox_timeout` reaps the
                            # sandbox regardless.
                            pass
            finally:
                with anyio.move_on_after(_TEARDOWN_TIMEOUT):
                    try:
                        await sandbox.detach.aio()  # pyright: ignore[reportUnknownMemberType]
                    except Exception:
                        # Best-effort like terminate: a failed local detach must not replace
                        # the exception unwinding through the body.
                        pass

    def _require_sandbox(self) -> modal.Sandbox:
        sandbox = self._sandbox
        if sandbox is None:
            raise ModalSandboxError('The sandbox is not running; use the session as an async context manager.')
        return sandbox

    def _unavailable_message(self) -> str:
        # In attach mode the real lifetime belongs to whoever created the sandbox; this
        # session's `sandbox_timeout` is pinned to its default there, so quoting it (or
        # advising to raise it, which attach mode rejects) would mislead.
        if self._sandbox_id is not None:
            return (
                f'The attached Modal sandbox {self._sandbox_id!r} is no longer running '
                '(terminated, or expired at its configured lifetime). '
                'Attach to a live sandbox, or create a new one.'
            )
        return (
            'The Modal sandbox is no longer running (it may have reached its '
            f'sandbox_timeout of {self._sandbox_timeout}s, or been terminated). '
            'Start a new run, or raise sandbox_timeout for longer work.'
        )

    def _open_error(self, e: modal.exception.Error) -> ModalSandboxError:
        """Map a Modal error raised while creating or attaching to a sandbox.

        Rejected credentials and a missing/terminated sandbox are terminal (the toolset
        never reaches this -- open errors abort the run before tools run -- but the typed
        error still tells a direct session caller what went wrong); anything else is a
        plain create failure.
        """
        import modal

        if isinstance(e, modal.exception.AuthError):
            return ModalSandboxAuthError(_AUTH_MESSAGE)
        if self._sandbox_id is not None and isinstance(e, _unavailable_sandbox_exc_types()):
            return ModalSandboxUnavailableError(
                f'Could not attach to Modal sandbox {self._sandbox_id!r}: it does not exist or has terminated.'
            )
        return ModalSandboxError(f'Could not start Modal sandbox: {e}')

    async def _ambiguous_error(self, e: modal.exception.Error) -> ModalSandboxError:
        """Classify a Modal error that may mask sandbox death by polling the sandbox.

        Two Modal layers report ambiguously: the filesystem wraps authentication
        failures as `SandboxFilesystemError` and transient control-plane failures as
        `NotFoundError`, and a first exec on a dead sandbox raises `ConflictError`
        (also used for transient aborts). Polling only after an error recovers the
        distinction without adding a round trip to successful operations.
        """
        import modal

        if isinstance(e, modal.exception.AuthError):
            return ModalSandboxAuthError(_AUTH_MESSAGE)
        sandbox = self._require_sandbox()
        try:
            returncode = await sandbox.poll.aio()
        except modal.exception.AuthError:
            return ModalSandboxAuthError(_AUTH_MESSAGE)
        except _unavailable_sandbox_exc_types():
            return ModalSandboxUnavailableError(self._unavailable_message())
        except Exception:
            # The classifying poll can itself fail, including with a raw transport error;
            # fall back to the original error rather than letting the probe abort the run.
            return ModalSandboxError(str(e))
        if returncode is not None:
            return ModalSandboxUnavailableError(self._unavailable_message())
        return ModalSandboxError(str(e))

    async def _resolve(self, path: str) -> str:
        """Resolve a possibly-relative path against the sandbox working directory.

        Modal's filesystem API only accepts absolute paths, while `run_command` runs
        in the sandbox working directory. Relative paths are joined with that directory
        -- queried once with `pwd` and cached -- so the file tools and shell commands
        share one view of the tree.
        """
        if posixpath.isabs(path):
            return path
        if self._cwd is None:
            # Single-flight the probe: a batch of concurrent tool calls resolving relative
            # paths all find `_cwd` unset, so without the lock each would run its own `pwd`.
            # The re-check inside the lock lets the losers use the winner's cached result.
            async with self._cwd_lock:
                if self._cwd is None:
                    result = await self.exec(['sh', '-c', 'pwd'], timeout=_INTERNAL_EXEC_TIMEOUT)
                    # Only cache a successful probe. A timeout (returncode -1) or error returns
                    # empty stdout; caching '/' from it would mis-resolve every later relative
                    # path with no error. Leave `_cwd` unset and fail this call so the next one
                    # probes again.
                    if result.returncode != 0:
                        raise ModalSandboxError(
                            'Could not determine the sandbox working directory to resolve a relative '
                            f'path ({path!r}); use an absolute path or retry.'
                        )
                    self._cwd = result.stdout.strip() or '/'
        if path in ('', '.'):
            return self._cwd
        return posixpath.join(self._cwd, path)

    async def exec(
        self, argv: Sequence[str], *, timeout: float | None = None, max_output_bytes: int | None = None
    ) -> ModalSandboxExecResult:
        """Run an argument vector in the sandbox (without a shell) and return its result.

        Modal does not currently expose a per-exec kill, so cancelling this coroutine stops
        us waiting for the command but does not stop the command: it keeps running until its
        `timeout` deadline (or until the sandbox itself is terminated). Pass a finite
        `timeout` so a cancelled or abandoned command cannot run on indefinitely;
        `timeout=None` leaves it unbounded, which is why the toolset always sets one.

        Args:
            argv: The command and its arguments.
            timeout: Per-command deadline in seconds, enforced server-side by Modal. None
                means no deadline (the command can outlive a cancellation).
            max_output_bytes: Cap on how much of each stream is retained in client memory.
                A command can print far more than the caller will ever show the model, so
                with this set only the last `max_output_bytes` bytes of each stream are kept
                exactly. A retained byte suffix can begin inside a multi-byte character and
                is decoded with replacement. None reads each stream in full -- fine for the
                small outputs of a direct session caller, but the toolset always sets it.
        """
        sandbox = self._require_sandbox()
        import modal

        if isinstance(argv, str):
            # A str is a Sequence[str] of characters, so 'ls -la' would splat into
            # one-character arguments; catch the mistake instead of running garbage.
            raise TypeError(f'argv must be a sequence of arguments, not a string; got {argv!r}.')
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
        if max_output_bytes is not None and (type(max_output_bytes) is not int or max_output_bytes <= 0):
            raise ValueError(f'max_output_bytes must be a positive integer or None, got {max_output_bytes!r}.')

        # Modal takes whole-second timeouts and treats 0 as "no timeout", so round a finite
        # request up and floor it at 1. Owning this here keeps the Modal quantization in the
        # mechanism layer: any caller passing a fractional or sub-second deadline still gets a
        # finite, Modal-legal one. The applied value rides back on ModalSandboxExecResult so the caller
        # can report the exact deadline without re-deriving it.
        deadline = None if timeout is None else max(1, math.ceil(timeout))
        # Time the command from before the exec RPC: the server's deadline clock starts at
        # exec start, so this client-side clock always reads at least as much elapsed time.
        # Used below to recognize a server-side deadline kill (see `timed_out`).
        started = time.monotonic()
        # Drain both streams and wait concurrently. They share the same server-side deadline,
        # so reading one to completion before starting the other can lose buffered output once
        # that deadline expires. Modal's text mode decodes strictly, so read bytes and decode here.
        try:
            process = await sandbox.exec.aio(*argv, timeout=deadline, text=False)
        except modal.exception.Error as e:
            raise await self._exec_error(e, 'Command could not run in the sandbox') from e

        stdout: tuple[str, bool] | None = None
        stderr: tuple[str, bool] | None = None
        returncode: int | None = None
        exited_after: float | None = None
        command_error: Exception | None = None

        # The readers catch Exception, not just modal's Error: stream iteration can surface
        # raw transport failures (grpclib stream errors, a ValueError on an empty message)
        # that are not modal.exception.Error, and an unmapped exception here would abort the
        # whole agent run instead of becoming a typed, retryable sandbox error.
        async def read_stdout() -> None:
            nonlocal stdout, command_error
            try:
                stdout = await self._read_stream(process.stdout, max_output_bytes)
            except Exception as e:
                command_error = e
                task_group.cancel_scope.cancel()

        async def read_stderr() -> None:
            nonlocal stderr, command_error
            try:
                stderr = await self._read_stream(process.stderr, max_output_bytes)
            except Exception as e:
                command_error = e
                task_group.cancel_scope.cancel()

        async def wait_for_exit() -> None:
            nonlocal returncode, exited_after, command_error
            try:
                returncode = await process.wait.aio()
                # Clock the exit here, not after the task group: the streams keep draining
                # concurrently, and drain time must not count toward the deadline window
                # (a slow drain would otherwise mislabel an early self-kill as a timeout).
                exited_after = time.monotonic() - started
            except Exception as e:
                command_error = e
                task_group.cancel_scope.cancel()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(read_stdout)
            task_group.start_soon(read_stderr)
            task_group.start_soon(wait_for_exit)
        if command_error is not None:
            raise await self._exec_error(
                command_error, 'Could not read the command result (the command may still run until its deadline)'
            ) from command_error
        if stdout is None or stderr is None or returncode is None or exited_after is None:  # pragma: no cover
            # Task-group completion contract: all four are set together on success.
            raise ModalSandboxError('Modal command result was incomplete.')
        # Modal's client reports `-1` when its local deadline kills the wait, but the server
        # enforces the same deadline first and its SIGKILL can win the race, surfacing as a
        # plain 137 exit. Read 137 as a timeout only when a deadline was set and the command
        # actually consumed its window, so a command that killed itself early stays a real exit.
        timed_out = deadline is not None and (
            returncode == -1 or (returncode == _SIGKILL_EXIT and exited_after >= deadline)
        )
        return ModalSandboxExecResult(
            stdout=stdout[0],
            stderr=stderr[0],
            returncode=returncode,
            stdout_truncated=stdout[1],
            stderr_truncated=stderr[1],
            timed_out=timed_out,
            applied_timeout=deadline,
        )

    async def _exec_error(self, e: Exception, context: str) -> ModalSandboxError:
        """Map an exception from running a command to a ModalSandbox error.

        A `ConflictError` is ambiguous (first exec on a dead sandbox, or a transient
        abort), so it is classified by polling. A terminated or missing sandbox and
        rejected credentials are terminal -- retrying cannot help, so the toolset ends
        the run instead of prompting the model to try again. Everything else -- another
        Modal error or a non-Modal transport failure -- stays a recoverable
        `ModalSandboxError`. `context` distinguishes "the command never started" from
        "the result could not be read", so the model is warned when the command may
        still be running.
        """
        import modal

        if isinstance(e, modal.exception.ConflictError):
            return await self._ambiguous_error(e)
        if isinstance(e, modal.exception.AuthError):
            return ModalSandboxAuthError(_AUTH_MESSAGE)
        if isinstance(e, _unavailable_sandbox_exc_types()):
            return ModalSandboxUnavailableError(self._unavailable_message())
        if isinstance(e, modal.exception.Error):
            return ModalSandboxError(f'{context}: {e}')
        return ModalSandboxError(f'{context}: {type(e).__name__}: {e}')

    @staticmethod
    async def _read_stream(
        stream: modal.io_streams.StreamReader[bytes], max_output_bytes: int | None
    ) -> tuple[str, bool]:
        """Drain a Modal exec stream, optionally retaining only its last `max_output_bytes`.

        Unbounded (`max_output_bytes is None`) reads the whole stream in one call. Bounded
        keeps the most recent bytes under the exact cap, including when one transport chunk
        is larger than the cap. The newest output -- where a command's error and exit status
        sit -- survives. Retained bytes are decoded as UTF-8 with replacement after selection.
        Returns the decoded text and whether any bytes were dropped, so the caller can mark
        the cut even when the retained text fits its own presentation caps.
        """
        if max_output_bytes is None:
            return (await stream.read.aio()).decode('utf-8', errors='replace'), False
        retained = bytearray()
        truncated = False
        async for chunk in stream:
            retained.extend(chunk)
            excess = len(retained) - max_output_bytes
            if excess > 0:
                truncated = True
                del retained[:excess]
        return bytes(retained).decode('utf-8', errors='replace'), truncated

    async def file_size(self, path: str) -> int:
        """Return a file's size in bytes via Modal's filesystem API, without reading it.

        Lets a caller check size before reading the whole file. A relative `path` is resolved
        against the sandbox working directory (see `_resolve`).

        Raises:
            ModalSandboxError: if the file cannot be stat-ed (missing, a directory, ...).
        """
        sandbox = self._require_sandbox()
        import modal

        target = await self._resolve(path)
        # Catch `Error`, not just `SandboxFilesystemError`: a transient connection or auth
        # failure raises a plain Modal `Error` here too, and it must surface as a
        # ModalSandboxError (a retryable tool error) rather than leak raw to the agent loop.
        try:
            info = await sandbox.filesystem.stat.aio(target)
        except modal.exception.Error as e:
            raise await self._ambiguous_error(e) from e
        return info.size

    async def read_bytes(self, path: str) -> bytes:
        """Read a file's raw bytes from the sandbox via Modal's filesystem API.

        The session deals in bytes so each tool layer can decode (or not) as it needs;
        text handling lives above the session, not here. A relative `path` is resolved
        against the sandbox working directory (see `_resolve`).

        Raises:
            ModalSandboxError: if the file cannot be read (missing, a directory, ...).
        """
        sandbox = self._require_sandbox()
        import modal

        target = await self._resolve(path)
        try:
            return await sandbox.filesystem.read_bytes.aio(target)
        except modal.exception.Error as e:
            raise await self._ambiguous_error(e) from e

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write raw bytes to a file in the sandbox, creating parent directories.

        A relative `path` is resolved against the sandbox working directory (see
        `_resolve`). Unlike shelling out, Modal's filesystem API streams the content,
        so the size is not bounded by the argument-length limit of a command, and it
        creates missing parent directories itself.

        Raises:
            ModalSandboxError: if the file cannot be written (bad path, permissions, ...).
        """
        sandbox = self._require_sandbox()
        import modal

        target = await self._resolve(path)
        try:
            await sandbox.filesystem.write_bytes.aio(data, target)
        except modal.exception.Error as e:
            raise await self._ambiguous_error(e) from e

    async def list_files(self, path: str) -> list[tuple[str, bool]]:
        """List a sandbox directory as `(name, is_dir)` pairs.

        A relative `path` is resolved against the sandbox working directory (see
        `_resolve`). The Modal-native `FileInfo` entries are normalized to plain tuples
        here so the provider type does not leak past the session.

        Raises:
            ModalSandboxError: if the directory cannot be listed.
        """
        sandbox = self._require_sandbox()
        import modal

        target = await self._resolve(path)
        try:
            entries = await sandbox.filesystem.list_files.aio(target)
        except modal.exception.Error as e:
            raise await self._ambiguous_error(e) from e
        return [(entry.name, entry.is_dir()) for entry in entries]
