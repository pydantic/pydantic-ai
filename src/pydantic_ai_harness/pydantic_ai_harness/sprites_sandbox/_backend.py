"""Fly.io Sprites backend for Pydantic AI's `WorkspaceBackend` protocol.

External assumptions last verified 2026-09-25 against sprites-py 0.7.0 source, the Sprites API
docs, and (2026-09-15) a local WebSocket transport probe, with no live cloud calls:

* `AsyncSpritesClient` requires its token as an argument and reads no environment variable; sprite
  creation uses the SDK's fixed 120-second request timeout, while `aclose` closes the local async
  HTTP client and control pools:
  https://github.com/superfly/sprites-py/blob/v0.7.0/src/sprites/async_client.py
* `create_sprite` and `get_sprite` raise `AuthenticationError` (401), `NotFoundError` (404),
  `NetworkError` (transport), and a plain `SpriteError` for any other HTTP failure:
  https://github.com/superfly/sprites-py/blob/v0.7.0/src/sprites/client.py
* Commands run over the documented exec WebSocket through the SDK's `WSCommand`, the SDK's own
  default path. It sends stdin EOF when no stdin is given, reads the exit status from the binary
  EXIT frame or the JSON `exit` message, raises `NetworkError` when the socket closes before either,
  and turns a failed handshake into a parsed `APIError` carrying the HTTP status. The working
  directory goes in the `dir` query parameter as the SDK sends it; the API page lists `dir` only
  for the HTTP exec endpoint:
  https://github.com/superfly/sprites-py/blob/v0.7.0/src/sprites/websocket.py
* The exec API starts the command as soon as the request arrives, before the client's output
  stream is attached, and replays only the last 16 or 64 KiB printed before then: the rest was lost
  with exit status 0 (observed against real Sprites on 2026-09-26). So every command waits for the
  client's stdin EOF, which the SDK sends once the socket is open, before it starts.
* The exec API takes argv in the WebSocket URL, which the Sprite refuses (HTTP 414) above about
  40 KB, so file writes go through the filesystem API (`PUT /fs/write`) instead. It writes through
  a symlink, creates missing parents, owns the file to the Sprite's user, and sets the mode it is
  given, replacing an existing file's; a 404 means either a missing path or a deleted Sprite
  (observed 2026-09-26):
  https://github.com/superfly/sprites-py/blob/v0.7.0/src/sprites/async_filesystem.py
* The multiplexed control protocol (`sprites.control`, used only in the SDK's opt-in control mode)
  is not used: in 0.7.0 its `op.complete` handler overwrites the exit status from the EXIT frame
  with the message's own `exitCode`, which defaults to 0, so every command reported success
  (observed against real Sprites on 2026-09-25):
  https://github.com/superfly/sprites-py/blob/v0.7.0/src/sprites/control.py
* The exec API documents that a set `env` replaces the default environment, so the backend does
  not pass `env`; it runs the command under the POSIX `env` utility instead, which adds the
  variables to the Sprite's own environment:
  https://sprites.dev/api/sprites/exec
* A disconnect does not stop a non-TTY command at once; it keeps running for
  `max_run_after_disconnect` (10 seconds by default; `0` means no limit, as the TTY default shows).
  The backend asks for one second, so closing the socket on a timeout or cancellation ends the
  command shortly after:
  https://sprites.dev/api/sprites/exec
* Provider retention is separate from local client disconnect:
  https://docs.sprites.dev/concepts/lifecycle/

Re-check these sources, the installed signatures, and the local transport probe before changing
lifecycle or command transport behavior. The integration uses the SDK's native asyncio client.
"""

from __future__ import annotations

# Native task cancellation can interrupt an AnyIO shield; the completion task below must stay independent.
import asyncio
import logging
import math
import os
import posixpath
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import TypeVar

import anyio
import httpx
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai.workspaces.workspace import _ShellFilesystem  # pyright: ignore[reportPrivateUsage]
from websockets.exceptions import InvalidHandshake, InvalidMessage

from pydantic_ai_harness._workspace_provider import absolute_path, command_argv, safe_credential_reason, stop_shielded

try:
    from sprites import AsyncSprite, AsyncSpritesClient
    from sprites.exceptions import (
        APIError,
        AuthenticationError,
        FileNotFoundError_,
        FilesystemError,
        IsADirectoryError_,
        NetworkError,
        NotADirectoryError_,
        NotFoundError,
        SpriteError,
    )
    from sprites.exceptions import TimeoutError as SpriteTimeoutError
    from sprites.websocket import WSCommand
except ImportError as exc:  # pragma: no cover - exercised by the isolated missing-extra test
    raise ImportError('Install `pydantic-ai-harness[sprites]` to use SpritesSandbox.') from exc

logger = logging.getLogger(__name__)
_T = TypeVar('_T')
_CLOSE_TIMEOUT = 6.0
# Above the SDK's fixed 120-second creation request timeout, so the SDK's own error wins when it fires.
_ACQUIRE_TIMEOUT = 150.0
# Bounds the internal `pwd` probe behind `working_dir()`, which may first wake a sleeping Sprite.
_INTERNAL_EXEC_TIMEOUT = 30
_AUTH_MESSAGE = (
    'Sprites rejected the credentials. Set SPRITE_TOKEN, or pass a configured `AsyncSpritesClient` as `client=`.'
)


async def _cleanup_call(call: Callable[[], Awaitable[object]], *, timeout: float) -> Exception | None:
    """Run one teardown RPC shielded from cancellation and bounded by `timeout`.

    Returns the failure instead of raising so the caller owns translation; a bare `TimeoutError`
    means the bound expired. Shielded because teardown must still go out while a run is being
    cancelled; bounded so a wedged control plane cannot hang teardown.
    """
    error: Exception | None = None
    with anyio.move_on_after(timeout, shield=True) as scope:
        try:
            await call()
        except Exception as exc:
            error = exc
    if scope.cancel_called:
        return TimeoutError()
    return error


async def _run_to_completion(call: Callable[[], Awaitable[_T]]) -> _T:
    """Await `call` in a task of its own and see it finish even if the caller is cancelled meanwhile.

    For work whose outcome must be recorded, such as a created Sprite or a closed client: the
    caller's cancellation is re-raised only once `call` has finished, and nothing outlives this
    await. `call` must be bounded; the caller waits for it.
    """
    # AnyIO has no detached task: keep this native task alive and referenced until its result is recorded.
    task = asyncio.ensure_future(call())
    # Preserve the native cancellation to re-raise after the provider call finishes.
    cancelled: asyncio.CancelledError | None = None
    # The AnyIO shield holds off cancel scopes, but native `Task.cancel()` can still interrupt the
    # caller. `asyncio.wait` does not propagate that cancellation to the independent provider task.
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.wait([task])
            except asyncio.CancelledError as error:
                cancelled = error
    if cancelled is not None:
        task.exception()  # Retrieve a provider failure before the native cancellation takes precedence.
        raise cancelled
    return task.result()


class _ExecCommand(WSCommand):
    """The SDK's exec WebSocket command, asking the Sprite to end the command soon after a disconnect."""

    def _build_websocket_url(self) -> str:
        # Stdin stays on: `_ending_with` waits for its EOF before starting the command. The stream is
        # kept plain (no TTY). A non-TTY command keeps running for 10 seconds after its socket closes
        # unless told otherwise, and `0` means no limit. One second makes closing the socket on a
        # timeout or cancellation stop the command.
        return f'{super()._build_websocket_url()}&tty=false&max_run_after_disconnect=1s'


async def _close_command(command: WSCommand) -> None:
    """Close an exec WebSocket; if that fails, abort its socket so nothing is left open.

    Finished even when the caller (a cancelled command) is cancelled meanwhile.
    """
    error = await _run_to_completion(lambda: _cleanup_call(command.close, timeout=_CLOSE_TIMEOUT))
    if error is not None:
        logger.warning('Could not close a Sprite exec connection, aborting it: %r', error)
        # `close` has nothing to fail on before `start` opened the socket.
        if command.ws is not None:  # pragma: no branch
            command.ws.transport.abort()


def _map_error(error: Exception, sprite_name: str | None) -> WorkspaceError | None:
    """Translate a Sprites failure, or return `None` for one that propagates as is.

    `sprite_name` is `None` while the Sprite is being created. Rejected credentials, a refused
    creation, and a missing Sprite end the run; any other request the API refused is a failed
    operation. Pre-connection transport failures, rate limits, server errors, and anything unknown
    propagate unchanged, for durable engines to retry.
    """
    # The exec handshake reports its HTTP status as an `APIError`.
    status = error.status_code if isinstance(error, APIError) else None
    if isinstance(error, AuthenticationError) or status == 401:
        # SDK error text may contain the rejected token; keep only a classified reason.
        return WorkspaceUnavailableError(f'{safe_credential_reason(error)}. {_AUTH_MESSAGE}')
    if not isinstance(error, SpriteError) or isinstance(error, (NetworkError, SpriteTimeoutError)):
        return None
    # sprites-py reports every other HTTP failure, a rate limit or a server error included, as a plain
    # `SpriteError` whose message names the status.
    if re.search(r'\(status (429|5\d\d)\)', str(error)) or (status is not None and (status == 429 or status >= 500)):
        return None
    if sprite_name is None:
        # An unknown runtime or a bad request fails the same way on every retry.
        return WorkspaceUnavailableError(f'Could not start Sprites sandbox: {error}')
    if isinstance(error, NotFoundError) or status == 404:
        return WorkspaceUnavailableError(
            f'The Sprite {sprite_name!r} no longer exists: it was deleted. '
            "Pass `workspace='new'` to start a fresh sandbox."
        )
    return WorkspaceError(f'Sprites refused the request: {error}')


async def _check_sigkill_sprite(sprite: AsyncSprite) -> None:
    # A user's SIGKILL also exits 137; only a control-plane 404 proves the Sprite died.
    # A stalled or failing lookup must not replace the command's real result.
    try:
        with anyio.fail_after(2):
            await sprite.client.get_sprite(sprite.name)
    except NotFoundError as error:
        raise WorkspaceUnavailableError(
            f'The Sprite {sprite.name!r} no longer exists: it was deleted. '
            "Pass `workspace='new'` to start a fresh sandbox."
        ) from error
    except Exception:
        pass


class SpritesSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem):
    """A Fly.io Sprite behind the Pydantic AI `WorkspaceBackend` protocol.

    Pass `sandbox=` to wrap a `sprites.AsyncSprite` you already have, or `ref=` to reattach to one.
    Construction does no I/O. The typed `sprites.AsyncSprite` is available through `get_sandbox()`.
    Without `client=`, the backend creates an `AsyncSpritesClient` from `SPRITE_TOKEN` on first use
    and closes it in `aclose()`.
    The backend does not delete the Sprite; that is the application's job, through the native
    handle. Commands run under `/bin/sh -c` with `shell=True`, in the Sprite's own environment
    plus `env`. File writes go through the Sprite's filesystem API; the other file operations
    run as shell commands.
    """

    def __init__(
        self,
        *,
        sandbox: AsyncSprite | None = None,
        client: AsyncSpritesClient | None = None,
        ref: WorkspaceRef | None = None,
        runtime: str | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None and ref.provider != 'sprites':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'sprites'")
        if sandbox is not None and ref is not None:
            raise ValueError('pass either `sandbox` or `ref`, not both')
        self._sandbox = sandbox
        self._ref = ref if sandbox is None else WorkspaceRef(provider='sprites', id=sandbox.name)
        self._new_sprite_name = f'pydantic-ai-{uuid.uuid4().hex}'
        self._uncertain_create = False
        self._runtime = runtime
        self._working_dir = absolute_path('working_dir', working_dir)
        self._env = dict(env or {})
        # `working_dir` as the Sprite resolves it (`pwd -P`): the protocol reports a canonical absolute path.
        self._resolved_working_dir: str | None = None
        self._client = client
        self._owns_client = client is None
        self._lock = anyio.Lock()

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    async def get_sandbox(self) -> AsyncSprite:
        """Return the typed `sprites.AsyncSprite`, creating or attaching to it on first use.

        This is the Sprite handle, not the `AsyncSpritesClient` passed as `client=`. A handle
        obtained before `aclose()` belongs to the closed client; call this again for a fresh one.

        The lock serializes concurrent first uses -- two callers each creating a Sprite
        would leave the loser billed and unreferenced. Attaching by `ref` to a Sprite that
        no longer exists raises `WorkspaceUnavailableError`; it does not create a replacement.
        """
        async with self._lock:
            if (sandbox := self._sandbox) is not None:
                return sandbox

            client = self._client
            if client is None:
                # The SDK reads no environment variable, so the token comes from `SPRITE_TOKEN` here.
                token = os.getenv('SPRITE_TOKEN')
                if not token:
                    raise WorkspaceUnavailableError(_AUTH_MESSAGE)
                client = AsyncSpritesClient(token=token)
                self._client = client

            ref = self._ref

            async def acquire() -> AsyncSprite:
                with anyio.move_on_after(_ACQUIRE_TIMEOUT):
                    if ref is not None:
                        try:
                            sandbox = await client.get_sprite(ref.id)
                        except NotFoundError as error:
                            if self._uncertain_create:
                                # A 404 during eventual visibility is not proof that creation failed.
                                raise NetworkError(f'Sprite {ref.id!r} may still be becoming visible') from error
                            raise
                    else:
                        try:
                            sandbox = await client.create_sprite(self._new_sprite_name, runtime=self._runtime)
                        except (NetworkError, TimeoutError, SpriteError) as error:
                            if isinstance(error, SpriteError) and not (
                                isinstance(error, NetworkError) or '(status 409)' in str(error)
                            ):
                                raise
                            # The create may have committed even if lookup is not yet visible. Keep its
                            # preallocated name for failure hooks and use lookup only on future attempts.
                            self._ref = WorkspaceRef(provider='sprites', id=self._new_sprite_name)
                            self._uncertain_create = True
                            try:
                                sandbox = await client.get_sprite(self._new_sprite_name)
                            except (NotFoundError, NetworkError, TimeoutError):
                                raise error from None
                    # Recorded as soon as the SDK returns, so a cancelled caller still leaves it named.
                    self._sandbox = sandbox
                    self._ref = WorkspaceRef(provider='sprites', id=sandbox.name)
                    self._uncertain_create = False
                    return sandbox
                # Only our own bound lands here; an SDK `TimeoutError` propagates as raised. A stalled
                # control plane is a transport failure, which propagates for a retry;
                # `WorkspaceTimeoutError` is reserved for command deadlines.
                if ref is None:
                    # A timed-out create may have committed; a subsequent call must only look it up.
                    self._ref = WorkspaceRef(provider='sprites', id=self._new_sprite_name)
                    self._uncertain_create = True
                action = 'creation' if ref is None else 'connection'
                raise TimeoutError(
                    f'Sprite {action} did not complete within {_ACQUIRE_TIMEOUT:g}s; '
                    'the Sprites control plane may be unreachable.'
                )

            try:
                # Creation runs to completion even if the caller is cancelled, so a Sprite that
                # was created is never left unnamed; attaching creates nothing and stays cancellable.
                return await (acquire() if ref is not None else _run_to_completion(acquire))
            except SpriteError as error:
                if (mapped := _map_error(error, None if ref is None else ref.id)) is None:
                    raise
                raise mapped from error

    async def aclose(self) -> None:
        """Close the `AsyncSpritesClient` this backend created, if it created one.

        The Sprite is untouched: the next operation opens a fresh client and reattaches by `ref`.
        A caller-supplied `client=` or `sandbox=` handle is never closed. `SpritesSandbox`
        calls this for the backend it supplied when each run ends. A close that fails or times out
        is logged, not raised, and the client is kept so the next `aclose()` tries again.
        """
        if not self._owns_client:
            return

        async def close() -> None:
            # Under the lock, so a Sprite still being created is recorded before its client closes.
            async with self._lock:
                client = self._client
                if client is None:
                    return
                error = await _cleanup_call(client.aclose, timeout=_CLOSE_TIMEOUT)
                if error is not None:
                    # Kept, so a later `aclose()` tries again.
                    logger.warning('Could not close Sprites SDK client: %r', error)
                else:
                    self._client = None
                    self._sandbox = None

        # Finished even when the caller (a run being cancelled) is cancelled meanwhile.
        await _run_to_completion(close)

    async def working_dir(self) -> str:
        if self._resolved_working_dir is None:
            result = await self.run(['pwd', '-P'], timeout=_INTERNAL_EXEC_TIMEOUT)
            printed = result.stdout.removesuffix('\n')
            if result.exit_code != 0 or not posixpath.isabs(printed):
                sandbox = await self.get_sandbox()
                raise WorkspaceError(
                    f'Could not determine the working directory of Sprite {sandbox.name!r}: '
                    f'`pwd -P` exited {result.exit_code} and printed {result.stdout!r}. Use absolute paths.'
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
        _capture_stderr: bool = True,
        _check_cwd: bool = True,
    ) -> CommandResult:
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
        directory = absolute_path('cwd', cwd) if cwd is not None else self._working_dir
        marker = f'pydantic-ai-end-{uuid.uuid4().hex}'
        capture = f'/tmp/pydantic-ai-stderr-{uuid.uuid4().hex}'
        # `env` runs inside the wrapper: a `sh` such as dash drops variables whose names are not shell
        # identifiers from the environment it passes on.
        args = _ending_with(marker, capture, _with_env(command_argv(command, shell), {**self._env, **(env or {})}))

        # Acquiring the Sprite has its own bound; the deadline is the command's alone.
        sandbox = await self.get_sandbox()
        if directory is not None and _check_cwd:
            # Sprites exec silently ignores a nonexistent `dir`; reject it before running user work.
            check = await self.run(['test', '-d', directory], timeout=_INTERNAL_EXEC_TIMEOUT, _check_cwd=False)
            if check.exit_code != 0:
                raise FileNotFoundError(directory)
        exec_command = _ExecCommand(sandbox.command(*args, cwd=directory))
        deadline = anyio.CancelScope(deadline=math.inf if timeout is None else anyio.current_time() + timeout)
        code = -1
        interrupted = False
        try:
            with deadline:
                try:
                    await exec_command.start()
                except (TimeoutError, InvalidMessage, InvalidHandshake) as error:
                    # Stdin EOF gates execution: an unsuccessful handshake with no socket cannot
                    # have started the command. Never retry after the connection is established.
                    if exec_command.ws is not None:
                        # An opened socket can have started the command even when start() fails.
                        raise WorkspaceUnavailableError(
                            'Sprites exec connection failed; command may have run'
                        ) from error
                    await anyio.sleep(0.1)
                    exec_command = _ExecCommand(sandbox.command(*args, cwd=directory))
                    try:
                        await exec_command.start()
                    except (TimeoutError, InvalidMessage, InvalidHandshake) as retry_error:
                        if exec_command.ws is not None:
                            raise WorkspaceUnavailableError(
                                'Sprites exec connection failed; command may have run'
                            ) from retry_error
                        raise NetworkError(f'Sprites exec handshake failed: {retry_error}') from retry_error
                try:
                    code = await exec_command.wait()
                except NetworkError as error:
                    # The socket opened: losing EXIT cannot prove the command did not execute.
                    raise WorkspaceUnavailableError('Sprites exec connection failed; command may have run') from error
            if timeout is not None and deadline.cancelled_caught:
                interrupted = True
                await _close_command(exec_command)
                partial = _split_output(exec_command.get_stdout(), exec_command.get_stderr(), marker)
                stderr = await self._collect_stderr(capture) if _capture_stderr else ''
                raise WorkspaceTimeoutError(
                    f'Command timed out after {timeout:g} seconds', stdout=partial[0], stderr=stderr + partial[1]
                )
        except BaseException as error:
            # On a timeout or a cancellation, closing the socket is what ends the command in the Sprite.
            if not interrupted:
                await _close_command(exec_command)
                if _capture_stderr:
                    await self._collect_stderr(capture)
            if isinstance(error, Exception) and (mapped := _map_error(error, sandbox.name)) is not None:
                raise mapped from error
            raise
        await _close_command(exec_command)
        stdout, stderr = _split_output(exec_command.get_stdout(), exec_command.get_stderr(), marker)
        if code == 137:
            await _check_sigkill_sprite(sandbox)
        return CommandResult(exit_code=code, stdout=stdout, stderr=stderr)

    async def _collect_stderr(self, path: str) -> str:
        output = ''

        async def collect() -> None:
            nonlocal output
            # Bounded to avoid moving an unbounded stderr capture through the exec URL/output buffer.
            result = await self.run(
                ['sh', '-c', 'head -c 65536 -- "$1"; rm -f -- "$1"', 'sh', path],
                timeout=1,
                _capture_stderr=False,
                _check_cwd=False,
            )
            output = result.stdout

        try:
            await stop_shielded(collect)
        except Exception:
            logger.warning('Could not retrieve Sprite stderr capture')
        return output

    async def write_bytes(self, path: str, data: bytes) -> None:
        # Not through a command: the exec API sends argv in the URL, which caps a command at about 40 KB.
        sandbox = await self.get_sandbox()
        target = sandbox.filesystem() / path
        try:
            mode = 0o644
            try:
                current = await target.stat()
            except FileNotFoundError_:
                pass
            else:
                # The API sets the mode it is given, so an existing file keeps its own (an executable
                # stays one). For a directory `stat` reports an entry inside it; the write refuses it.
                if current.path == path and not current.is_dir:
                    mode = int(current.mode, 8)
            await target.write_bytes(data, mode=mode)
        except IsADirectoryError_ as error:
            raise IsADirectoryError(path) from error
        except NotADirectoryError_ as error:
            raise NotADirectoryError(path) from error
        except FileNotFoundError_ as error:
            # Missing parents are created, so a 404 here means the Sprite itself is gone: a command
            # reports that as `WorkspaceUnavailableError`.
            await self.run(['true'], timeout=_INTERNAL_EXEC_TIMEOUT)
            raise WorkspaceError(f'Sprites could not write {path!r}: {error}') from error
        except FilesystemError as error:
            if isinstance(error.__cause__, httpx.RequestError):
                raise  # A transport failure propagates, for durable engines to retry.
            raise WorkspaceError(f'Sprites refused writing {path!r}: {error}') from error

    # The other file operations are the ones Pydantic AI derives from `run` for a command-only backend.
    async def read_bytes(self, path: str) -> bytes:
        return await _ShellFilesystem(self).read_bytes(path)

    async def stat(self, path: str) -> FileEntry:
        return await _ShellFilesystem(self).stat(path)

    async def list_dir(self, path: str) -> tuple[FileEntry, ...]:
        return await _ShellFilesystem(self).list_dir(path)

    async def make_dir(self, path: str) -> None:
        await _ShellFilesystem(self).make_dir(path)

    async def remove(self, path: str) -> None:
        await _ShellFilesystem(self).remove(path)

    async def exists(self, path: str) -> bool:
        return await _ShellFilesystem(self).exists(path)


def _ending_with(marker: str, capture: str, args: list[str]) -> list[str]:
    """`args` run under a `sh` that reports their stdout, a `marker` line, then their stderr, all on stdout.

    The `sh` first reads stdin to its EOF, which the client sends once its socket is open: output
    printed before the client's stream attaches is lost, beyond a short replay.

    The live Sprite's stderr stream is not dependable: the same command's stderr arrived on the stderr
    stream in one run and on the stdout stream in the next, whole lines included (2026-09-25), while
    stdout arrived intact every time. So the command's stderr goes to a temporary file in the Sprite,
    printed on stdout after the marker line, and `_split_output` separates the two again. The file is
    removed when the command finishes; one stopped by a timeout or a cancellation can leave it in
    `/tmp`, but every command gets a fresh name, so no later command reads it. The exit status is the
    command's.
    """
    script = (
        'cat >/dev/null; err=$1; shift; : >"$err" || exit 125; '
        f'"$@" 2>"$err"; status=$?; printf "\\n%s\\n" {marker}; cat "$err"; rm -f "$err"; exit "$status"'
    )
    return ['sh', '-c', script, 'sh', capture, *args]


def _split_output(stdout: bytes, stderr: bytes, marker: str) -> tuple[str, str]:
    """The command's stdout and stderr from what `_ending_with` printed.

    Without the marker line (the command was stopped before it finished), stdout is all the output
    there is, and the command's stderr is lost with its unlinked file. Anything on the stderr stream itself came
    from the wrapper and is kept.
    """
    head, _, tail = _decode(stdout).partition(f'\n{marker}\n')
    return head, tail + _decode(stderr)


def _with_env(args: list[str], env: dict[str, str]) -> list[str]:
    """`args` run under the POSIX `env` utility, which adds `env` to the Sprite's own environment."""
    if not env:
        return args
    for key, value in env.items():
        if not key or '=' in key or '\0' in key or '\0' in value:
            raise ValueError(
                f'illegal environment variable {key!r}: a name is non-empty without "=" or NUL, a value without NUL'
            )
    # `--` ends `env`'s options, so a name starting with `-` is not read as one.
    return ['env', '--', *(f'{key}={value}' for key, value in env.items()), *args]


def _decode(data: bytes) -> str:
    return data.decode('utf-8', errors='replace')
