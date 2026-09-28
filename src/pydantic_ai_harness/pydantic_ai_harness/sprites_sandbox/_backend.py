"""Fly.io Sprites backend for Pydantic AI's `WorkspaceBackend` protocol.

External assumptions last verified against sprites-py 0.7.0 source, the Sprites API docs, a local
WebSocket transport probe (2026-09-15), and the live service on the dates given below:

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
  for the HTTP exec endpoint. A `dir` that does not exist fails the exec with exit status 1 and a
  `chdir` message on stdout, without starting the command (observed 2026-09-28):
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
lifecycle or command transport behavior. The integration uses the SDK's native asyncio client, so
it runs on asyncio only.
"""

from __future__ import annotations

# Native task cancellation can interrupt an AnyIO shield; the completion task below must stay independent.
import asyncio
import hashlib
import logging
import math
import os
import posixpath
import re
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TypeVar

import anyio
import anyio.to_thread
import httpx
import sniffio
from websockets.exceptions import InvalidHandshake, InvalidMessage

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai.workspaces.workspace import _ShellFilesystem  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_harness._workspace_provider import absolute_path, command_argv, safe_credential_reason, stop_shielded

try:
    from sprites import AsyncCmd, AsyncSprite, AsyncSpritesClient
    from sprites.exceptions import (
        APIError,
        AuthenticationError,
        FileNotFoundError_,
        FilesystemError,
        IsADirectoryError_,
        NetworkError,
        NotADirectoryError_,
        NotFoundError,
        PermissionError_,
        SpriteError,
        TimeoutError as SpriteTimeoutError,
    )
    from sprites.websocket import WSCommand
except ImportError as exc:
    raise ImportError('Install `pydantic-ai-harness[sprites]` to use SpritesSandbox.') from exc

logger = logging.getLogger(__name__)
_T = TypeVar('_T')
_CLOSE_TIMEOUT = 6.0
# Above the SDK's fixed 120-second creation request timeout, so the SDK's own error wins when it fires.
_ACQUIRE_TIMEOUT = 150.0
# Seconds before each retry of an exec handshake that failed before its socket opened: three
# attempts over about five seconds. A single retry after 0.1 seconds ended 4 of 14 live eval runs.
_HANDSHAKE_RETRY_DELAYS = (1.0, 4.0)
# Bounds the internal `pwd` probe behind `working_dir()`, which may first wake a sleeping Sprite.
_INTERNAL_EXEC_TIMEOUT = 30
# Combined output ceiling and error preview size, as in Pydantic AI's local backend.
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_PREVIEW_BYTES = 64 * 1024
_AUTH_MESSAGE = (
    'Sprites rejected the credentials. Set SPRITE_TOKEN, or pass a configured `AsyncSpritesClient` as `client=`.'
)
# Nothing was sent, so nothing was rejected.
_MISSING_TOKEN_MESSAGE = (
    'No Sprites credentials found. Set SPRITE_TOKEN, or pass a configured `AsyncSpritesClient` as `client=`.'
)


# Seconds a successful lookup is trusted by later backends in this process.
_LOOKUP_TTL = 60.0


@dataclass
class _Lookup:
    expires_at: float


class _LookupCache:
    """Sprites that backends in this process recently found to exist.

    Under Temporal each activity builds a new backend, which would otherwise repeat the `get_sprite`
    lookup. The working directory is not cached: `working_dir()` is often an activity's first call,
    and running `pwd -P` is what reports a Sprite deleted elsewhere. Only plain data is kept, never a client or a Sprite handle: the
    SDK client wraps an `httpx.AsyncClient`, whose pooled connections belong to the event loop that
    opened them, and activities may run on other loops or threads. Keyed by credentials and API
    base URL too, as another token may not see the Sprite. Failures are never recorded, and an
    entry is dropped once an operation finds the Sprite unavailable or it is destroyed.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self._entries: dict[tuple[str, str, str], _Lookup] = {}
        # Backends on other threads' event loops share the cache.
        self._lock = threading.Lock()

    @staticmethod
    def key(client: AsyncSpritesClient, name: str) -> tuple[str, str, str]:
        # A digest, so the process-wide cache holds no token.
        return hashlib.sha256(client.token.encode()).hexdigest(), client.base_url, name

    def _live(self, key: tuple[str, str, str]) -> _Lookup | None:
        entry = self._entries.get(key)
        if entry is not None and entry.expires_at <= self.clock():
            del self._entries[key]
            return None
        return entry

    def exists(self, key: tuple[str, str, str]) -> bool:
        with self._lock:
            return self._live(key) is not None

    def record(self, key: tuple[str, str, str]) -> None:
        with self._lock:
            self._entries[key] = _Lookup(self.clock() + _LOOKUP_TTL)

    def forget(self, name: str) -> None:
        """Drop what any credentials learned about the Sprite `name`."""
        with self._lock:
            for key in [key for key in self._entries if key[2] == name]:
                del self._entries[key]


_lookups = _LookupCache()


def _require_asyncio() -> None:
    if sniffio.current_async_library() != 'asyncio':
        raise UserError('Sprites needs the asyncio event loop: the Sprites SDK runs its calls on asyncio tasks.')


async def _new_client() -> AsyncSpritesClient:
    """An `AsyncSpritesClient` for `SPRITE_TOKEN`, built off the event loop.

    The SDK reads no environment variable, so the token is read here. The client computes its
    headers on first construction, reading `/proc` on Linux.
    """
    token = os.getenv('SPRITE_TOKEN')
    if not token:
        raise WorkspaceUnavailableError(_MISSING_TOKEN_MESSAGE)
    return await anyio.to_thread.run_sync(lambda: AsyncSpritesClient(token=token))


async def destroy_sprite(client: AsyncSpritesClient | None, name: str) -> None:
    """Delete a Sprite by name without attaching or waking it.

    A Sprite that no longer exists returns quietly; rejected credentials raise
    `WorkspaceUnavailableError`, like any other operation. Without `client`, one is opened from
    `SPRITE_TOKEN` and closed afterwards.

    Raises:
        UserError: The event loop is not asyncio.
    """
    _require_asyncio()
    try:
        if client is not None:
            await client.destroy_sprite(name)
        else:
            async with await _new_client() as owned:
                await owned.destroy_sprite(name)
    except NotFoundError:
        return
    except AuthenticationError as error:
        raise WorkspaceUnavailableError(f'{safe_credential_reason(error)}. {_AUTH_MESSAGE}') from error
    finally:
        # Even a failed delete may have gone through, so no later backend skips its lookup.
        _lookups.forget(name)


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
    """The SDK's exec WebSocket command, asking the Sprite to end the command soon after a disconnect.

    It also stops reading once the output passes `_MAX_OUTPUT_BYTES`; `marker` is `_ending_with`'s.
    """

    def __init__(self, cmd: AsyncCmd, marker: str) -> None:
        super().__init__(cmd)
        self._marker = marker
        self._overflowed = False

    def _build_websocket_url(self) -> str:
        # Stdin stays on: `_ending_with` waits for its EOF before starting the command. The stream is
        # kept plain (no TTY). A non-TTY command keeps running for 10 seconds after its socket closes
        # unless told otherwise, and `0` means no limit. One second makes closing the socket on a
        # timeout or cancellation stop the command.
        return f'{super()._build_websocket_url()}&tty=false&max_run_after_disconnect=1s'

    async def _handle_message(self, message: str | bytes) -> None:
        await super()._handle_message(message)
        # Counted as frames arrive, so a runaway command cannot grow the buffers past one frame over the cap.
        if len(self._stdout_buffer) + len(self._stderr_buffer) > _MAX_OUTPUT_BYTES:
            # Ends the read loop without an exit status, so `wait()` fails.
            self._overflowed = True
            self.done = True

    async def wait(self) -> int:
        try:
            return await super().wait()
        except NetworkError:
            if not self._overflowed:
                raise
        # Raised inside `_run`'s `try`, whose handler closes the socket, which stops the command.
        stdout, stderr = _split_output(
            self.get_stdout()[:_PREVIEW_BYTES], self.get_stderr()[:_PREVIEW_BYTES], self._marker
        )
        raise WorkspaceOutputLimitError(
            "Sprites command output exceeded 10 MiB safety limit; redirect the command's "
            'output to a file and read part of it instead',
            limit=_MAX_OUTPUT_BYTES,
            stdout=stdout,
            stderr=stderr,
        )


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


async def _close_client(client: AsyncSpritesClient | None) -> None:
    if client is not None and (error := await _cleanup_call(client.aclose, timeout=_CLOSE_TIMEOUT)) is not None:
        logger.warning('Could not close Sprites SDK client: %r', error)


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
    plus `env`. A `working_dir` is created on a Sprite the backend creates; an attached or
    caller-supplied Sprite must already have it. File writes go through the Sprite's filesystem
    API; the other file operations run as shell commands.
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
        if env is not None and any(type(key) is not str or type(value) is not str for key, value in env.items()):
            raise TypeError('env keys and values must be strings')
        self._sandbox = sandbox
        self._ref = ref if sandbox is None else WorkspaceRef(provider='sprites', id=sandbox.name)
        self._new_sprite_name = f'pydantic-ai-{uuid.uuid4().hex}'
        self._uncertain_create = False
        self._runtime = runtime
        self._working_dir = absolute_path('working_dir', working_dir)
        # A Sprite this backend creates starts without `working_dir`; it is made on first acquisition.
        self._working_dir_pending = sandbox is None and ref is None and self._working_dir is not None
        self._env = dict(env or {})
        # `working_dir` as the Sprite resolves it (`pwd -P`): the protocol reports a canonical absolute path.
        self._resolved_working_dir: str | None = None
        self._client = client
        self._owns_client = client is None
        self._lock = anyio.Lock()
        # Set while nothing will call `aclose()`: the owned client then closes when the last
        # operation in flight ends, so a dropped backend leaves no open connection behind.
        self._close_after_operation = False
        self._operations = 0
        # The Sprite this backend has looked up, so a later client (one per operation, while nothing
        # will call `aclose()`) attaches by name without another control-plane lookup.
        self._attached_name: str | None = None
        # Set while `_sandbox` is such an unfetched handle, which `get_sandbox()` does not hand out.
        self._unfetched = False

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    async def get_sandbox(self) -> AsyncSprite:
        """Return the typed `sprites.AsyncSprite`, for Sprites features the workspace API does not cover.

        On a backend with no Sprite yet, this creates one (which Sprites bills) or attaches to the one
        `ref` names, just like the first operation. Attaching by `ref` to a Sprite that no longer
        exists raises `WorkspaceUnavailableError`; it does not create a replacement. After that it
        returns the cached handle without checking that the Sprite still exists: one deleted
        elsewhere surfaces on the next operation. Calling this does not make you responsible for
        deleting the Sprite; whoever holds the `ref` decides, as before.

        The handle belongs to the backend's `AsyncSpritesClient`, and this never returns a handle the
        backend built without looking the Sprite up. Once `aclose()` closes the backend's own client,
        handles from it stop working, so call this again. A backend you build yourself outside a run
        opens its own client, which you close with `aclose()`.

        Raises:
            UserError: The event loop is not asyncio.
        """
        return await self._get_sandbox(fetched=True)

    async def _get_sandbox(self, *, fetched: bool = False) -> AsyncSprite:
        """The Sprite handle, looked up at most once per backend unless `fetched` asks for a fresh lookup.

        An unfetched handle is safe for the backend's own calls: a deleted Sprite still fails the
        command or file call itself, as `WorkspaceUnavailableError`.
        """
        # Every operation acquires the Sprite first, so this one check covers them all.
        _require_asyncio()
        async with self._lock:
            sandbox = await self._acquire_locked(fetched=fetched)
            if self._working_dir_pending:
                # Still pending after a failed or cancelled attempt, so the next acquisition retries it.
                await self._make_working_dir(sandbox)
            return sandbox

    async def _make_working_dir(self, sandbox: AsyncSprite) -> None:
        directory = self._working_dir
        assert directory is not None
        # Run from `/`: exec refuses to start in a `dir` that does not exist.
        result = await self._run(
            ['mkdir', '-p', '--', directory],
            timeout=_INTERNAL_EXEC_TIMEOUT,
            capture_stderr=False,
            check_working_dir=False,
            sandbox=sandbox,
            cwd='/',
        )
        if result.exit_code != 0:
            raise WorkspaceError(
                f'Could not create working_dir {directory!r} in Sprite {sandbox.name!r}: '
                f'`mkdir -p` exited {result.exit_code}: {result.stderr.strip()}'
            )
        self._working_dir_pending = False

    async def _acquire_locked(self, *, fetched: bool) -> AsyncSprite:
        """Create or attach to the Sprite; the caller holds `_lock`."""
        if (sandbox := self._sandbox) is not None and not (fetched and self._unfetched):
            return sandbox

        client = self._client
        if client is None:
            client = await _new_client()
            self._client = client

        ref = self._ref
        if (
            not fetched
            and ref is not None
            and (ref.id == self._attached_name or _lookups.exists(_LookupCache.key(client, ref.id)))
        ):
            self._sandbox = sandbox = client.sprite(ref.id)
            self._attached_name = ref.id
            self._unfetched = True
            return sandbox

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
                    # Logged as soon as it exists: a Sprite persists until deleted, whatever ends the run later.
                    logger.info('Created Sprite %s', sandbox.name)
                # Recorded as soon as the SDK returns, so a cancelled caller still leaves it named.
                self._sandbox = sandbox
                self._ref = WorkspaceRef(provider='sprites', id=sandbox.name)
                self._uncertain_create = False
                self._attached_name = sandbox.name
                self._unfetched = False
                _lookups.record(_LookupCache.key(client, sandbox.name))
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
            if ref is not None:
                # A lookup `get_sandbox()` made outside an operation also clears what later backends trust.
                _lookups.forget(ref.id)
            raise mapped from error

    async def aclose(self) -> None:
        """Close the `AsyncSpritesClient` this backend created, if it created one.

        The Sprite is untouched: the next operation opens a fresh client and reattaches by `ref`.
        A caller-supplied `client=` or `sandbox=` handle is never closed. `SpritesSandbox`
        calls this for the backend it supplied when each run ends. A close that fails or times out
        is logged, not raised.
        """
        # Under another event loop every operation refused before opening a client: nothing to close.
        if not self._owns_client or sniffio.current_async_library() != 'asyncio':
            return

        async def close() -> None:
            # Under the lock, so a Sprite still being created is recorded before its client closes.
            async with self._lock:
                client, self._client, self._sandbox = self._client, None, None
            await _close_client(client)

        # Finished even when the caller (a run being cancelled) is cancelled meanwhile.
        await _run_to_completion(close)

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[None, None]:
        self._operations += 1
        try:
            yield
        except WorkspaceUnavailableError:
            # The Sprite may be gone or the credentials revoked: the next operation, and the next
            # backend, look it up again.
            self._attached_name = None
            if self._ref is not None:
                _lookups.forget(self._ref.id)
            raise
        finally:
            self._operations -= 1
            # Without a client (none opened yet, or every operation refused under another event loop)
            # there is nothing to close.
            if self._operations == 0 and self._close_after_operation and self._owns_client and self._client is not None:
                # Detached before any await, so an operation starting meanwhile opens its own client.
                client, self._client, self._sandbox = self._client, None, None
                await _run_to_completion(lambda: _close_client(client))

    async def working_dir(self) -> str:
        async with self._operation():
            return await self._working_dir_unscoped()

    async def _working_dir_unscoped(self) -> str:
        if self._resolved_working_dir is None:
            sandbox = await self._get_sandbox()
            result = await self.run(['pwd', '-P'], timeout=_INTERNAL_EXEC_TIMEOUT)
            printed = result.stdout.removesuffix('\n')
            if result.exit_code != 0 or not posixpath.isabs(printed):
                raise WorkspaceError(
                    f'Could not determine the working directory of Sprite {sandbox.name!r}: '
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
            return await self._run(command, shell=shell, env=env, timeout=timeout)

    async def _run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        capture_stderr: bool = True,
        check_working_dir: bool = True,
        sandbox: AsyncSprite | None = None,
        cwd: str | None = None,
    ) -> CommandResult:
        """`run` inside an operation, with the switches the backend's own commands need.

        `capture_stderr=False` skips reading a stopped command's stderr capture back, and
        `check_working_dir=False` skips confirming a missing `working_dir` on failure: both are for
        the backend's own commands, which must not recurse into another capture read or check. `sandbox` is for a
        command run while `_lock` is held, and `cwd` replaces `working_dir` as the directory.
        """
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
        directory = cwd or self._working_dir
        marker = f'pydantic-ai-end-{uuid.uuid4().hex}'
        capture = f'/tmp/pydantic-ai-stderr-{uuid.uuid4().hex}'
        # `env` runs inside the wrapper: a `sh` such as dash drops variables whose names are not shell
        # identifiers from the environment it passes on.
        args = _ending_with(marker, capture, _with_env(command_argv(command, shell), {**self._env, **(env or {})}))

        # Acquiring the Sprite has its own bound; the command's deadline starts after it.
        if sandbox is None:
            sandbox = await self._get_sandbox()
        deadline = anyio.CancelScope(deadline=math.inf if timeout is None else anyio.current_time() + timeout)
        exec_command = _ExecCommand(sandbox.command(*args, cwd=directory), marker)
        code = -1
        interrupted = False
        try:
            with deadline:
                # Stdin EOF gates execution: a handshake that failed before the socket opened cannot
                # have started the command, so it is retried with backoff; an opened socket never is.
                for delay in (*_HANDSHAKE_RETRY_DELAYS, None):  # pragma: no branch - each breaks or raises
                    try:
                        await exec_command.start()
                        break
                    except (TimeoutError, InvalidMessage, InvalidHandshake) as error:
                        if exec_command.ws is not None:
                            raise WorkspaceUnavailableError(
                                'Sprites exec connection failed; command may have run'
                            ) from error
                        if delay is None:
                            raise NetworkError(f'Sprites exec handshake failed: {error}') from error
                        await anyio.sleep(delay)
                        exec_command = _ExecCommand(sandbox.command(*args, cwd=directory), marker)
                try:
                    code = await exec_command.wait()
                except NetworkError as error:
                    # The socket opened: losing EXIT cannot prove the command did not execute.
                    raise WorkspaceUnavailableError('Sprites exec connection failed; command may have run') from error
            if timeout is not None and deadline.cancelled_caught:
                interrupted = True
                # A handshake that never opened started no command, so there is no capture to read.
                started = exec_command.ws is not None
                await _close_command(exec_command)
                partial = _split_output(exec_command.get_stdout(), exec_command.get_stderr(), marker)
                # Not shielded: nothing is cancelled here, and a live exec round trip takes seconds.
                stderr = await self._collect_stderr(capture) if capture_stderr and started else ''
                raise WorkspaceTimeoutError(
                    f'Command timed out after {timeout:g} seconds', stdout=partial[0], stderr=stderr + partial[1]
                )
        except BaseException as error:
            # On a timeout or a cancellation, closing the socket is what ends the command in the Sprite.
            if not interrupted:
                started = exec_command.ws is not None
                await _close_command(exec_command)
                if capture_stderr and started:
                    # Removes the capture file, as far as the shielded grace allows.
                    await stop_shielded(lambda: self._read_capture(capture))
            if isinstance(error, Exception) and (mapped := _map_error(error, sandbox.name)) is not None:
                raise mapped from error
            raise
        await _close_command(exec_command)
        stdout, stderr = _split_output(exec_command.get_stdout(), exec_command.get_stderr(), marker)
        result = CommandResult(exit_code=code, stdout=stdout, stderr=stderr)
        if (
            code == 1
            and check_working_dir
            and directory is not None
            and marker.encode() not in exec_command.get_stdout()
        ):
            # Exec in a missing `dir` exits 1 with a `chdir` message on stdout before the wrapper
            # runs (observed 2026-09-28), which a command's own output could imitate; confirm it.
            await self._check_working_dir(sandbox, directory, result, timeout=timeout, deadline=deadline.deadline)
        if code == 137:
            await _check_sigkill_sprite(sandbox)
        return result

    async def _check_working_dir(
        self, sandbox: AsyncSprite, directory: str, failed: CommandResult, *, timeout: float | None, deadline: float
    ) -> None:
        """Raise `WorkspaceError` if `directory` does not exist in the Sprite.

        The check counts against the `failed` command's `timeout`, which ends at `deadline`.
        """
        with anyio.CancelScope(deadline=deadline) as scope:
            check = await self._run(
                ['test', '-d', directory],
                timeout=_INTERNAL_EXEC_TIMEOUT,
                check_working_dir=False,
                sandbox=sandbox,
                cwd='/',
            )
            if check.exit_code != 0:
                raise WorkspaceError(
                    f'working_dir {directory!r} does not exist in Sprite {sandbox.name}. '
                    'Create it there, or pass a working_dir that exists.'
                )
        if scope.cancelled_caught:
            raise WorkspaceTimeoutError(
                f'Command timed out after {timeout:g} seconds', stdout=failed.stdout, stderr=failed.stderr
            )

    async def _collect_stderr(self, path: str) -> str:
        """The stderr capture of a timed-out command, or `''` if it cannot be read."""
        try:
            return await self._read_capture(path)
        except Exception:
            logger.warning('Could not retrieve Sprite stderr capture')
            return ''

    async def _read_capture(self, path: str) -> str:
        """Read and remove the stderr capture of a command stopped before its wrapper printed it."""
        # Bounded to avoid moving an unbounded stderr capture through the exec URL/output buffer.
        # A fresh exec takes a live round trip of about two seconds (observed 2026-09-28), so the
        # read gets the internal command bound, not a sub-second one.
        result = await self._run(
            ['sh', '-c', 'head -c 65536 -- "$1"; rm -f -- "$1"', 'sh', path],
            timeout=_INTERNAL_EXEC_TIMEOUT,
            capture_stderr=False,
            check_working_dir=False,
        )
        return result.stdout

    async def write_bytes(self, path: str, data: bytes) -> None:
        async with self._operation():
            await self._write_bytes(path, data)

    async def _write_bytes(self, path: str, data: bytes) -> None:
        # Not through a command: the exec API sends argv in the URL, which caps a command at about 40 KB.
        sandbox = await self._get_sandbox()
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
        except PermissionError_ as error:
            raise PermissionError(path) from error
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
        async with self._operation():
            return await _ShellFilesystem(self).read_bytes(path)

    async def stat(self, path: str) -> FileEntry:
        async with self._operation():
            return await _ShellFilesystem(self).stat(path)

    async def list_dir(self, path: str) -> tuple[FileEntry, ...]:
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
