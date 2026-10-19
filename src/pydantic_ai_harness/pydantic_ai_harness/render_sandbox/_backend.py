"""A Render sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

External assumptions verified 2026-10-09 against Render SDK 1.2.0 and
https://render.com/docs/sandboxes: asynchronous exec streams keep running after the
client disconnects, `copy_to`/`copy_from` transfer file contents, and the default image
provides Python 3 on Linux. Recheck them with the live suite when upgrading the
experimental SDK API.
"""

from __future__ import annotations

import asyncio
import logging
import os
import posixpath
import shlex
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import TypeVar

import anyio

from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    WorkspaceCommand,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._workspace_provider import absolute_path, check_timeout, command_argv

try:
    from render import RenderAsync
    from render.client.errors import ClientError, RateLimitError
    from render.experimental.sandbox import SandboxNotFoundError
except ImportError as _import_error:  # pragma: no cover - exercised by the clean-import test
    raise ImportError('Install `pydantic-ai-harness[render-sandbox]` to use RenderSandbox.') from _import_error

from ._commands import run_command
from ._filesystem import NativeFilesystem

logger = logging.getLogger(__name__)

_T = TypeVar('_T')

DEFAULT_SANDBOX_TIMEOUT = 3_600
DEFAULT_WORKING_DIR = '/workspace'
_READINESS_TIMEOUT = 60.0
_OPERATION_TIMEOUT = 30.0
_STOP_TIMEOUT = 12.0
_MAX_OUTPUT_BYTES = 1_048_576
# Render allows 100 API requests a minute; readiness polling stays well under that.
_POLL_INTERVAL = 1.0
_MISSING_CREDENTIALS = 'Set RENDER_API_KEY and RENDER_WORKSPACE_ID in the environment, or pass a configured `client=`.'


def _check_env(env: Mapping[str, str] | None) -> dict[str, str]:
    values = dict(env or {})
    for key, value in values.items():
        if type(key) is not str or type(value) is not str:
            raise TypeError('env keys and values must be strings')
        if not key or '=' in key or '\x00' in key or '\x00' in value:
            raise ValueError(f'invalid environment variable {key!r}: names must be non-empty without `=`, and no NUL')
    return values


def _open_client() -> RenderAsync:
    if not os.environ.get('RENDER_WORKSPACE_ID'):
        raise WorkspaceUnavailableError(_MISSING_CREDENTIALS)
    try:
        return RenderAsync()
    except ValueError as error:
        raise WorkspaceUnavailableError(_MISSING_CREDENTIALS) from error


async def _close_client(client: RenderAsync) -> None:
    # Render 1.2 exposes no close on `RenderAsync`; its pool lives on the sandbox API client.
    await client.experimental.sandboxes.client.get_async_httpx_client().aclose()


async def _wait_despite_cancellation(task: asyncio.Task[_T]) -> _T:
    """Wait for `task` to finish even if the caller is cancelled, then re-raise the cancellation."""
    cancelled: asyncio.CancelledError | None = None
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                # Wait without forwarding cancellation or installing shield's Python 3.14+ detached-error logger.
                await asyncio.wait((task,))
            except asyncio.CancelledError as error:
                cancelled = error
    try:
        result = task.result()
    except BaseException as error:
        if cancelled is not None:
            raise cancelled from error
        raise
    if cancelled is not None:
        raise cancelled
    return result


class RenderSandboxBackend:
    """One Render sandbox; construction is lazy and does no network I/O.

    Built without a ref, the first operation creates a sandbox; built with one, it attaches and never
    creates a replacement. The backend never terminates the sandbox: whoever holds the ref does, with
    `RenderSandbox.destroy(ref)`.
    """

    def __init__(
        self,
        *,
        client: RenderAsync | None = None,
        ref: WorkspaceRef | None = None,
        allow_internet_access: bool = False,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        working_dir: str = DEFAULT_WORKING_DIR,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None and ref.provider != 'render':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'render'")
        self._env = _check_env(env)
        self._client = client
        self._ref = ref
        self._allow_internet_access = allow_internet_access
        self._sandbox_timeout = sandbox_timeout
        self._working_dir = absolute_path('working_dir', working_dir) or DEFAULT_WORKING_DIR
        self._filesystem = NativeFilesystem(self, _OPERATION_TIMEOUT)
        self._lock = anyio.Lock()
        self._create_task: asyncio.Task[None] | None = None
        # Without a caller-supplied client, each top-level operation opens its own and closes it when
        # done, so a backend dropped by a durable task or another event loop leaves no pool behind.
        self._operation_clients: dict[asyncio.Task[object], RenderAsync] = {}
        self._resolved_working_dir: str | None = None
        # A sandbox this backend created still needs its `working_dir` made, even if readiness is retried.
        self._working_dir_pending = False

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[RenderAsync]:
        if self._client is not None:
            yield self._client
            return
        task = asyncio.current_task()
        assert task is not None
        # File operations run commands on the same task: they share the outer operation's client.
        if (client := self._operation_clients.get(task)) is not None:
            yield client
            return
        client = _open_client()
        self._operation_clients[task] = client
        try:
            yield client
        finally:
            del self._operation_clients[task]
            with anyio.CancelScope(shield=True):
                await _close_client(client)

    async def _create(self, client: RenderAsync) -> None:
        try:
            async with asyncio.timeout(_OPERATION_TIMEOUT):
                sandbox = await client.experimental.sandboxes.create(
                    timeout_seconds=self._sandbox_timeout,
                    network_policy='allow-all' if self._allow_internet_access else 'deny-all',
                    env=self._env or None,
                )
        except RateLimitError:
            # Nothing was created, so the next operation may try again.
            self._create_task = None
            raise
        except ClientError as error:
            # A rejected 4xx creates nothing. Render also wraps some transport errors in `ClientError` with a
            # cause; those, like any other failure, cannot prove that, so the backend never creates again.
            if error.__cause__ is not None:
                raise WorkspaceUnavailableError('Render sandbox creation outcome is unknown') from error
            self._create_task = None
            raise WorkspaceUnavailableError('Render rejected sandbox creation') from error
        self._ref = WorkspaceRef(provider='render', id=sandbox.id)
        self._working_dir_pending = True
        logger.info('Created Render sandbox %s', sandbox.id)

    async def _ensure_ready(self, client: RenderAsync) -> str:
        async with self._lock:
            if self._resolved_working_dir is not None:
                return self._resolved_working_dir
            if self._ref is None:
                if self._create_task is None:
                    self._create_task = asyncio.create_task(self._create(client))
                # Wait for the create to record its ref even if this caller is cancelled. The lock keeps
                # concurrent first operations from creating twice, and this operation's client stays open.
                await _wait_despite_cancellation(self._create_task)
            assert self._ref is not None
            try:
                async with asyncio.timeout(_READINESS_TIMEOUT):
                    while True:
                        sandbox = await client.experimental.sandboxes.from_id(self._ref.id)
                        if sandbox.status == 'running':
                            break
                        if sandbox.status in ('errored', 'terminated'):
                            raise WorkspaceUnavailableError(f'Render sandbox is {sandbox.status}')
                        await asyncio.sleep(_POLL_INTERVAL)
                path = shlex.quote(self._working_dir)
                prepare = f'mkdir -p -- {path} && ' if self._working_dir_pending else ''
                result = await run_command(
                    client,
                    self._ref.id,
                    command_argv(f'{prepare}cd -- {path} && pwd -P', shell=True),
                    working_dir='/',
                    timeout=_OPERATION_TIMEOUT,
                    max_output_bytes=_MAX_OUTPUT_BYTES,
                    stop_timeout=_STOP_TIMEOUT,
                )
            except RateLimitError:
                raise
            except (ClientError, SandboxNotFoundError) as error:
                raise WorkspaceUnavailableError('Render sandbox is missing or access was rejected') from error
            except TimeoutError as error:
                raise WorkspaceUnavailableError('Render sandbox readiness exceeded its deadline') from error
            cwd = result.stdout.rstrip('\n')
            if result.exit_code != 0 or not cwd.startswith('/') or '\n' in cwd:
                raise WorkspaceUnavailableError(f'working_dir {self._working_dir!r} is unavailable in the sandbox')
            self._working_dir_pending = False
            self._resolved_working_dir = posixpath.normpath(cwd)
            return self._resolved_working_dir

    async def working_dir(self) -> str:
        async with self._operation() as client:
            return await self._ensure_ready(client)

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run once with stdin at EOF; a timeout or cancellation stops only this command."""
        argv = command_argv(command, shell)
        check_timeout(timeout)
        values = {**self._env, **_check_env(env)}
        async with self._operation() as client:
            cwd = await self._ensure_ready(client)
            assert self._ref is not None
            try:
                return await run_command(
                    client,
                    self._ref.id,
                    argv,
                    working_dir=cwd,
                    env=values,
                    timeout=timeout,
                    max_output_bytes=_MAX_OUTPUT_BYTES,
                    stop_timeout=_STOP_TIMEOUT,
                )
            except RateLimitError:
                raise
            except (ClientError, SandboxNotFoundError) as error:
                raise WorkspaceUnavailableError('Render sandbox is missing or access was rejected') from error
            except (WorkspaceTimeoutError, WorkspaceOutputLimitError, asyncio.CancelledError):
                raise
            except Exception as error:
                # After an uncertain stream, report a dead sandbox as unavailable; never replay the command.
                try:
                    async with asyncio.timeout(_OPERATION_TIMEOUT):
                        sandbox = await client.experimental.sandboxes.from_id(self._ref.id)
                except RateLimitError:
                    raise
                except (ClientError, SandboxNotFoundError) as status_error:
                    raise WorkspaceUnavailableError('Render sandbox is unavailable') from status_error
                if sandbox.status != 'running':
                    raise WorkspaceUnavailableError(f'Render sandbox is {sandbox.status}') from error
                raise

    async def read_bytes(self, path: str) -> bytes:
        async with self._operation() as client:
            return await self._filesystem.read_bytes(client, path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        async with self._operation() as client:
            await self._filesystem.write_bytes(client, path, data)

    async def stat(self, path: str) -> FileEntry:
        async with self._operation():
            return await self._filesystem.stat(path)

    async def list_dir(self, path: str) -> tuple[FileEntry, ...]:
        async with self._operation():
            return await self._filesystem.list_dir(path)

    async def make_dir(self, path: str) -> None:
        async with self._operation():
            await self._filesystem.make_dir(path)

    async def remove(self, path: str) -> None:
        async with self._operation():
            await self._filesystem.remove(path)

    async def exists(self, path: str) -> bool:
        async with self._operation():
            return await self._filesystem.exists(path)


async def terminate_sandbox(client: RenderAsync | None, sandbox_id: str) -> None:
    """Terminate a sandbox by ID; one that no longer exists is already gone."""
    owned = client is None
    client = client or _open_client()
    try:
        await client.experimental.sandboxes.terminate(sandbox_id)
    except SandboxNotFoundError:
        pass
    finally:
        if owned:
            with anyio.CancelScope(shield=True):
                await _close_client(client)
