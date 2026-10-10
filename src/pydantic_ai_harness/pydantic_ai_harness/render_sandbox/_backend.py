"""A lazy Render environment implementing Pydantic AI's public workspace protocols."""

from __future__ import annotations

import asyncio
import math
import posixpath
import shlex
import warnings
from collections.abc import AsyncGenerator, Callable, Coroutine, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeVar
from weakref import WeakKeyDictionary

import anyio
from render import RenderAsync
from render.client.errors import ClientError, RateLimitError
from render.experimental.sandbox import SandboxNotFoundError

from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    WorkspaceCommand,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from ._commands import run_command
from ._filesystem import NativeFilesystem
from ._types import RenderSandboxCleanupError, RenderSandboxEvent

_T = TypeVar('_T')


def _positive(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a positive finite number')
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite number')


def _environment(env: Mapping[str, str] | None) -> dict[str, str]:
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in (env or {}).items()):
        raise ValueError('env must map strings to strings')
    values = dict(env or {})
    if any(not key or '=' in key or '\0' in key or '\0' in value for key, value in values.items()):
        raise ValueError('env contains an invalid name or NUL')
    return values


@dataclass(frozen=True, kw_only=True)
class RenderSandboxConfig:
    """Host-side bounds and explicit settings for newly created environments.

    Command deadlines come from Workspace.run(timeout=...), including Harness tool
    defaults. A timed-out command stops its process group and leaves the sandbox usable.
    Normal dispatches share command_interval pacing across one RenderAsync client.
    Queue time precedes the command deadline; cancellation control bypasses the queue.
    Separate clients/processes are not coordinated. Set command_interval=0 to disable.
    """

    lifetime_seconds: int = 300
    readiness_timeout: float = 60.0
    operation_timeout: float = 30.0
    cleanup_timeout: float = 30.0
    stop_timeout: float = 12.0
    # Leave room for control requests and other callers sharing the API quota.
    command_interval: float = 2.5
    max_output_bytes: int = 1_048_576
    working_dir: str = '/workspace'
    network_policy: Literal['allow-all', 'deny-all'] = 'deny-all'
    env: Mapping[str, str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.lifetime_seconds) is not int or not 1 <= self.lifetime_seconds <= 86_400:
            raise ValueError('lifetime_seconds must be an integer between 1 and 86400')
        for name, value in (
            ('readiness_timeout', self.readiness_timeout),
            ('operation_timeout', self.operation_timeout),
            ('cleanup_timeout', self.cleanup_timeout),
            ('stop_timeout', self.stop_timeout),
        ):
            _positive(value, name)
        if self.command_interval != 0 or isinstance(self.command_interval, bool):
            _positive(self.command_interval, 'command_interval')
        if type(self.max_output_bytes) is not int or self.max_output_bytes < 1:
            raise ValueError('max_output_bytes must be a positive integer')
        if (
            not self.working_dir.startswith('/')
            or '\0' in self.working_dir
            or '\n' in self.working_dir
            or '\r' in self.working_dir
        ):
            raise ValueError('working_dir must be an absolute POSIX path without NUL or newlines')
        if self.network_policy not in ('allow-all', 'deny-all'):
            raise ValueError('network_policy must be allow-all or deny-all')
        if self.env is not None:
            object.__setattr__(self, 'env', MappingProxyType(_environment(self.env)))


class _DispatchPacer:
    """Space starts, not command lifetimes, so independent commands can overlap."""

    def __init__(self) -> None:
        self._lock = anyio.Lock()
        self._last_dispatch: float | None = None
        self._last_interval = 0.0

    async def wait(self, interval: float) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            if self._last_dispatch is not None:
                # Different backend configurations share the client: honor the larger
                # gap requested by either neighboring dispatch, without reserving slots
                # for callers cancelled while waiting.
                delay = self._last_dispatch + max(interval, self._last_interval) - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            self._last_dispatch = loop.time()
            self._last_interval = interval


# Weak keys avoid retaining SDK clients after their callers release them. The async
# SDK client and its pacing lock are used within the same event loop.
_DISPATCH_PACERS: WeakKeyDictionary[RenderAsync, _DispatchPacer] = WeakKeyDictionary()


async def _drain(operation: Coroutine[object, object, _T]) -> _T:
    """Finish an already-bounded ownership operation despite repeated cancellation."""
    task = asyncio.create_task(operation)
    cancelled: asyncio.CancelledError | None = None
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancelled = error
            except Exception:
                break
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
    """One Render environment; construction is lazy and never performs network I/O.

    A supplied ref attaches to that environment and never creates a replacement.
    The backend never destroys itself when a run ends. Call destroy() explicitly to
    terminate it, including an attached environment, once its owner has finished.
    File contents use native SDK transfers; metadata uses the public shell fallback.
    """

    def __init__(
        self,
        client: RenderAsync | None = None,
        *,
        config: RenderSandboxConfig | None = None,
        ref: WorkspaceRef | None = None,
        on_sandbox_event: Callable[[RenderSandboxEvent], None] | None = None,
    ) -> None:
        if ref is not None and (ref.provider != 'render' or not ref.id):
            raise ValueError('ref must identify a Render sandbox')
        self.client = client if client is not None else RenderAsync()
        self.config = config if config is not None else RenderSandboxConfig()
        pacer = _DISPATCH_PACERS.get(self.client)
        if pacer is None:
            pacer = _DispatchPacer()
            _DISPATCH_PACERS[self.client] = pacer
        self._pacer = pacer
        self._filesystem = NativeFilesystem(self, self.client, self.config.operation_timeout, self._pace_transfer)
        self._ref = ref
        self._owned = ref is None
        self._observer = on_sandbox_event
        self._acquire_lock = anyio.Lock()
        self._create_task: asyncio.Task[None] | None = None
        self._destroy_task: asyncio.Task[None] | None = None
        self._active: dict[asyncio.Task[object], asyncio.Event] = {}
        self._cwd: str | None = None
        self._terminal = False
        self._creation_uncertain = False

    @property
    def ref(self) -> WorkspaceRef | None:
        """Stable identity, recorded immediately after create and retained after deletion."""
        return self._ref

    @property
    def owned(self) -> bool:
        """Whether this backend was configured to create its own environment."""
        return self._owned

    def _emit(self, event: str, *, status: str | None = None, exit_code: int | None = None) -> None:
        if self._observer is not None:
            try:
                self._observer(RenderSandboxEvent(event, self._ref.id if self._ref else None, status, exit_code))
            except Exception:
                try:
                    warnings.warn('Sandbox lifecycle observer raised an exception', RuntimeWarning, stacklevel=2)
                except Exception:
                    # Warning-as-error policies or warning handlers must not abort termination.
                    pass

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[None]:
        if self._terminal:
            raise WorkspaceUnavailableError('This Render workspace has been destroyed')
        task = asyncio.current_task()
        assert task is not None
        # File operations call commands on this same task. Keep the outer call
        # registered until its transfer and staging cleanup have both finished.
        if task in self._active:
            yield
            return
        finished = asyncio.Event()
        self._active[task] = finished
        try:
            yield
        finally:
            self._active.pop(task, None)
            finished.set()

    async def _create(self) -> None:
        self._emit('creating')
        try:
            async with asyncio.timeout(self.config.operation_timeout):
                sandbox = await self.client.experimental.sandboxes.create(
                    timeout_seconds=self.config.lifetime_seconds,
                    network_policy=self.config.network_policy,
                    env=dict(self.config.env) if self.config.env is not None else None,
                )
            self._ref = WorkspaceRef(provider='render', id=sandbox.id)
            self._emit('created', status=sandbox.status)
        except RateLimitError:
            raise
        except ClientError as error:
            # Direct API ClientError is a rejected 4xx response. Render also wraps
            # some transport RequestErrors in ClientError with an explicit cause;
            # those cannot prove that the provider created nothing.
            self._creation_uncertain = error.__cause__ is not None
            message = (
                'Render sandbox creation outcome is unknown'
                if self._creation_uncertain
                else 'Render rejected sandbox creation'
            )
            raise WorkspaceUnavailableError(message) from error
        except Exception:
            # A lost create response does not prove the provider created nothing.
            self._creation_uncertain = True
            raise

    async def _await_create(self) -> None:
        assert self._create_task is not None
        await asyncio.shield(self._create_task)

    async def _ensure_ready(self) -> str:
        if self._terminal:
            raise WorkspaceUnavailableError('This Render workspace has been destroyed')
        async with self._acquire_lock:
            if self._cwd is not None:
                return self._cwd
            if self._ref is None:
                if self._create_task is None:
                    self._create_task = asyncio.create_task(self._create())
                # On cancellation, wait for the bounded create to record its ID before
                # releasing this lock. Concurrent first users cannot create twice.
                await _drain(self._await_create())
            assert self._ref is not None
            try:
                async with asyncio.timeout(self.config.readiness_timeout):
                    while True:
                        sandbox = await self.client.experimental.sandboxes.from_id(self._ref.id)
                        self._emit('status', status=sandbox.status)
                        if sandbox.status == 'running':
                            break
                        if sandbox.status in ('errored', 'terminated'):
                            raise WorkspaceUnavailableError(f'Render sandbox is {sandbox.status}')
                        await asyncio.sleep(0.25)
                path = shlex.quote(self.config.working_dir)
                prepare = f'mkdir -p -- {path} && ' if self._owned else ''
                result = await run_command(
                    self.client,
                    self._ref.id,
                    f'{prepare}cd -- {path} && pwd -P',
                    working_dir='/',
                    timeout=self.config.operation_timeout,
                    max_output_bytes=self.config.max_output_bytes,
                    stop_timeout=self.config.stop_timeout,
                    pace=self._pace_transfer,
                )
                cwd = result.stdout.rstrip('\n')
                if result.exit_code != 0 or not cwd.startswith('/') or '\n' in cwd:
                    raise WorkspaceUnavailableError('The sandbox working directory is unavailable')
                self._cwd = posixpath.normpath(cwd)
                self._emit('ready', status='running')
                return self._cwd
            except RateLimitError:
                raise
            except (ClientError, SandboxNotFoundError) as error:
                raise WorkspaceUnavailableError('Render sandbox is missing or access was rejected') from error
            except TimeoutError as error:
                raise WorkspaceUnavailableError('Render sandbox readiness exceeded its deadline') from error

    async def working_dir(self) -> str:
        async with self._operation():
            return await self._ensure_ready()

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run once, with stdin at EOF; timeout and cancellation stop only this command."""
        if isinstance(command, str):
            if not shell:
                raise TypeError('A string command requires shell=True')
            invocation = command
        else:
            if shell:
                raise TypeError('An argv command requires shell=False')
            if not command:
                raise ValueError('command argv cannot be empty')
            invocation = command
        if any('\0' in part for part in ([invocation] if isinstance(invocation, str) else invocation)):
            raise ValueError('command cannot contain NUL')
        if timeout is not None:
            _positive(timeout, 'timeout')
        values = _environment(self.config.env)
        values.update(_environment(env))
        async with self._operation():
            cwd = await self._ensure_ready()
            assert self._ref is not None
            # The runner starts its deadline only after this cancellable pacing wait.
            # Its separate command-stop dispatch deliberately bypasses normal pacing.
            try:
                result = await run_command(
                    self.client,
                    self._ref.id,
                    invocation,
                    working_dir=cwd,
                    env=values or None,
                    timeout=timeout,
                    max_output_bytes=self.config.max_output_bytes,
                    stop_timeout=self.config.stop_timeout,
                    pace=self._pace_transfer,
                )
            except RateLimitError:
                raise
            except (ClientError, SandboxNotFoundError) as error:
                raise WorkspaceUnavailableError('Render sandbox is missing or access was rejected') from error
            except (WorkspaceTimeoutError, WorkspaceOutputLimitError, asyncio.CancelledError):
                raise
            except Exception as error:
                # Read status after an uncertain stream; never replay a command.
                try:
                    async with asyncio.timeout(self.config.operation_timeout):
                        sandbox = await self.client.experimental.sandboxes.from_id(self._ref.id)
                except RateLimitError:
                    raise
                except (ClientError, SandboxNotFoundError) as status_error:
                    raise WorkspaceUnavailableError('Render sandbox is unavailable') from status_error
                if sandbox.status != 'running':
                    raise WorkspaceUnavailableError(f'Render sandbox is {sandbox.status}') from error
                raise
            self._emit('command_finished', exit_code=result.exit_code)
            return result

    async def _pace_transfer(self) -> None:
        await self._pacer.wait(self.config.command_interval)

    async def read_bytes(self, path: str) -> bytes:
        async with self._operation():
            return await self._filesystem.read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        async with self._operation():
            await self._filesystem.write_bytes(path, data)

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

    async def _destroy(self, caller: asyncio.Task[object] | None) -> None:
        self._terminal = True
        try:
            async with asyncio.timeout(self.config.cleanup_timeout):
                active = [(task, event) for task, event in self._active.items() if task is not caller]
                for task, _ in active:
                    task.cancel()
                if active:
                    # Drain only the workspace calls, not their caller tasks: a Pydantic
                    # graph task may itself be waiting for this run hook to finish.
                    await asyncio.gather(*(event.wait() for _, event in active))
                if self._create_task is not None:
                    try:
                        await asyncio.shield(self._create_task)
                    except Exception:
                        pass
                if self._ref is None:
                    if self._creation_uncertain:
                        raise RenderSandboxCleanupError(
                            'Create outcome is unknown; no ID is available for cleanup. '
                            f'The requested {self.config.lifetime_seconds}s lifetime remains the bound.'
                        )
                    return
                self._emit('terminating')
                try:
                    await self.client.experimental.sandboxes.terminate(self._ref.id)
                except SandboxNotFoundError:
                    pass
                self._emit('termination_acknowledged')
                while True:
                    try:
                        sandbox = await self.client.experimental.sandboxes.from_id(self._ref.id)
                    except SandboxNotFoundError:
                        self._emit('terminated', status='not_found')
                        return
                    if sandbox.status == 'terminated':
                        self._emit('terminated', status='terminated')
                        return
                    await asyncio.sleep(0.25)
        except BaseException as error:
            self._emit('cleanup_failed')
            if isinstance(error, RenderSandboxCleanupError):
                raise
            raise RenderSandboxCleanupError('Render sandbox termination could not be confirmed') from error

    async def _await_destroy(self) -> None:
        assert self._destroy_task is not None
        await asyncio.shield(self._destroy_task)

    async def destroy(self) -> None:
        """Explicitly terminate this environment, whether created here or attached.

        The ref remains available for recovery if termination cannot be confirmed.
        """
        if self._destroy_task is None or (self._destroy_task.done() and self._destroy_task.exception()):
            self._destroy_task = asyncio.create_task(self._destroy(asyncio.current_task()))
        await _drain(self._await_destroy())
