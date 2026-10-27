"""A Render sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

External assumptions verified 2026-10-09 against Render SDK 1.2.0 and
https://render.com/docs/sandboxes: asynchronous exec streams keep running after the
client disconnects, `copy_to`/`copy_from` transfer file contents, and the default image
provides Python 3 on Linux. Recheck them with the live suite when upgrading the
experimental SDK API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import posixpath
import shlex
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import anyio

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
from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    check_env,
    check_integer,
    check_timeout,
    command_argv,
    run_to_completion,
    running_on_asyncio,
)

try:
    from render import RenderAsync
    from render.client.errors import ClientError, RateLimitError
    from render.experimental.sandbox import SandboxDownloadError, SandboxFileNotFoundError, SandboxNotFoundError
except ImportError as _import_error:  # pragma: no cover - exercised by the clean-import test
    raise ImportError('Install `pydantic-ai-harness[render]` to use RenderSandbox.') from _import_error

from ._commands import client_scope, prepare_command, run_command

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_TIMEOUT = 3_600
DEFAULT_WORKING_DIR = '/workspace'
_READINESS_TIMEOUT = 60.0
_OPERATION_TIMEOUT = 30.0
# Render allows 100 API requests a minute; readiness polling stays well under that.
_POLL_INTERVAL = 1.0
# Transfers get the base timeout plus this much per byte, as E2B's uploads do, so large files can finish.
_MIN_BYTES_PER_SECOND = 256 * 1024
_HELPER = files('pydantic_ai_harness.render_sandbox').joinpath('_file_transfer.py').read_text()


def check_sandbox_timeout(value: int) -> None:
    """Raise `UserError` unless `value` is a lifetime Render accepts."""
    check_integer('sandbox_timeout', value)
    if value > 86_400:
        raise UserError(f'sandbox_timeout must be at most 86400 seconds, got {value!r}.')


def _require_asyncio() -> None:
    if not running_on_asyncio():
        raise UserError('Render sandbox needs the asyncio event loop: the backend runs its calls on asyncio tasks.')


class RenderSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem):
    """A [Render sandbox](https://render.com/docs/sandboxes) as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Building one does no I/O. Built without a ref, the first operation creates a sandbox; built with one,
    it attaches and never creates a replacement. The backend never terminates the sandbox: whoever holds
    the ref does, with `RenderSandbox.destroy(ref)`.

    Each command runs under a small Python supervisor sent with Render's exec request, which stops the
    command's process group on a timeout or cancellation. File contents travel through Render's
    `copy_to`/`copy_from`; metadata operations run as shell commands.

    Args:
        client: A caller-owned `render.RenderAsync`, never closed here. When `None`, each operation opens
            a client from `RENDER_API_KEY` and `RENDER_WORKSPACE_ID` and closes it when the operation ends.
        ref: Identity of an existing sandbox to attach to on first use.
        allow_internet_access: Whether a newly created sandbox may reach the internet.
        sandbox_timeout: Seconds a newly created sandbox lives before Render terminates it, 1 to 86400.
        working_dir: Absolute directory commands start in and relative paths resolve against. Created on a
            new sandbox; an attached sandbox must already have it.
        env: Environment variables every command gets; per-command `env` is layered on top. Nothing is
            read from the host environment.
    """

    def __init__(
        self,
        *,
        client: RenderAsync | None = None,
        ref: WorkspaceRef | None = None,
        allow_internet_access: bool = True,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        working_dir: str = DEFAULT_WORKING_DIR,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None and ref.provider != 'render':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'render'")
        check_sandbox_timeout(sandbox_timeout)
        check_env(env)
        self._env = dict(env or {})
        self._client = client
        self._ref = ref
        self._allow_internet_access = allow_internet_access
        self._sandbox_timeout = sandbox_timeout
        absolute_path('working_dir', working_dir)
        self._working_dir = working_dir
        # Metadata operations use core's shell implementation over this backend's commands, as Sprites does.
        self._shell = _ShellFilesystem(self)
        self._lock = anyio.Lock()
        self._resolved_working_dir: str | None = None
        # A sandbox this backend created still needs its `working_dir` made, even if readiness is retried.
        self._working_dir_pending = False

    @property
    def ref(self) -> WorkspaceRef | None:
        """The sandbox's identity: `None` until a fresh one is created, then set for good."""
        return self._ref

    def _operation(self) -> AbstractAsyncContextManager[RenderAsync]:
        # Without a caller-supplied client, each top-level operation opens its own and closes it when
        # done, so a backend dropped by a durable task or another event loop leaves no pool behind.
        _require_asyncio()
        return client_scope(self._client)

    async def _create(self, client: RenderAsync) -> None:
        try:
            async with asyncio.timeout(_OPERATION_TIMEOUT):
                sandbox = await client.experimental.sandboxes.create(
                    timeout_seconds=self._sandbox_timeout,
                    network_policy='allow-all' if self._allow_internet_access else 'deny-all',
                )
        except RateLimitError:
            raise
        except (ClientError, TimeoutError) as error:
            raise WorkspaceUnavailableError('Could not create a Render sandbox') from error
        self._ref = WorkspaceRef(provider='render', id=sandbox.id)
        self._working_dir_pending = True
        logger.info('Created Render sandbox %s', sandbox.id)

    async def _ensure_ready(self, client: RenderAsync) -> str:
        async with self._lock:
            if self._resolved_working_dir is not None:
                return self._resolved_working_dir
            if self._ref is None:
                # Record the paid sandbox's ref even if this caller is cancelled meanwhile.
                await run_to_completion(lambda: self._create(client))
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
                resolve = prepare_command(
                    command_argv(f'{prepare}cd -- {path} && pwd -P', shell=True),
                    working_dir='/',
                )
                result = await run_command(
                    client, self._ref.id, resolve, timeout=_OPERATION_TIMEOUT, stop_client=self._client
                )
            except RateLimitError:
                raise
            except ClientError as error:
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
        check_env(env)
        values = {**self._env, **(env or {})}
        # Checked against the configured directory first, so a command too large to send creates no sandbox.
        prepare_command(argv, working_dir=self._working_dir, env=values, timeout=timeout)
        async with self._operation() as client:
            return await self._run(client, argv, values, timeout)

    async def _run(
        self, client: RenderAsync, argv: list[str], env: Mapping[str, str], timeout: float | None
    ) -> CommandResult:
        cwd = await self._ensure_ready(client)
        assert self._ref is not None
        prepared = prepare_command(argv, working_dir=cwd, env=env, timeout=timeout)
        try:
            return await run_command(client, self._ref.id, prepared, timeout=timeout, stop_client=self._client)
        except RateLimitError:
            raise
        except ClientError as error:
            raise WorkspaceUnavailableError('Render sandbox is missing or access was rejected') from error
        except (WorkspaceTimeoutError, WorkspaceOutputLimitError, WorkspaceUnavailableError):
            raise
        except Exception as error:
            # After an uncertain stream, report a dead sandbox as unavailable. If the status check itself
            # fails, raise the stream error: the command may have run, so nothing may look retryable.
            try:
                async with asyncio.timeout(_OPERATION_TIMEOUT):
                    sandbox = await client.experimental.sandboxes.from_id(self._ref.id)
            except SandboxNotFoundError as status_error:
                raise WorkspaceUnavailableError('Render sandbox is unavailable') from status_error
            except Exception as status_error:
                raise error from status_error
            if sandbox.status != 'running':
                raise WorkspaceUnavailableError(f'Render sandbox is {sandbox.status}') from error
            raise

    def _deadline(self, size: int) -> float:
        return _OPERATION_TIMEOUT + size / _MIN_BYTES_PER_SECOND

    async def _helper(self, client: RenderAsync, action: str, path: str, directory: str, size: int = 0) -> int:
        argv = ['python3', '-I', '-c', _HELPER, action, path, directory, str(size)]
        result = await self._run(client, argv, self._env, self._deadline(size))
        try:
            value = json.loads(result.stdout)
            if result.exit_code == 1 and type(value.get('errno')) is int:
                raise OSError(value['errno'], value.get('message', 'File operation failed'), path)
            if result.exit_code == 0 and type(value.get('size')) is int and value['size'] >= 0:
                return value['size']
        except (ValueError, AttributeError):
            pass
        raise WorkspaceError(f'Render file {action} returned an invalid result for {path!r}')

    async def _transfer(self, client: RenderAsync, path: str, data: bytes | None) -> bytes:
        if not path.startswith('/') or '\0' in path:
            raise ValueError('file path must be absolute and contain no NUL')
        await self._ensure_ready(client)
        ref = self._ref
        assert ref is not None
        # Known before dispatch, so cancellation before acknowledgement still permits
        # cleanup. Never put agent file bytes in command arguments or command output.
        remote = f'/tmp/.render-pydantic-{uuid4().hex}'
        try:
            size = await self._helper(client, 'read' if data is None else 'prepare', path, remote)
            with TemporaryDirectory(prefix='render-pydantic-') as directory:
                local = Path(directory) / 'data'
                if data is not None:
                    await anyio.Path(local).write_bytes(data)
                # The public SDK has per-connection timeouts but no total file deadline.
                # Never retry an uncertain upload or hide failure with a shell fallback.
                async with asyncio.timeout(self._deadline(size if data is None else len(data))):
                    if data is None:
                        await client.experimental.sandboxes.copy_from(ref.id, f'{remote}/data', local)
                    else:
                        await client.experimental.sandboxes.copy_to(ref.id, local, f'{remote}/data')
                if data is None:
                    # Reject a changed snapshot before allocating the returned bytes.
                    if local.is_symlink() or local.stat().st_size != size:
                        raise WorkspaceError('Render file download size did not match its snapshot')
                    return await anyio.Path(local).read_bytes()
                await self._helper(client, 'commit', path, remote, len(data))
                return b''
        except SandboxFileNotFoundError as error:
            raise FileNotFoundError(path) from error
        except SandboxNotFoundError as error:
            raise WorkspaceUnavailableError('Render sandbox is unavailable') from error
        except TimeoutError as error:
            raise WorkspaceTimeoutError(f'Render file transfer timed out for {path!r}') from error
        except RateLimitError:
            raise
        except (ClientError, SandboxDownloadError) as error:
            # A rejected request or an unsafe download; transient SDK errors propagate for durable engines to retry.
            raise WorkspaceError(f'Render file transfer failed for {path!r}') from error
        finally:
            # Shield bounded best-effort staging cleanup, preserving the original error.
            # Process death can leave staging until the owner terminates the sandbox.
            with anyio.move_on_after(_OPERATION_TIMEOUT, shield=True):
                try:
                    argv = ['python3', '-I', '-c', _HELPER, 'cleanup', path, remote]
                    await self._run(client, argv, self._env, _OPERATION_TIMEOUT)
                except Exception:
                    pass

    async def read_bytes(self, path: str) -> bytes:
        async with self._operation() as client:
            return await self._transfer(client, path, None)

    async def write_bytes(self, path: str, data: bytes) -> None:
        async with self._operation() as client:
            await self._transfer(client, path, data)

    async def stat(self, path: str) -> FileEntry:
        return await self._shell.stat(path)

    async def list_dir(self, path: str) -> tuple[FileEntry, ...]:
        return await self._shell.list_dir(path)

    async def make_dir(self, path: str) -> None:
        await self._shell.make_dir(path)

    async def remove(self, path: str) -> None:
        await self._shell.remove(path)

    async def exists(self, path: str) -> bool:
        return await self._shell.exists(path)


async def terminate_sandbox(client: RenderAsync | None, sandbox_id: str) -> None:
    """Terminate a sandbox by ID; Render accepts an already terminated one, and an unknown ID is already gone."""
    _require_asyncio()
    async with client_scope(client) as scope:
        try:
            await scope.experimental.sandboxes.terminate(sandbox_id)
        except SandboxNotFoundError:
            pass
        except RateLimitError:
            raise
        except ClientError as error:
            raise WorkspaceUnavailableError('Render rejected the termination; check the credentials') from error
