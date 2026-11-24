"""The Runta sandbox workspace backend.

The control surface is small: one microVM (a Runta "runtime") per backend, created on first use or
attached by its UUID, and commands over Runta's exec WebSocket. File operations are the ones
Pydantic AI derives from `run`: Runta's file endpoints move at most 1 MiB and cannot list, stat or
remove, so the backend does not implement `SupportsFilesystem`.

Measured against Runta (2026-11-23):

- A runtime path takes a UUID only; a name answers 422. `WorkspaceRef.id` is the UUID.
- A create repeated with the same `Idempotency-Key` returns the same runtime.
- Closing the exec socket ends the command, including one that ignores `SIGTERM`.
- The exec handshake is refused for a paused runtime, and the runtime stays paused: nothing wakes it
  but a resume. A lifecycle call returns before the change lands.
- A process ended by a signal reports exit code -1.
- A runtime deleted under a running command ends it with an `error` frame, not a closed socket.
- Creating a runtime takes 13 to 15 seconds. The default image is Ubuntu 24.04 with git and Python 3,
  and without Node.js.
- Writing an 8 MiB file through commands and reading it back took 285 seconds.
"""

from __future__ import annotations

import logging
import os
import posixpath
import uuid
from collections.abc import Mapping
from typing import Literal

import anyio

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    CommandResult,
    SupportsCommands,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    check_env_types,
    check_provider,
    check_timeout,
    command_argv,
    command_deadline,
    require_asyncio,
    run_to_completion,
)

try:
    from pydantic_ai_harness.runta_sandbox._client import (
        DEFAULT_ENDPOINT,
        ExecLost,
        ExecRefused,
        OutputLimitExceeded,
        RuntaAPIError,
        RuntaClient,
        RuntaError,
        RuntaExec,
        Runtime,
    )
# The slim install has no `websockets`.
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install `pydantic-ai-harness[runta]` to use RuntaSandbox.') from exc

logger = logging.getLogger(__name__)

_ASYNCIO_ONLY = 'Runta needs the asyncio event loop: its exec connection runs on asyncio.'
_MISSING_TOKEN = 'RuntaSandbox needs a Runta API key: set the RUNTA_TOKEN environment variable or pass `token=`.'
_READY_TIMEOUT = 240.0
_POLL_INTERVAL = 0.5
_INTERNAL_EXEC_TIMEOUT = 30
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_PREVIEW_BYTES = 64 * 1024
# The exit status the wrapper reports when it cannot enter `working_dir`, together with a marker on stderr.
_NO_WORKING_DIR = 125
# Runs `"$@"` in the directory `$2`, when there is one. Exec has no working-directory field, and `exec`
# replaces the shell, so the program's own exit status and signals are what Runta reports.
_WRAPPER = (
    'marker=$1; directory=$2; shift 2; '
    'if [ -n "$directory" ]; then cd -- "$directory" 2>/dev/null || { printf %s "$marker" >&2; exit 125; }; fi; '
    'exec "$@"'
)
_WAKING: dict[str, Literal['start', 'resume']] = {'paused': 'resume', 'suspended': 'resume', 'shutdown': 'start'}
_GONE = frozenset({'deleting', 'error', 'crashed'})


class RuntaSandboxBackend(WorkspaceBackend, SupportsCommands):
    """A Runta runtime behind the Pydantic AI `WorkspaceBackend` protocol.

    Pass `ref=` to attach to an existing runtime by its UUID; without one, the first operation creates a
    runtime. Construction does no I/O. The backend does not delete the runtime: that is the application's
    job, through `RuntaSandbox.destroy()` or the Runta API. A paused, suspended or shut-down runtime is
    started again on first use and when it refuses a command.

    Commands run under `/bin/sh -c` with `shell=True`, in the runtime's own environment plus `env`. A
    `working_dir` is created on a runtime the backend creates; an attached runtime must already have it.
    """

    def __init__(
        self,
        *,
        ref: WorkspaceRef | None = None,
        token: str | None = None,
        endpoint: str | None = None,
        image: str | None = None,
        vcpus: int | None = None,
        memory_mib: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None:
            check_provider(ref, 'runta')
        check_env_types(env)
        self._ref = ref
        # The name of the runtime this backend creates. It is also the create's idempotency key, so a
        # create that is retried after a lost response returns the same runtime.
        self._pending_name = f'pydantic-ai-{uuid.uuid4().hex}' if ref is None else None
        self._token = token
        self._endpoint = endpoint
        self._image = image
        self._vcpus = vcpus
        self._memory_mib = memory_mib
        self._working_dir = absolute_path('working_dir', working_dir)
        self._working_dir_pending = ref is None and self._working_dir is not None
        self._env = dict(env or {})
        self._resolved_working_dir: str | None = None
        self._runtime: Runtime | None = None
        self._lock = anyio.Lock()

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    def new_client(self) -> RuntaClient:
        """A client for one operation. A handful of requests per command does not need a shared pool."""
        return _open_client(self._token, self._endpoint)

    async def _get_runtime(self, client: RuntaClient) -> Runtime:
        """The running runtime, created or attached at most once per backend."""
        async with self._lock:
            if self._runtime is None:
                # A create runs to completion even if the caller is cancelled, so its runtime is recorded.
                runtime = await (
                    run_to_completion(lambda: self._create(client))
                    if self._ref is None
                    else _read_runtime(client, self._ref.id)
                )
                self._runtime = await self._wait_running(client, runtime)
            if self._working_dir_pending:
                await self._make_working_dir(self._runtime)
            return self._runtime

    async def _create(self, client: RuntaClient) -> Runtime:
        assert self._pending_name is not None
        runtime = await client.create_runtime(
            self._pending_name, image=self._image, vcpus=self._vcpus, memory_mib=self._memory_mib
        )
        # Recorded as soon as the API returns, before any setup that can fail.
        self._ref = WorkspaceRef(provider='runta', id=runtime.id)
        logger.info('Created Runta runtime %s', runtime.id)
        return runtime

    async def _wait_running(self, client: RuntaClient, runtime: Runtime) -> Runtime:
        """Start `runtime` if it is paused, suspended or shut down, and wait until it runs."""
        woken = False
        try:
            with anyio.fail_after(_READY_TIMEOUT):
                while runtime.status != 'running':
                    if runtime.status in _GONE:
                        raise WorkspaceUnavailableError(f'Runta runtime {runtime.id} is {runtime.status}.')
                    if not woken and (action := _WAKING.get(runtime.status)) is not None:
                        await client.set_running(runtime, action)
                        woken = True
                    await anyio.sleep(_POLL_INTERVAL)
                    runtime = await _read_runtime(client, runtime.id)
        except TimeoutError as error:
            raise WorkspaceError(
                f'Runta runtime {runtime.id} is still {runtime.status} after {_READY_TIMEOUT:g} seconds.'
            ) from error
        return runtime

    async def _make_working_dir(self, runtime: Runtime) -> None:
        assert self._working_dir is not None
        result = await self._run(
            ['mkdir', '-p', '--', self._working_dir], timeout=_INTERNAL_EXEC_TIMEOUT, runtime=runtime, directory=''
        )
        if result.exit_code != 0:
            raise WorkspaceError(
                f'Could not create working_dir {self._working_dir!r} in Runta runtime {runtime.id}: '
                f'`mkdir -p` exited {result.exit_code}: {result.stderr.strip()}'
            )
        self._working_dir_pending = False

    async def working_dir(self) -> str:
        if self._resolved_working_dir is None:
            result = await self._run(['pwd', '-P'], timeout=_INTERNAL_EXEC_TIMEOUT)
            printed = result.stdout.removesuffix('\n')
            if result.exit_code != 0 or not posixpath.isabs(printed):
                raise WorkspaceError(
                    f'Could not determine the working directory of the Runta runtime: '
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
        return await self._run(command, shell=shell, env=env, timeout=timeout)

    async def _run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        runtime: Runtime | None = None,
        directory: str | None = None,
    ) -> CommandResult:
        """`run`, with the switches the backend's own commands need.

        `runtime` is for a command run while `_lock` is held, and `directory` replaces `working_dir`.
        """
        check_timeout(timeout)
        check_env_types(env)
        marker = f'pydantic-ai-no-working-dir-{uuid.uuid4().hex}'
        directory = (self._working_dir or '') if directory is None else directory
        args = [
            'sh',
            '-c',
            _WRAPPER,
            'sh',
            marker,
            directory,
            *_with_env(command_argv(command, shell), {**self._env, **(env or {})}),
        ]
        require_asyncio(_ASYNCIO_ONLY)
        async with self.new_client() as client:
            if runtime is None:
                return await self._run_args(client, await self._get_runtime(client), args, timeout, marker)
            return await self._run_args(client, runtime, args, timeout, marker, locked=True)

    async def _run_args(
        self,
        client: RuntaClient,
        runtime: Runtime,
        args: list[str],
        timeout: float | None,
        marker: str,
        *,
        locked: bool = False,
    ) -> CommandResult:
        # The handshake has its own bound, so the command's deadline starts once it is dispatched.
        process = await self._start(client, runtime, args, locked=locked)
        try:
            async with command_deadline(timeout, stop=process.close, output=lambda: _output(process)):
                code = await process.wait(limit=_MAX_OUTPUT_BYTES)
        except OutputLimitExceeded as error:
            stdout, stderr = _output(process)
            raise WorkspaceOutputLimitError(
                f'Command output exceeded {_MAX_OUTPUT_BYTES} bytes',
                limit=_MAX_OUTPUT_BYTES,
                stdout=stdout[:_PREVIEW_BYTES],
                stderr=stderr[:_PREVIEW_BYTES],
            ) from error
        except ExecLost as error:
            # A command cut off by a deleted runtime looks like any dropped connection: ask.
            await self._raise_if_gone(client, runtime, error)
            raise WorkspaceUnavailableError('The Runta exec connection failed; the command may have run.') from error
        except RuntaError as error:
            # Runta reports a command cut off by its runtime's deletion as an error frame.
            await self._raise_if_gone(client, runtime, error)
            raise
        finally:
            with anyio.CancelScope(shield=True):
                await process.close()
        stdout, stderr = _output(process)
        if code == _NO_WORKING_DIR and stderr == marker:
            raise WorkspaceError(
                f'working_dir {self._working_dir!r} does not exist in Runta runtime {runtime.id}. '
                'Create it there, or pass a working_dir that exists.'
            )
        return CommandResult(exit_code=code, stdout=stdout, stderr=stderr)

    async def _start(self, client: RuntaClient, runtime: Runtime, args: list[str], *, locked: bool) -> RuntaExec:
        """Dispatch a command, waking a runtime that refuses the handshake because it is not running.

        `locked` says the caller already holds `_lock`, which is not reentrant.
        """
        try:
            return await client.exec(runtime.id, args, {})
        except ExecRefused as error:
            # A refused handshake dispatched nothing, so the command can be sent again once the runtime runs.
            try:
                current = await client.get_runtime(runtime.id)
            except RuntaAPIError as lookup_error:
                raise _unavailable(runtime.id, lookup_error) or error from lookup_error
            if current.status == 'running':
                raise
            if locked:
                self._runtime = await self._wait_running(client, current)
            else:
                async with self._lock:
                    self._runtime = await self._wait_running(client, current)
            return await client.exec(runtime.id, args, {})

    async def _raise_if_gone(self, client: RuntaClient, runtime: Runtime, cause: Exception) -> None:
        try:
            current = await client.get_runtime(runtime.id)
        except RuntaAPIError as error:
            if (unavailable := _unavailable(runtime.id, error)) is not None:
                raise unavailable from cause
            return
        if current.status in _GONE:
            raise WorkspaceUnavailableError(f'Runta runtime {runtime.id} is {current.status}.') from cause


async def destroy_runtime(runtime_id: str, *, token: str | None, endpoint: str | None) -> None:
    """Delete a runtime by UUID without attaching to it or waking it."""
    async with _open_client(token, endpoint) as client:
        await client.delete_runtime(runtime_id)


def _open_client(token: str | None, endpoint: str | None) -> RuntaClient:
    token = token or os.getenv('RUNTA_TOKEN')
    if not token:
        raise UserError(_MISSING_TOKEN)
    return RuntaClient(token=token, endpoint=endpoint or os.getenv('RUNTA_ENDPOINT') or DEFAULT_ENDPOINT)


async def _read_runtime(client: RuntaClient, runtime_id: str) -> Runtime:
    """Read a runtime, reporting one the API no longer has as unavailable."""
    try:
        return await client.get_runtime(runtime_id)
    except RuntaAPIError as error:
        if (unavailable := _unavailable(runtime_id, error)) is not None:
            raise unavailable from error
        raise


def _unavailable(runtime_id: str, error: RuntaAPIError) -> WorkspaceUnavailableError | None:
    """The error for a runtime the API no longer has, or `None` for any other failure."""
    if error.status == 404:
        return WorkspaceUnavailableError(f'Runta runtime {runtime_id} no longer exists.')
    if error.status in (401, 403):
        return WorkspaceUnavailableError('Runta rejected the API key.')
    return None


def _with_env(args: list[str], env: dict[str, str]) -> list[str]:
    """`args` run under the POSIX `env` utility, which adds `env` to the runtime's own environment.

    `env` runs inside the wrapper: a `sh` such as dash drops variables whose names are not shell
    identifiers from the environment it passes on. `--` ends its options, so a program or a name
    that starts with `-` is not read as one.
    """
    for key, value in env.items():
        if not key or '=' in key or '\0' in key or '\0' in value:
            raise ValueError(
                f'illegal environment variable {key!r}: a name is non-empty without "=" or NUL, a value without NUL'
            )
    return ['env', '--', *(f'{key}={value}' for key, value in env.items()), *args]


def _output(process: RuntaExec) -> tuple[str, str]:
    return process.stdout.decode(errors='replace'), process.stderr.decode(errors='replace')
