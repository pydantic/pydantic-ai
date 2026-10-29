"""An [OpenSandbox](https://github.com/opensandbox-group/OpenSandbox) sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

Behavior verified 2026-10-29 against the `opensandbox` SDK 1.1.1 and 1.2.0 and `opensandbox-server` 1.2.0, on its
Docker runtime and on Kubernetes with both workload providers (`batchsandbox` and `agent-sandbox`), with execd
1.1.0, 1.1.1 and 1.2.0. Re-check these by running the live tier (see `tests/harness/opensandbox_sandbox`) before
changing how sandboxes are created, attached to, or how commands run and stop:

- A command keeps running when its client goes away, and execd's own `timeout` and interrupt signal only its
  leader; execd starts every command as the leader of a process group of its own.
- execd runs shell text under `bash -c`, or `sh -c` in an image without bash; it reports a program it cannot
  start with no exit code and a signal death as `-1`.
- execd before 1.1.1 strips the newline from every line of output.
- A cancelled `Sandbox.create` leaves the sandbox running. On Kubernetes the server answers a create only once
  the pod runs, which can take longer than the SDK's 30-second request timeout; the sandbox goes on starting.
- A killed sandbox breaks a running command's stream at once, but still reads `Running`, then `Stopping` or
  `Failed`, for up to about two seconds before the server answers 404.
- Errors carry the HTTP status; their codes are prefixed by the runtime (`DOCKER::`, `KUBERNETES::`).
- A sandbox's `timeout` deletes it when it runs out, files included; `renew` sets a new one.
- The SDK runs on asyncio only, and each `Sandbox` holds an HTTP transport bound to the event loop that opened it.

File operations go through the shell, as for `DockerSandboxBackend`: execd's own file API reads a directory as a
500 and deletes a missing path without error, so native file access needs its own probes and is left for later.
"""

from __future__ import annotations as _annotations

import logging
import posixpath
import secrets
import shlex
import time
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from datetime import timedelta

import anyio
import anyio.lowlevel

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    CommandResult,
    SupportsCommands,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._workspace_provider import (
    check_env,
    check_timeout,
    command_argv,
    run_to_completion,
    running_on_asyncio,
    stop_shielded,
)

try:
    from opensandbox import Sandbox
    from opensandbox.config import ConnectionConfig
    from opensandbox.exceptions import SandboxApiException, SandboxConnectionException, SandboxException
    from opensandbox.manager import SandboxManager
    from opensandbox.models.execd import ExecutionHandlers, OutputMessage, RunCommandOpts
    from opensandbox.models.sandboxes import SandboxFilter
except ImportError as _import_error:  # pragma: no cover
    raise ImportError('Install `pydantic-ai-harness[opensandbox]` to use OpenSandbox.') from _import_error

__all__ = ('OpenSandboxBackend', 'destroy_sandbox')

PROVIDER = 'opensandbox'

logger = logging.getLogger(__name__)

_PID_DIR = '/tmp/pydantic-ai'
"""Where each command records its PID for a later stop. The server creates the sandbox, so unlike Docker's there is
no mount of the backend's own to use; `/tmp` is the directory every image can write."""

_WRAPPER = f"""tag=$1
pidfile={_PID_DIR}/$tag.pid
stopped={_PID_DIR}/$tag.stopped
shift
if ! {{ mkdir -p {_PID_DIR} && echo $$ > "$pidfile"; }} 2> /dev/null; then
    echo "cannot write $pidfile, which stops the command on a timeout: {_PID_DIR} must be writable" >&2
    exit 1
fi
if [ -e "$stopped" ]; then
    rm -f "$pidfile" "$stopped"
    exit 143
fi
(exec "$@")
status=$?
rm -f "$pidfile"
exit "$status"
"""
"""Runs as `sh -c WRAPPER sh <tag> <argv...>`: records its PID so a second command can stop it, then runs the argv.

`DockerSandboxBackend`'s wrapper, without the markers `docker exec` needs. A subshell `exec` runs the program
itself and exits 127 when it is missing, 126 when it cannot run, and 128 plus the signal number when a signal
ends it, as a shell reports them.
"""

_STOP = f"""pidfile={_PID_DIR}/$1.pid
stopped={_PID_DIR}/$1.stopped
mkdir -p {_PID_DIR} && : > "$stopped"
read -r pid 2> /dev/null < "$pidfile" || exit 0
[ -n "$pid" ] || exit 0
kill -KILL "-$pid" "$pid" 2> /dev/null
rm -f "$pidfile" "$stopped"
exit 0"""
"""Kills the command recorded under the tag `$1` and its process group, as `DockerSandboxBackend`'s stop does.

The `.stopped` marker it leaves first fences a command whose start the stop overtook: whichever of the two
comes second sees the other's file.
"""

_STOP_TIMEOUT = 2.0
"""Bounds how late a stopped command's timeout or cancellation is raised when the server stops answering."""

_CREATE_READY_TIMEOUT = timedelta(seconds=120)
"""How long a new sandbox may take to answer after the server created it, image pull included."""

_GONE_STATES = frozenset({'Stopping', 'Deleting', 'Terminated', 'Failed'})
_GONE_SETTLE = 2.0
"""How long a broken command waits for the server to stop reporting a killed sandbox as running."""

_CREATE_TAG = 'pydantic-ai-create'
"""Metadata key a new sandbox is tagged with, to find it when its create response is lost."""

_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_OUTPUT_PREVIEW_CHARS = 64 * 1024


def _require_asyncio() -> None:
    if not running_on_asyncio():
        raise UserError(
            'OpenSandbox needs the asyncio event loop: the OpenSandbox SDK runs its calls on asyncio tasks.'
        )


def _unavailable(sandbox_id: str) -> WorkspaceUnavailableError:
    return WorkspaceUnavailableError(
        f'The OpenSandbox sandbox {sandbox_id!r} no longer exists: it was killed or its `sandbox_timeout` ran out. '
        "Pass `workspace='new'` to start a fresh sandbox."
    )


def _refusal(error: SandboxApiException, context: str) -> WorkspaceUnavailableError | None:
    """The protocol's error for a request no retry can fix, or `None` for one that may be transient."""
    if error.status_code in (401, 403):
        # The server's message is fixed text, but the code alone keeps any credential out of the error.
        return WorkspaceUnavailableError(f'{context}: OpenSandbox rejected the API key ({error.error.code}).')
    if error.status_code in (400, 404, 422):
        return WorkspaceUnavailableError(f'{context}: {error}')
    return None


async def destroy_sandbox(sandbox_id: str, connection: ConnectionConfig | None = None) -> None:
    """Kill a sandbox by ID without connecting to it; one that is already gone returns quietly."""
    _require_asyncio()
    manager = await SandboxManager.create(connection_config=connection or ConnectionConfig())
    try:
        await manager.kill_sandbox(sandbox_id)
    except SandboxApiException as error:
        if error.status_code == 404:
            return
        if (refused := _refusal(error, f'Could not destroy OpenSandbox sandbox {sandbox_id!r}')) is not None:
            raise refused from error
        raise
    finally:
        await manager.close()


class OpenSandboxBackend(WorkspaceBackend, SupportsCommands):
    """Run commands in an [OpenSandbox](https://open-sandbox.ai) sandbox, created on first use.

    Built with `image`, the first operation creates a sandbox from it; built with `ref`, it attaches to that
    sandbox and raises [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] if it is
    gone. The backend never kills the sandbox: call
    [`OpenSandbox.destroy`][pydantic_ai_harness.opensandbox_sandbox.OpenSandbox.destroy] with its ref.

    The image needs a POSIX `sh` and the usual file utilities. File operations run through that shell. The
    OpenSandbox SDK runs on asyncio only, so under Trio every operation raises `UserError`.

    Args:
        image: The image for a new sandbox. Pass it or `ref`, not both.
        ref: A `WorkspaceRef(provider='opensandbox', id=<sandbox id>)` to attach to instead of creating one.
        connection: The `opensandbox.config.ConnectionConfig` for the server; `None` reads `OPEN_SANDBOX_DOMAIN`
            and `OPEN_SANDBOX_API_KEY`. A transport you pass in it is yours to close.
        working_dir: Absolute directory commands start in and relative paths resolve against, created on first
            use if missing; `None` uses the image's own.
        env: Environment variables every command gets; the per-call `env` goes on top. Nothing is read from the host.
        resource: Resource limits for a new sandbox, such as `{'cpu': '500m', 'memory': '1Gi'}`; `None` keeps the
            SDK's default of one CPU and 2 GiB.
        sandbox_timeout: Seconds a sandbox may go unused before the server deletes it, files included. Every
            operation renews it as it starts, so a single command must finish within it. `None` keeps it until it
            is destroyed.
    """

    def __init__(
        self,
        *,
        image: str | None = None,
        ref: WorkspaceRef | None = None,
        connection: ConnectionConfig | None = None,
        working_dir: str | None = '/workspace',
        env: Mapping[str, str] | None = None,
        resource: Mapping[str, str] | None = None,
        sandbox_timeout: int | None = None,
    ) -> None:
        if (image is None) == (ref is None):
            raise ValueError('pass exactly one of `image`, to create a sandbox, or `ref`, to attach to one')
        if ref is not None and ref.provider != PROVIDER:
            raise ValueError(f'unsupported workspace provider {ref.provider!r}; expected {PROVIDER!r}')
        if working_dir is not None and not posixpath.isabs(working_dir):
            raise ValueError(f'working_dir must be an absolute path in the sandbox, got {working_dir!r}')
        if sandbox_timeout is not None and (type(sandbox_timeout) is not int or sandbox_timeout < 1):
            raise ValueError(f'sandbox_timeout must be an integer of at least 1 or None, got {sandbox_timeout!r}')
        check_env(env)
        self._image = image
        self._ref = ref
        self._connection = connection
        self._working_dir = working_dir
        self._env = dict(env or {})
        self._resource = dict(resource) if resource is not None else None
        self._sandbox_timeout = sandbox_timeout
        self._sandbox: Sandbox | None = None
        self._resolved_working_dir: str | None = None
        self._lock = anyio.Lock()
        self._operations = 0
        self._close_after_operation = False

    @property
    def ref(self) -> WorkspaceRef | None:
        """`WorkspaceRef(provider='opensandbox', id=<sandbox id>)`, or `None` before a sandbox has been created."""
        return self._ref

    async def get_sandbox(self) -> Sandbox:
        """The `opensandbox.Sandbox`, for features the workspace API does not cover.

        On a backend with no sandbox yet this creates one, or attaches to the one `ref` names, just like the
        first operation. It shares the backend's transport, so it stops working once the backend closes: at
        `aclose()`, and, for a backend `OpenSandbox` built that no run owns, after each operation.
        """
        # Not an operation: a backend that closes after each operation would close the handle it returns.
        _require_asyncio()
        return await self._acquire()

    async def working_dir(self) -> str:
        async with self._operation():
            await self._acquire()
            assert self._resolved_working_dir is not None
            return self._resolved_working_dir

    async def aclose(self) -> None:
        """Close the HTTP transport the SDK opened; the sandbox keeps running.

        The transport belongs to the event loop that opened it, so a backend used from another loop must be
        closed first; a later operation attaches again by `ref`. An operation in flight keeps it open until the
        last one ends. `OpenSandbox` calls this when the run that used the backend ends.
        """
        if not running_on_asyncio():
            # Every operation refuses Trio, so there is no transport to close.
            return
        if self._operations:
            self._close_after_operation = True
            return

        async def close() -> None:
            # Under the lock, so a sandbox still being created is recorded before its transport closes.
            async with self._lock:
                sandbox, self._sandbox = self._sandbox, None
            if sandbox is not None:
                await sandbox.close()

        await run_to_completion(close)

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[None]:
        _require_asyncio()
        self._operations += 1
        try:
            yield
        finally:
            self._operations -= 1
            if self._operations == 0 and self._close_after_operation and self._sandbox is not None:
                # Detached before any await, so an operation starting meanwhile attaches on its own.
                sandbox, self._sandbox = self._sandbox, None
                await run_to_completion(sandbox.close)

    async def _acquire(self) -> Sandbox:
        """Create or attach once, however many first operations race; renew an idle timeout on the way."""
        async with self._lock:
            sandbox = self._sandbox
            if sandbox is None:
                sandbox = await self._create() if self._ref is None else await self._attach(self._ref.id)
                self._sandbox = sandbox
            elif self._sandbox_timeout is not None:
                # The server only has an expiry time, so each operation moves it a full timeout away.
                await self._renew(sandbox)
            if self._resolved_working_dir is None:
                self._resolved_working_dir = await self._enter_working_dir(sandbox)
        await anyio.lowlevel.checkpoint_if_cancelled()
        return sandbox

    async def _create(self) -> Sandbox:
        assert self._image is not None
        image = self._image
        timeout = None if self._sandbox_timeout is None else timedelta(seconds=self._sandbox_timeout)
        # The server takes no ID from the client, so a tag finds the sandbox if the create response is lost.
        tag = secrets.token_hex(16)

        async def create() -> Sandbox:
            # Recorded here, inside the task a cancelled caller waits for, so the sandbox, or the ref of one whose
            # create failed, is kept even when that caller then raises its cancellation instead of the outcome.
            try:
                sandbox = await Sandbox.create(
                    image,
                    timeout=timeout,
                    ready_timeout=_CREATE_READY_TIMEOUT,
                    resource=self._resource,
                    metadata={_CREATE_TAG: tag},
                    connection_config=self._connection,
                )
            except SandboxApiException as error:
                if (refused := _refusal(error, f'Could not create an OpenSandbox sandbox from {image!r}')) is not None:
                    raise refused from error
                await self._find_created(tag)
                raise
            except SandboxException:
                await self._find_created(tag)
                raise
            self._sandbox = sandbox
            self._ref = WorkspaceRef(provider=PROVIDER, id=sandbox.id)
            # Logged at creation, so the sandbox can be found whatever ends the run before its ref is stored.
            logger.info('Created OpenSandbox sandbox %s', sandbox.id)
            return sandbox

        # A cancelled create leaves the sandbox running, so it finishes and is recorded first.
        return await run_to_completion(create)

    async def _find_created(self, tag: str) -> None:
        """Record the ref of a sandbox whose create failed after the server made it, so it can be destroyed.

        The original error still propagates: a durable engine's retry then attaches to the sandbox instead of
        creating another. A failed lookup leaves that error alone.
        """
        try:
            manager = await SandboxManager.create(connection_config=self._connection or ConnectionConfig())
            try:
                found = await manager.list_sandbox_infos(SandboxFilter(metadata={_CREATE_TAG: tag}))
            finally:
                await manager.close()
        except Exception:
            logger.warning('Could not look up the OpenSandbox sandbox a failed create may have left', exc_info=True)
            return
        for info in found.sandbox_infos:
            # The SDK kills a sandbox whose create failed after the server answered, such as one that never
            # became ready; by the time its error arrives here, the server reports that sandbox as going.
            if info.status.state in _GONE_STATES:
                continue
            self._ref = WorkspaceRef(provider=PROVIDER, id=info.id)
            logger.info('Created OpenSandbox sandbox %s, but its create failed: %s', info.id, info.status.state)

    async def _attach(self, sandbox_id: str) -> Sandbox:
        context = f'Could not attach to OpenSandbox sandbox {sandbox_id!r}'
        try:
            sandbox = await Sandbox.connect(sandbox_id, connection_config=self._connection)
        except SandboxApiException as error:
            if error.status_code == 404:
                raise _unavailable(sandbox_id) from error
            if (refused := _refusal(error, context)) is not None:
                raise refused from error
            raise
        try:
            # A killed sandbox still answers for a few seconds while the runtime deletes it.
            if (await sandbox.get_info()).status.state in _GONE_STATES:
                raise _unavailable(sandbox_id)
            if self._sandbox_timeout is not None:
                await self._renew(sandbox)
        except BaseException:
            await sandbox.close()
            raise
        return sandbox

    async def _renew(self, sandbox: Sandbox) -> None:
        assert self._sandbox_timeout is not None
        try:
            await sandbox.renew(timedelta(seconds=self._sandbox_timeout))
        except SandboxApiException as error:
            # A sandbox whose timeout ran out between operations.
            if error.status_code == 404:
                raise _unavailable(sandbox.id) from error
            raise

    async def _enter_working_dir(self, sandbox: Sandbox) -> str:
        if self._working_dir is None:
            # No `cwd`: execd then starts the command in the image's working directory.
            result = await self._exec(sandbox, ['pwd', '-P'], {}, None, None)
        else:
            script = 'mkdir -p "$1" && cd "$1" && pwd -P'
            result = await self._exec(sandbox, ['sh', '-c', script, 'sh', self._working_dir], {}, None, '/')
        if result.exit_code == 0 and not result.stdout.endswith('\n'):
            raise WorkspaceUnavailableError(
                f'OpenSandbox sandbox {sandbox.id} runs an execd older than 1.1.1, which drops the line breaks from '
                "command output; set the server's `execd_image` to 1.1.1 or later."
            )
        printed = result.stdout.removesuffix('\n')
        if result.exit_code != 0 or not posixpath.isabs(printed):
            raise WorkspaceError(
                f'Could not create or enter the working directory of OpenSandbox sandbox {sandbox.id}: '
                f'exit code {result.exit_code}: {result.stderr.strip() or result.stdout!r}'
            )
        return printed

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        check_timeout(timeout)
        argv = command_argv(command, shell)
        check_env(env)
        async with self._operation():
            # The first command also creates the sandbox, which the command's timeout does not bound.
            sandbox = await self._acquire()
            assert self._resolved_working_dir is not None
            return await self._exec(sandbox, argv, {**self._env, **(env or {})}, timeout, self._resolved_working_dir)

    async def _exec(
        self, sandbox: Sandbox, argv: list[str], env: Mapping[str, str], timeout: float | None, cwd: str | None
    ) -> CommandResult:
        tag = secrets.token_hex(8)
        stdout: list[str] = []
        stderr: list[str] = []
        size = 0
        overflow = anyio.CancelScope()

        def collect(into: list[str]):
            async def handler(message: OutputMessage) -> None:
                nonlocal size
                into.append(message.text)
                size += len(message.text.encode())
                if size > _MAX_OUTPUT_BYTES:
                    overflow.cancel()

            return handler

        # Shell text rather than the SDK's argv form, which execd before 1.2 refuses; `shlex.join` keeps every
        # element one word.
        line = shlex.join(['sh', '-c', _WRAPPER, 'sh', tag, *argv])
        execution = None
        try:
            with anyio.move_on_after(timeout), overflow:
                execution = await sandbox.commands.run(
                    line,
                    opts=RunCommandOpts(working_directory=cwd, envs=dict(env) or None),
                    handlers=ExecutionHandlers(on_stdout=collect(stdout), on_stderr=collect(stderr)),
                )
        except BaseException as error:
            await self._stop(sandbox, tag)
            if isinstance(error, SandboxException):
                raise await self._translate(sandbox, error) from error
            raise
        if execution is None:
            await self._stop(sandbox, tag)
            if overflow.cancel_called:
                raise WorkspaceOutputLimitError(
                    "OpenSandbox workspace output exceeded its 10 MiB safety limit; redirect the command's output to a "
                    'file and read part of it instead',
                    limit=_MAX_OUTPUT_BYTES,
                    stdout=''.join(stdout)[:_OUTPUT_PREVIEW_CHARS],
                    stderr=''.join(stderr)[:_OUTPUT_PREVIEW_CHARS],
                )
            assert timeout is not None
            raise WorkspaceTimeoutError(
                f'Command timed out after {timeout:g} seconds', stdout=''.join(stdout), stderr=''.join(stderr)
            )
        if execution.exit_code is None or execution.exit_code < 0:
            # The wrapper exits with its command's status; only a dying sandbox kills the wrapper itself.
            await self._raise_if_gone(sandbox)
            raise WorkspaceError(f'OpenSandbox could not run the command: {execution.error}')
        return CommandResult(exit_code=execution.exit_code, stdout=''.join(stdout), stderr=''.join(stderr))

    async def _stop(self, sandbox: Sandbox, tag: str) -> None:
        # The command outlives its stream, and execd would signal only the wrapper, so a second command stops it.
        async def stop() -> None:
            await sandbox.commands.run(
                shlex.join(['sh', '-c', _STOP, 'sh', tag]), opts=RunCommandOpts(working_directory='/')
            )

        await stop_shielded(stop, grace=_STOP_TIMEOUT)

    async def _translate(self, sandbox: Sandbox, error: SandboxException) -> Exception:
        """A gone sandbox is unavailable and a request execd refused is the command's error; the rest stays transient."""
        try:
            # Only a broken stream waits for the server to catch up with a kill.
            await self._raise_if_gone(
                sandbox, settle=_GONE_SETTLE if isinstance(error, SandboxConnectionException) else 0
            )
        except WorkspaceUnavailableError as gone:
            return gone
        if isinstance(error, SandboxApiException) and error.status_code == 400:
            # execd validates the request, such as a working directory a command has since deleted: the sandbox
            # is fine, so this is the command's error.
            return WorkspaceError(f'OpenSandbox refused the command: {error}')
        return error

    async def _raise_if_gone(self, sandbox: Sandbox, *, settle: float = _GONE_SETTLE) -> None:
        """Raise `WorkspaceUnavailableError` if the sandbox is gone or going.

        A killed sandbox breaks the command stream before the server stops reporting it as running, so a running
        sandbox is asked again for up to `settle` seconds before the failure is taken as transient.
        """
        # Not `move_on_after`: with no settle time the sandbox must still be asked once.
        deadline = time.monotonic() + settle
        while True:
            try:
                info = await sandbox.get_info()
            except SandboxApiException as error:
                if error.status_code == 404:
                    raise _unavailable(sandbox.id) from error
                return
            except SandboxConnectionException:
                return
            if info.status.state in _GONE_STATES:
                raise _unavailable(sandbox.id)
            if time.monotonic() >= deadline:
                return
            await anyio.sleep(0.1)
