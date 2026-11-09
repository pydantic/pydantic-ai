"""A [kubernetes-sigs/agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

SDK assumptions verified 2026-11-02 against `k8s-agent-sandbox` 1.0.9, sandboxd from agent-sandbox
v1.0.7, and a kind cluster running the v1.0.7 controller, and re-checked 2026-11-09 against SDK 1.0.10
(none of the calls below changed) and the v1.0.10 controller:

* `AsyncSandboxClient.create_sandbox` claims a sandbox from a `SandboxWarmPool`, waits until it is
  Ready, and deletes the claim it named if that fails or is cancelled once (it awaits the delete
  under `asyncio.shield`, which a second cancellation abandons); `get_sandbox` attaches to a
  claim by name, waits 30 seconds for a claim that does not exist, and raises `SandboxNotFoundError`
  for any failed lookup;
* `client.k8s_helper.get_sandbox_claim` and `get_sandbox` return `None` only for a 404, and
  `delete_sandbox_claim` ignores only a 404; the Sandbox object they read is the one a claim binds;
* `AsyncSandbox.commands.start()` runs a string under `/bin/sh -c` over sandboxd's streaming
  `ProcessService.Start`, reached over a `kubectl port-forward` or an in-cluster address depending
  on the connection config, and sends the config's `auth_token`. It restarts a port-forward that
  died before each command, and re-raises gRPC errors as `RuntimeError`;
* closing the handle cancels the call, which SIGKILLs the command's process group (a `setsid` child
  may survive, which the workspace protocol allows); a command killed by signal n exits 128+n;
* sandboxd runs commands in its `--root-dir` (`/workspace` by default), and when a background child
  keeps stdout open, ends the stream about 5 seconds after the command exits;
* deleting a claim deletes its Sandbox, then its pod: sandboxd passes the pod's SIGTERM on to the
  commands it runs and reports them as exiting 143. A deleted pod whose claim remains is recreated.

Re-check these before changing the integration.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal, TypeVar, get_args

import anyio

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
    EXEC_SCRIPT,
    absolute_path,
    check_env_types,
    check_provider,
    check_timeout,
    command_argv,
    require_asyncio,
    run_to_completion,
)


def _missing(name: str) -> bool:
    # A `None` entry in `sys.modules` blocks the import, as `find_spec` reports.
    return sys.modules.get(name) is None and importlib.util.find_spec(name) is None


# Checked without importing: the SDK loads the generated `kubernetes` client models, so it is
# imported on first use instead.
if _missing('k8s_agent_sandbox') or _missing('grpc'):
    raise ImportError('Install `pydantic-ai-harness[kubernetes]` to use KubernetesSandbox.')

logger = logging.getLogger(__name__)

PROVIDER = 'kubernetes'

DEFAULT_SANDBOX_TIMEOUT = 24 * 60 * 60

ConnectionMode = Literal['port-forward', 'in-cluster-service', 'in-cluster-pod-ip']
"""How the client reaches a sandbox's sandboxd: a `kubectl port-forward` to the pod (from outside the
cluster), or directly from inside the cluster through the sandbox's Service or its pod IP."""

_InClusterMode = Literal['in-cluster-service', 'in-cluster-pod-ip']

_IN_CLUSTER_MODES: dict[_InClusterMode, _InClusterMode] = {
    'in-cluster-service': 'in-cluster-service',
    'in-cluster-pod-ip': 'in-cluster-pod-ip',
}
"""Our in-cluster mode names to `SandboxdInClusterConnectionConfig.mode`'s. They match today, but the
SDK renamed its values once (v1.0.6), and our names are public API: a rename upstream changes only
this map's values. The keys are checked against our modes, and the values against the SDK's own
`Literal` where `_client` passes them, so type checking catches a rename on either side."""

if TYPE_CHECKING:
    from typing import TypeAlias

    from k8s_agent_sandbox.async_sandbox import AsyncSandbox
    from k8s_agent_sandbox.async_sandbox_client import AsyncSandboxClient
    from k8s_agent_sandbox.models import SandboxdInClusterConnectionConfig, SandboxdPodTunnelConnectionConfig

    Connection: TypeAlias = ConnectionMode | SandboxdPodTunnelConnectionConfig | SandboxdInClusterConnectionConfig
    """A connection mode with the SDK's default ports and timeouts, or the SDK's own sandboxd connection config."""


def connection_problem(connection: object) -> str | None:
    """Say what is wrong with a `connection` argument, or return `None` when the SDK reaches sandboxd with it."""
    if isinstance(connection, str):
        if connection in get_args(ConnectionMode):
            return None
        return f'connection must be one of {", ".join(map(repr, get_args(ConnectionMode)))}, got {connection!r}.'
    # Whoever built a config object has imported the SDK already, so this import costs nothing.
    from k8s_agent_sandbox.models import SandboxdInClusterConnectionConfig, SandboxdPodTunnelConnectionConfig

    if isinstance(connection, SandboxdPodTunnelConnectionConfig | SandboxdInClusterConnectionConfig):
        return None
    return (
        'connection must be a connection mode, a `SandboxdPodTunnelConnectionConfig`, or a '
        f'`SandboxdInClusterConnectionConfig`, got {type(connection).__name__}: commands run through '
        'sandboxd, which the SDK reaches only with those configs.'
    )


def parse_ref(ref: WorkspaceRef) -> tuple[str, str]:
    """Split a `kubernetes` ref id, `<namespace>/<claim name>`."""
    check_provider(ref, PROVIDER)
    namespace, separator, claim_name = ref.id.partition('/')
    if not separator or not namespace or not claim_name or '/' in claim_name:
        raise ValueError(f'a {PROVIDER!r} workspace ref id is `<namespace>/<claim name>`, got {ref.id!r}')
    return namespace, claim_name


def _client(connection: Connection) -> AsyncSandboxClient[AsyncSandbox]:
    from k8s_agent_sandbox.async_sandbox_client import AsyncSandboxClient
    from k8s_agent_sandbox.models import SandboxdInClusterConnectionConfig, SandboxdPodTunnelConnectionConfig

    if connection == 'port-forward':
        config = SandboxdPodTunnelConnectionConfig()
    elif isinstance(connection, str):
        config = SandboxdInClusterConnectionConfig(mode=_IN_CLUSTER_MODES[connection])
    else:
        config = connection
    # `cleanup=False`: by default the SDK deletes the claims it created when the interpreter exits,
    # and a backend never destroys its environment; whoever holds the ref does. This also skips the
    # SDK's exit hook that stops `kubectl port-forward` processes, so `aclose()` is what stops them.
    return AsyncSandboxClient(connection_config=config, cleanup=False)


_ARGUMENT_PREFIX = 'PYDANTIC_AI_HARNESS_ARG_'
"""Names the environment variables that carry a command's arguments into its script; unset before it runs."""

# Output is collected in this process, so `yes` or a large `cat` stops at the cap instead of growing
# it. sandboxd's own output limits cover only its unary `Execute` RPCs, not the streaming `Start`
# that `commands.start()` uses (https://github.com/kubernetes-sigs/agent-sandbox/pull/2117).
_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_OUTPUT_PREVIEW_BYTES = 64 * 1024


async def _run_on_sandbox(
    sandbox: AsyncSandbox, script: str, *, env: Mapping[str, str], timeout: float | None
) -> CommandResult:
    """Run `script` with the SDK's streaming `commands.start()`, returning its complete output.

    Raises `WorkspaceTimeoutError` with the output that arrived when `timeout` passes, and
    `WorkspaceOutputLimitError` when the output passes 10 MiB. Either way, closing the handle
    cancels the stream, and sandboxd kills the command's process group.
    """
    from k8s_agent_sandbox.models import ProcessExitEvent

    commands = sandbox.commands
    # The SDK clears `commands` on close, and the backend detaches its handle before closing it.
    assert commands is not None
    stdout = bytearray()
    stderr = bytearray()
    exit_code: int | None = None
    with anyio.move_on_after(timeout) as scope:
        try:
            async with await commands.start(script, env=dict(env)) as handle:
                async for event in handle.events():
                    if isinstance(event, ProcessExitEvent):
                        exit_code = event.exit_code
                        continue
                    (stdout if event.type == 'stdout' else stderr).extend(event.data)
                    if len(stdout) + len(stderr) > _MAX_OUTPUT_BYTES:
                        raise WorkspaceOutputLimitError(
                            'Command output exceeded the 10 MiB limit; redirect it to a file and read part of it instead',
                            limit=_MAX_OUTPUT_BYTES,
                            stdout=stdout[:_OUTPUT_PREVIEW_BYTES].decode(errors='replace'),
                            stderr=stderr[:_OUTPUT_PREVIEW_BYTES].decode(errors='replace'),
                        )
        except asyncio.CancelledError:
            # Closing a channel cancels its calls, so another command's lost connection can end this
            # one with a cancellation no one asked for. Raised as the connection error it is.
            task = asyncio.current_task()
            if task is not None and not task.cancelling():
                raise ConnectionError('The connection to sandboxd closed while the command was running') from None
            raise
    if scope.cancelled_caught:
        raise WorkspaceTimeoutError(
            f'Command timed out after {timeout:g} seconds',
            stdout=stdout.decode(errors='replace'),
            stderr=stderr.decode(errors='replace'),
        )
    # `events()` raises when the stream ends without an exit event.
    assert exit_code is not None
    return CommandResult(
        exit_code=exit_code, stdout=stdout.decode(errors='replace'), stderr=stderr.decode(errors='replace')
    )


# The SDK runs on `kubernetes_asyncio`, `grpc.aio` and asyncio subprocesses.
_ASYNCIO_ONLY = '`KubernetesSandboxBackend` needs asyncio: the agent-sandbox SDK and `grpc.aio` do not run on Trio.'


_GONE_CHECK_TIMEOUT = 10
"""Seconds to wait for the API server to say whether a sandbox still exists after a command fails."""

_T = TypeVar('_T')


async def _cancel_once(call: Callable[[], Awaitable[_T]]) -> _T:
    """Await `call` in a task of its own; if the caller is cancelled, cancel that task once and wait for it.

    A cancelled `create_sandbox` deletes its claim under `asyncio.shield`. An AnyIO cancel scope
    cancels the waiting task again at every await, which abandons that delete; one cancellation lets it finish.
    """
    task = asyncio.ensure_future(call())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.cancel()
        with anyio.CancelScope(shield=True), anyio.move_on_after(30):
            await asyncio.wait([task])
        raise


class KubernetesSandboxBackend(WorkspaceBackend, SupportsCommands):
    """A sandbox claimed from an agent-sandbox `SandboxWarmPool`, or attached to by its claim.

    Construction does no I/O. Without `ref`, the first operation claims a sandbox from `warm_pool`;
    with one, it attaches to that claim and raises `WorkspaceUnavailableError` if it is gone. The
    backend never deletes the claim; call `KubernetesSandbox.destroy(ref)` or rely on `sandbox_timeout`.

    Commands run through the pod's sandboxd, and file operations run as shell commands, so they see
    the same filesystem. The image needs `/bin/sh` and coreutils.
    """

    # File operations don't use sandboxd's file API: it serves only paths under its root and moves
    # any other absolute path under it (`/etc/hosts` reads `/workspace/etc/hosts`). Not tracked upstream yet.

    def __init__(
        self,
        *,
        ref: WorkspaceRef | None = None,
        warm_pool: str | None = None,
        namespace: str = 'default',
        connection: Connection = 'port-forward',
        sandbox_timeout: int | None = DEFAULT_SANDBOX_TIMEOUT,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None:
            parse_ref(ref)
        elif warm_pool is None:
            raise ValueError('pass `warm_pool` to claim a new sandbox, or `ref` to attach to one')
        check_env_types(env)
        if problem := connection_problem(connection):
            raise ValueError(problem)
        self._ref = ref
        self._warm_pool = warm_pool
        self._namespace = namespace
        self._connection: Connection = connection
        self._sandbox_timeout = sandbox_timeout
        # `None`: sandboxd's `--root-dir`, where it starts every command.
        self._working_dir = absolute_path('working_dir', working_dir)
        # The directory commands start in as the sandbox resolves it (`pwd -P`), read on first use:
        # the protocol reports a canonical path, and a configured one may go through a symlink.
        self._resolved_working_dir: str | None = None
        self._missing_cwd_marker = f'pydantic-ai-missing-working-dir-{uuid.uuid4().hex}'
        self._env = dict(env or {})
        self._client: AsyncSandboxClient[AsyncSandbox] | None = None
        self._sandbox: AsyncSandbox | None = None
        self._lock = anyio.Lock()
        # Set while nothing will call `aclose()`: the connection (a `kubectl port-forward` process,
        # with `connection='port-forward'`) then closes when the last operation in flight ends.
        self._close_after_operation = False
        # An `aclose()` that came while operations were in flight; the last of them closes the connection once.
        self._close_requested = False
        self._operations = 0

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    async def working_dir(self) -> str:
        if self._resolved_working_dir is None:
            # sandboxd has no API for its root directory, but commands start there (or in `working_dir`).
            result = await self.run(['pwd', '-P'])
            if result.exit_code != 0 or not result.stdout.startswith('/'):
                raise WorkspaceUnavailableError(
                    f'Could not read the sandbox working directory: {result.stderr.strip()}'
                )
            self._resolved_working_dir = result.stdout.rstrip('\n')
        return self._resolved_working_dir

    async def get_sandbox(self) -> AsyncSandbox:
        """Return the SDK's `AsyncSandbox` handle, claiming or attaching first like any operation.

        The handle's connection belongs to this backend: after `aclose()`, call this again.
        """
        require_asyncio(_ASYNCIO_ONLY)
        async with self._lock:
            return await self._acquire()

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        check_timeout(timeout)
        check_env_types(env)
        argv = command_argv(command, shell)
        require_asyncio(_ASYNCIO_ONLY)
        # `commands.start()` runs one string under sandboxd's `/bin/sh -c`. Each argument goes in its
        # own environment variable, which the script moves into its positional parameters and unsets,
        # so each is still one `execve` string under Linux's 128 KiB limit per string, as it would
        # be in an argv; quoted into the script, they would share that limit. Workaround: sandboxd's
        # `Start` leaves stdin open, where the protocol wants it at EOF, so the exec script redirects
        # it. sandboxd also moves a `cwd` outside its root under the root instead of refusing it, so
        # another directory is entered from the shell. Not tracked upstream yet.
        if self._working_dir is None:
            arguments, enter = argv, ''
        else:
            # A missing `working_dir` exits 125 with only the marker on stderr, before the command runs.
            arguments = [self._working_dir, *argv]
            enter = f'cd -- "$1" 2>/dev/null || {{ printf %s {self._missing_cwd_marker} >&2; exit 125; }}\nshift\n'
        names = [f'{_ARGUMENT_PREFIX}{index}' for index in range(len(arguments))]
        references = ' '.join(f'"${name}"' for name in names)
        script = f'set -- {references}\nunset {" ".join(names)}\n{enter}{EXEC_SCRIPT}'
        command_env = {**self._env, **(env or {}), **dict(zip(names, arguments))}
        async with self._operation():
            async with self._lock:
                sandbox = await self._acquire()
            try:
                result = await _run_on_sandbox(sandbox, script, env=command_env, timeout=timeout)
            except (WorkspaceTimeoutError, WorkspaceOutputLimitError):
                raise
            except Exception as error:
                await self._raise_if_gone(sandbox, error)
                raise
            if result.exit_code == 125 and not result.stdout and result.stderr == self._missing_cwd_marker:
                assert self._ref is not None
                raise WorkspaceError(
                    f'working_dir {self._working_dir!r} does not exist in Kubernetes sandbox {self._ref.id!r}. '
                    'Create it there, or pass a working_dir that exists.'
                )
            if result.exit_code > 128 or result.exit_code < 0:
                # Deleting a sandbox stops its pod with SIGTERM, and sandboxd reports the command it
                # interrupted as an ordinary exit, 143. A command killed by a signal may be one of those.
                # sandboxd before v1.0.7 reports a signal death as -1, as `ProcessExitEvent` still documents.
                await self._raise_if_gone(sandbox, None)
            return result

    async def aclose(self) -> None:
        """Close the connection to the sandbox (including a `kubectl port-forward`); the sandbox keeps running."""
        if self._operations:
            # Closing the channel would cancel the commands still using it: the last one closes it.
            self._close_requested = True
            return

        async def detach_and_close() -> None:
            async with self._lock:
                client, self._client, self._sandbox = self._client, None, None
            if client is not None:
                await client.close()

        # Also called from a `finally:` after the run was cancelled, where waiting for the lock
        # would raise before anything closed.
        await run_to_completion(detach_and_close)

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[None, None]:
        self._operations += 1
        try:
            yield
        finally:
            self._operations -= 1
            if self._operations == 0:
                close, self._close_requested = self._close_after_operation or self._close_requested, False
                if close and self._client is not None:
                    # Detached before any await, so an operation starting meanwhile opens its own client.
                    client, self._client, self._sandbox = self._client, None, None
                    await run_to_completion(client.close)

    async def _acquire(self) -> AsyncSandbox:
        """Claim or attach on first use; the caller holds `_lock`, so concurrent first operations share one claim."""
        if self._sandbox is not None:
            return self._sandbox
        from k8s_agent_sandbox.exceptions import SandboxClaimFailedError, SandboxWarmPoolNotFoundError

        if self._client is None:
            self._client = _client(self._connection)
        if self._ref is not None:
            namespace, claim_name = parse_ref(self._ref)
            # Workaround: `get_sandbox` watches a missing claim for 30 seconds before failing, and then
            # raises the `SandboxNotFoundError` it raises for any failed lookup, an unreachable API server
            # too. The API server's answer is immediate, and only its 404 means gone for good; any other
            # failure of `get_sandbox` propagates for the caller to retry. Not tracked upstream yet.
            if await self._client.k8s_helper.get_sandbox_claim(claim_name, namespace) is None:
                raise WorkspaceUnavailableError(f'Kubernetes sandbox {self._ref.id!r} no longer exists')
            sandbox = await self._client.get_sandbox(claim_name, namespace=namespace)
            self._sandbox = sandbox
            return sandbox
        client, warm_pool = self._client, self._warm_pool
        assert warm_pool is not None

        async def claim() -> AsyncSandbox:
            # Named by the SDK, which deletes the claim again if this fails or is cancelled before it is Ready.
            sandbox = await client.create_sandbox(
                warm_pool, namespace=self._namespace, shutdown_after_seconds=self._sandbox_timeout
            )
            # Typed optional because the SDK clears it when the handle is deleted; a new handle has it.
            claim_name = sandbox.claim_name
            assert claim_name is not None
            self._ref = WorkspaceRef(provider=PROVIDER, id=f'{self._namespace}/{claim_name}')
            self._sandbox = sandbox
            # Logged at claim time, so the claim can be found whatever ends the run before its ref is stored.
            logger.info('Claimed Kubernetes sandbox %s', self._ref.id)
            return sandbox

        try:
            return await _cancel_once(claim)
        except (SandboxWarmPoolNotFoundError, SandboxClaimFailedError) as error:
            raise WorkspaceUnavailableError(
                f'Could not claim a sandbox from warm pool {warm_pool!r}: {error}'
            ) from error

    async def _raise_if_gone(self, sandbox: AsyncSandbox, error: Exception | None) -> None:
        """Raise `WorkspaceUnavailableError` if the sandbox is gone; otherwise return, leaving the outcome as it is."""
        # A lost connection looks the same whether the pod was deleted or the network blinked, so
        # ask the API server. Only a deleted Sandbox is gone for good: one whose pod the controller is
        # recreating still exists and comes back, so its failure is one durable engines may retry.
        # The helper's `None` for a 404 is the same check `status()` makes, without its status strings.
        assert self._ref is not None
        gone = False
        with anyio.move_on_after(_GONE_CHECK_TIMEOUT, shield=True) as scope:
            try:
                gone = await sandbox.k8s_helper.get_sandbox(sandbox.sandbox_id, sandbox.namespace) is None
            except Exception:
                logger.warning(
                    'Could not ask the API server whether Kubernetes sandbox %s still exists',
                    self._ref.id,
                    exc_info=True,
                )
        if scope.cancelled_caught:
            logger.warning(
                'The API server did not say within %g seconds whether Kubernetes sandbox %s exists',
                _GONE_CHECK_TIMEOUT,
                self._ref.id,
            )
        if gone:
            raise WorkspaceUnavailableError(
                f'Kubernetes sandbox {self._ref.id!r} is gone: its Sandbox was deleted'
            ) from error


async def destroy_sandbox(ref: WorkspaceRef) -> None:
    """Delete the claim a ref names, and with it the sandbox; a claim that is already gone returns quietly."""
    namespace, claim_name = parse_ref(ref)
    require_asyncio(_ASYNCIO_ONLY)
    client = _client('port-forward')  # Talks only to the API server, so the connection is never used.
    try:
        # Workaround: `AsyncSandboxClient.delete_sandbox` logs and swallows every error, so a failed
        # delete would look like success. The SDK's Kubernetes helper ignores only a 404. Use
        # `delete_sandbox` once it raises (https://github.com/kubernetes-sigs/agent-sandbox/pull/1856).
        await client.k8s_helper.delete_sandbox_claim(claim_name, namespace)
    finally:
        await run_to_completion(client.close)
