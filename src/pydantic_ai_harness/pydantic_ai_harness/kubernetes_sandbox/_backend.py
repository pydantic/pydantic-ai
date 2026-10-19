"""A [kubernetes-sigs/agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) sandbox behind Pydantic AI's `WorkspaceBackend` protocol.

SDK assumptions verified 2026-10-19 against `k8s-agent-sandbox` 1.0.6 and 1.0.7 and sandboxd from
agent-sandbox v1.0.7:

* `AsyncSandboxClient.create_sandbox` claims a sandbox from a `SandboxWarmPool` and waits until it is
  Ready; `get_sandbox` attaches to a claim by name; `AsyncSandbox.terminate` deletes the claim;
* `AsyncSandbox.connector.grpc_channel()` returns a `grpc.aio` channel to the pod's sandboxd, over a
  `kubectl port-forward` or an in-cluster address, depending on the connection config;
* sandboxd runs commands in its `--root-dir` (`/workspace` by default).

Re-check these before changing the integration.
"""

from __future__ import annotations

import importlib.util
import sys
import uuid
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

import anyio

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    CommandResult,
    SupportsCommands,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._workspace_provider import absolute_path, check_timeout, command_argv, running_on_asyncio
from pydantic_ai_harness.kubernetes_sandbox._process import start_command


def _missing(name: str) -> bool:
    # A `None` entry in `sys.modules` blocks the import, as `find_spec` reports.
    return sys.modules.get(name) is None and importlib.util.find_spec(name) is None


# Checked without importing: the SDK loads the generated `kubernetes` client models, so it is
# imported on first use instead.
if _missing('k8s_agent_sandbox') or _missing('grpc'):
    raise ImportError('Install `pydantic-ai-harness[kubernetes]` to use KubernetesSandbox.')

if TYPE_CHECKING:
    from k8s_agent_sandbox.async_sandbox import AsyncSandbox
    from k8s_agent_sandbox.async_sandbox_client import AsyncSandboxClient

PROVIDER = 'kubernetes'

Connection = Literal['port-forward', 'in-cluster-service', 'in-cluster-pod-ip']
"""How the client reaches a sandbox's sandboxd: a `kubectl port-forward` to the pod (from outside the
cluster), or directly from inside the cluster through the sandbox's Service or its pod IP."""


def parse_ref(ref: WorkspaceRef) -> tuple[str, str]:
    """Split a `kubernetes` ref id, `<namespace>/<claim name>`."""
    if ref.provider != PROVIDER:
        raise ValueError(f'unsupported workspace provider {ref.provider!r}; expected {PROVIDER!r}')
    namespace, separator, claim_name = ref.id.partition('/')
    if not separator or not namespace or not claim_name or '/' in claim_name:
        raise ValueError(f'a {PROVIDER!r} workspace ref id is `<namespace>/<claim name>`, got {ref.id!r}')
    return namespace, claim_name


def _client(connection: Connection) -> AsyncSandboxClient[AsyncSandbox]:
    from k8s_agent_sandbox.async_sandbox_client import AsyncSandboxClient
    from k8s_agent_sandbox.models import SandboxdInClusterConnectionConfig, SandboxdPodTunnelConnectionConfig

    if connection == 'port-forward':
        config = SandboxdPodTunnelConnectionConfig()
    else:
        config = SandboxdInClusterConnectionConfig(mode=connection)
    # `cleanup=False`: by default the SDK deletes the claims it created when the interpreter exits,
    # and a backend never destroys its environment; whoever holds the ref does.
    return AsyncSandboxClient(connection_config=config, cleanup=False)


def _require_asyncio() -> None:
    # The SDK runs on `kubernetes_asyncio`, `grpc.aio` and asyncio subprocesses.
    if not running_on_asyncio():
        raise UserError(
            '`KubernetesSandboxBackend` needs asyncio: the agent-sandbox SDK and `grpc.aio` do not run on Trio.'
        )


class KubernetesSandboxBackend(WorkspaceBackend, SupportsCommands):
    """A sandbox claimed from an agent-sandbox `SandboxWarmPool`, or attached to by its claim.

    Construction does no I/O. Without `ref`, the first operation claims a sandbox from `warm_pool`;
    with one, it attaches to that claim and raises `WorkspaceUnavailableError` if it is gone. The
    backend never deletes the claim; that is the application's job (`KubernetesSandbox.destroy`).

    Commands run through the pod's sandboxd. File operations go through those commands too: sandboxd's
    own file API serves only paths under its root and silently moves any other absolute path under
    it (`/etc/hosts` reads `/workspace/etc/hosts`), so it cannot serve the whole filesystem the
    commands see. Not tracked upstream yet.
    """

    def __init__(
        self,
        *,
        ref: WorkspaceRef | None = None,
        warm_pool: str | None = None,
        namespace: str = 'default',
        connection: Connection = 'port-forward',
        sandbox_timeout: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None:
            parse_ref(ref)
        elif warm_pool is None:
            raise ValueError('pass `warm_pool` to claim a new sandbox, or `ref` to attach to one')
        if env is not None and any(type(key) is not str or type(value) is not str for key, value in env.items()):
            raise TypeError('env keys and values must be strings')
        self._ref = ref
        self._warm_pool = warm_pool
        self._namespace = namespace
        self._connection: Connection = connection
        self._sandbox_timeout = sandbox_timeout
        # `None`: sandboxd's `--root-dir`, where it starts every command, read from the sandbox on first use.
        self._working_dir = absolute_path('working_dir', working_dir)
        self._resolved_working_dir = self._working_dir
        self._env = dict(env or {})
        self._client: AsyncSandboxClient[AsyncSandbox] | None = None
        self._sandbox: AsyncSandbox | None = None
        self._lock = anyio.Lock()
        # Set while nothing will call `aclose()`: the connection (a `kubectl port-forward` process,
        # with `connection='port-forward'`) then closes when the last operation in flight ends.
        self._close_after_operation = False
        self._operations = 0

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    async def working_dir(self) -> str:
        if self._resolved_working_dir is None:
            # sandboxd has no API for its root directory, but commands start there.
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
        _require_asyncio()
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
        argv = command_argv(command, shell)
        _require_asyncio()
        if self._working_dir is not None:
            # sandboxd starts every command in its root directory and moves a `cwd` outside it under
            # the root instead of refusing it, so another directory is entered from a shell.
            argv = ['/bin/sh', '-c', 'cd -- "$1" || exit 1; shift; exec "$@"', 'sh', self._working_dir, *argv]
        async with self._operation():
            async with self._lock:
                sandbox = await self._acquire()
            try:
                channel = await sandbox.connector.grpc_channel()
                return await start_command(channel, argv, env={**self._env, **(env or {})}, timeout=timeout)
            except (WorkspaceTimeoutError, WorkspaceUnavailableError):
                raise
            except Exception as error:
                await self._raise_if_gone(sandbox, error)
                raise

    async def aclose(self) -> None:
        """Close the connection to the sandbox (including a `kubectl port-forward`); the sandbox keeps running."""
        async with self._lock:
            client, self._client, self._sandbox = self._client, None, None
        if client is not None:
            with anyio.CancelScope(shield=True):
                await client.close()

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[None, None]:
        self._operations += 1
        try:
            yield
        finally:
            self._operations -= 1
            if self._operations == 0 and self._close_after_operation:
                await self.aclose()

    async def _acquire(self) -> AsyncSandbox:
        """Claim or attach on first use; the caller holds `_lock`, so concurrent first operations share one claim."""
        if self._sandbox is not None:
            return self._sandbox
        from k8s_agent_sandbox.exceptions import (
            SandboxClaimFailedError,
            SandboxNotFoundError,
            SandboxWarmPoolNotFoundError,
        )

        if self._client is None:
            self._client = _client(self._connection)
        if self._ref is not None:
            namespace, claim_name = parse_ref(self._ref)
            try:
                sandbox = await self._client.get_sandbox(claim_name, namespace=namespace)
            except SandboxNotFoundError as error:
                raise WorkspaceUnavailableError(f'Kubernetes sandbox {self._ref.id!r} no longer exists') from error
            self._sandbox = sandbox
            return sandbox
        assert self._warm_pool is not None
        # Named here, not by the SDK, so the ref is known even if this task is cancelled while the
        # claim waits to become Ready; the SDK leaves an explicitly named claim in place on failure.
        ref = WorkspaceRef(provider=PROVIDER, id=f'{self._namespace}/pydantic-ai-{uuid.uuid4().hex[:16]}')
        try:
            # The SDK types its unused `volume_claim_templates` parameter as a bare `list[dict]`.
            sandbox = await self._client.create_sandbox(  # pyright: ignore[reportUnknownMemberType]
                self._warm_pool,
                namespace=self._namespace,
                claim_name=parse_ref(ref)[1],
                shutdown_after_seconds=self._sandbox_timeout,
            )
        except anyio.get_cancelled_exc_class():
            self._ref = ref
            raise
        except (SandboxWarmPoolNotFoundError, SandboxClaimFailedError) as error:
            raise WorkspaceUnavailableError(
                f'Could not claim a sandbox from warm pool {self._warm_pool!r}: {error}'
            ) from error
        self._ref = ref
        self._sandbox = sandbox
        return sandbox

    async def _raise_if_gone(self, sandbox: AsyncSandbox, error: Exception) -> None:
        """Raise `WorkspaceUnavailableError` if the sandbox is gone; otherwise leave `error` to propagate."""
        # A lost connection looks the same whether the pod was deleted or the network blinked, so
        # ask the API server. A sandbox that is still Ready means a transient failure that durable
        # engines may retry.
        state, message = 'SandboxReady', ''
        with anyio.CancelScope(shield=True), anyio.move_on_after(10):
            try:
                state, message = await sandbox.status()
            except Exception:
                # The API server cannot say either: keep the original error.
                pass
        if state != 'SandboxReady':
            raise WorkspaceUnavailableError(
                f'Kubernetes sandbox {sandbox.claim_name!r} is gone ({state}: {message})'
            ) from error


async def destroy_sandbox(ref: WorkspaceRef, *, connection: Connection = 'port-forward') -> None:
    """Delete the claim a ref names, and with it the sandbox; a claim that is already gone returns quietly."""
    from k8s_agent_sandbox.exceptions import SandboxNotFoundError

    namespace, claim_name = parse_ref(ref)
    client = _client(connection)
    try:
        # Workaround: `AsyncSandboxClient.delete_sandbox` logs and swallows every error, so a failed
        # delete would look like success. Attaching and calling `terminate()` raises instead.
        # Not tracked upstream yet (the OpenAI integration hit the same in #1932).
        try:
            sandbox = await client.get_sandbox(claim_name, namespace=namespace)
        except SandboxNotFoundError:
            return
        await sandbox.terminate()
    finally:
        with anyio.CancelScope(shield=True):
            await client.close()
