"""Capability that supplies a kubernetes-sigs/agent-sandbox sandbox to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace import innermost_backend
from pydantic_ai_harness._workspace_provider import check_integer, check_working_dir
from pydantic_ai_harness.kubernetes_sandbox._backend import (
    DEFAULT_SANDBOX_TIMEOUT,
    PROVIDER,
    KubernetesSandboxBackend,
    connection_problem,
    destroy_sandbox,
)

if TYPE_CHECKING:
    from pydantic_ai.agent import AgentRunResult
    from pydantic_ai_harness.kubernetes_sandbox._backend import Connection


class _SuppliedBackend(KubernetesSandboxBackend):
    """The backend `KubernetesSandbox` built for one run; the capability closes its connection when that run ends."""

    def __init__(self, *, run_id: str | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_id = run_id
        # Until a run claims it, nothing closes this backend: Temporal builds one per activity and
        # drops it, so it closes its connection after each operation instead.
        self._close_after_operation = True

    def claim(self, owned: bool) -> None:
        """Keep the connection open across operations while the run that closes it is in progress."""
        self._close_after_operation = not owned


@dataclass(kw_only=True)
class KubernetesSandbox(AbstractCapability[AgentDepsT]):
    """Run the agent's workspace in a [Kubernetes agent sandbox](https://agent-sandbox.sigs.k8s.io).

    A run with no reference claims a sandbox from `warm_pool` on first use; pass a `WorkspaceRef`
    to attach to an existing claim. Ending a run does not delete the sandbox; `destroy(ref)` does,
    and `sandbox_timeout` deletes it in any case. Commands run in the pod's sandboxd container, so its
    image decides which tools are available.

    This capability supplies the workspace only. Compose it with capabilities that use it, such
    as `Coder`, `Shell`, and `FileSystem`. See
    [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.
    """

    warm_pool: str
    """Name of the `SandboxWarmPool` new sandboxes are claimed from. Its `SandboxTemplate` must run sandboxd."""

    namespace: str = 'default'
    """Kubernetes namespace of the warm pool and of the claims made from it."""

    connection: Connection = 'port-forward'
    """How to reach the sandbox. `'port-forward'` (default) runs `kubectl port-forward` from outside the
    cluster and doesn't work on gVisor or Kata. `'in-cluster-pod-ip'` and `'in-cluster-service'` connect
    directly from a pod in the cluster. For other ports or timeouts, pass the SDK's
    `SandboxdPodTunnelConnectionConfig` or `SandboxdInClusterConnectionConfig`."""

    sandbox_timeout: int | None = DEFAULT_SANDBOX_TIMEOUT
    """Seconds after which a newly claimed sandbox is deleted, whether or not anyone still uses it;
    `None` keeps it until `destroy(ref)`. Defaults to 24 hours."""

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; `None` is sandboxd's
    root directory (its `--root-dir`, `/workspace` by default). It must already exist."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, on top of the container's own; a command's `env` is layered on top."""

    def __post_init__(self) -> None:
        if self.defer_loading:
            # Core picks the run's workspace from the always-on capabilities only.
            raise UserError(
                '`KubernetesSandbox` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.'
            )
        # Checked here rather than at the first workspace operation, so a bad value fails where it is written.
        check_integer('sandbox_timeout', self.sandbox_timeout, optional=True)
        check_working_dir(self.working_dir)
        if problem := connection_problem(self.connection):
            raise UserError(problem)

    def backend(self, ref: WorkspaceRef) -> KubernetesSandboxBackend:
        """Construct a lazy backend for an existing claim; close its connection with `aclose()`."""
        return KubernetesSandboxBackend(ref=ref, connection=self.connection, working_dir=self.working_dir, env=self.env)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Delete the claim, and with it the sandbox, without running anything in it.

        A claim that no longer exists is already gone, so destroying it returns quietly.
        """
        await destroy_sandbox(ref)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it claims or attaches on first use."""
        if ref is not None and ref.provider != PROVIDER:
            return None
        return _SuppliedBackend(
            run_id=ctx.run_id,
            ref=ref,
            warm_pool=self.warm_pool,
            namespace=self.namespace,
            connection=self.connection,
            sandbox_timeout=self.sandbox_timeout,
            working_dir=self.working_dir,
            env=self.env,
        )

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        # Only the backend this run built: one passed in as `workspace=`, such as a parent run's
        # handed to a subagent, belongs to the run that built it and may still be in use.
        backend = innermost_backend(ctx.workspace)
        owned = backend if isinstance(backend, _SuppliedBackend) and backend.run_id == ctx.run_id else None
        if owned is not None:
            owned.claim(True)
        try:
            return await handler()
        finally:
            if owned is not None:
                # `result.workspace` outlives the run, with no one left to close it.
                owned.claim(False)
                await owned.aclose()
