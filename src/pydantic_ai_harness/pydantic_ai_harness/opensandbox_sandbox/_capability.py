"""Capability that supplies an OpenSandbox sandbox to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace import innermost_backend
from pydantic_ai_harness._workspace_provider import check_env, check_integer, check_working_dir
from pydantic_ai_harness.opensandbox_sandbox._backend import PROVIDER, OpenSandboxBackend, destroy_sandbox

if TYPE_CHECKING:
    from opensandbox.config import ConnectionConfig

    from pydantic_ai.agent import AgentRunResult


class _SuppliedBackend(OpenSandboxBackend):
    """The backend `OpenSandbox` built for one run; the capability closes it when that run ends."""

    def __init__(self, *, run_id: str | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_id = run_id
        # Until a run claims it, nothing closes this backend: Temporal builds one per activity and
        # drops it, so it closes its transport after each operation instead.
        self._close_after_operation = True

    def claim(self, owned: bool) -> None:
        """Keep the transport open across operations while the run that closes it is in progress."""
        self._close_after_operation = not owned


@dataclass(kw_only=True)
class OpenSandbox(AbstractCapability[AgentDepsT]):
    """Supply an [OpenSandbox](https://open-sandbox.ai) sandbox as the run's workspace, created on first use.

    ```python {test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai_harness import Coder, OpenSandbox

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[OpenSandbox(image='python:3.13-slim'), Coder()])
    ```

    A run with no reference creates a fresh sandbox on your OpenSandbox server, on whichever runtime it uses
    (Docker or Kubernetes). Pass a `WorkspaceRef` to attach to a sandbox created earlier. Ending a run does not
    delete the sandbox: remove it with [`destroy`][pydantic_ai_harness.opensandbox_sandbox.OpenSandbox.destroy],
    or set `sandbox_timeout`. Commands run under `sh -c` as the image's user. Outbound network access is the
    server's policy.

    This capability supplies the workspace only. Compose it with capabilities that use it, such as `Coder`,
    `Shell`, and `FileSystem`. See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.
    """

    image: str | None = None
    """The image for a new sandbox, such as `'python:3.13-slim'`; it needs a POSIX `sh` and the usual file utilities.

    Without one, the capability only attaches to and destroys sandboxes by ref, and leaves a run without a ref
    to the agent's other workspace capabilities.
    """

    connection: ConnectionConfig | None = field(default=None, repr=False)
    """The `opensandbox.config.ConnectionConfig` for your server. When `None`, each run reads
    `OPEN_SANDBOX_DOMAIN` and `OPEN_SANDBOX_API_KEY`. On Kubernetes without the ingress gateway, pass
    `ConnectionConfig(use_server_proxy=True)`."""

    resource: Mapping[str, str] | None = None
    """Resource limits for a new sandbox, such as `{'cpu': '500m', 'memory': '1Gi'}`; `None` keeps the SDK's
    default of one CPU and 2 GiB."""

    sandbox_timeout: int | None = None
    """Seconds a sandbox may go unused before the server deletes it, files included. An operation that starts once
    half of it has passed renews it, so a single command must finish within it. `None` keeps it until it is destroyed."""

    working_dir: str | None = '/workspace'
    """Absolute directory commands start in and relative paths resolve against, created on first use if
    missing; `None` uses the image's own."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, also in an attached sandbox; a command's `env` is layered on
    top. Nothing is read from the host."""

    def __post_init__(self) -> None:
        if self.defer_loading:
            # Core picks the run's workspace from the always-on capabilities only.
            raise UserError(
                '`OpenSandbox` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.'
            )
        # Checked here rather than at the first workspace operation, so a bad value fails where it is written.
        check_working_dir(self.working_dir)
        check_integer('sandbox_timeout', self.sandbox_timeout, optional=True)
        check_env(self.env)

    def backend(self, ref: WorkspaceRef) -> OpenSandboxBackend:
        """Construct a lazy backend for an existing sandbox."""
        return OpenSandboxBackend(
            ref=ref,
            connection=self.connection,
            working_dir=self.working_dir,
            env=self.env,
            sandbox_timeout=self.sandbox_timeout,
        )

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Delete a sandbox by ID, files included; one that no longer exists returns quietly."""
        if ref.provider != PROVIDER:
            raise ValueError(f'unsupported workspace provider {ref.provider!r}; expected {PROVIDER!r}')
        await destroy_sandbox(ref.id, self.connection)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it creates or attaches on first use."""
        if ref is not None and ref.provider != PROVIDER:
            return None
        if ref is None and self.image is None:
            return None
        return _SuppliedBackend(
            run_id=ctx.run_id,
            image=self.image if ref is None else None,
            ref=ref,
            connection=self.connection,
            working_dir=self.working_dir,
            env=self.env,
            resource=self.resource,
            sandbox_timeout=self.sandbox_timeout,
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
