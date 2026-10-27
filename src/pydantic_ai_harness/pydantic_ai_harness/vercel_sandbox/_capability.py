"""Capability that supplies a Vercel Sandbox to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace import innermost_backend
from pydantic_ai_harness._workspace_provider import check_env_types, check_integer, check_working_dir
from pydantic_ai_harness.vercel_sandbox._backend import DEFAULT_SANDBOX_TIMEOUT, VercelSandboxBackend, destroy_sandbox

if TYPE_CHECKING:
    from vercel.sandbox import SandboxClient

    from pydantic_ai.agent import AgentRunResult


class _SuppliedBackend(VercelSandboxBackend):
    """The backend `VercelSandbox` built for one run; the capability closes its client when that run ends."""

    def __init__(self, *, run_id: str | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_id = run_id
        # Until a run claims it, nothing closes this backend: Temporal builds one per activity and
        # drops it, so its own client closes after each operation instead.
        self._close_after_operation = True

    def claim(self, owned: bool) -> None:
        """Keep the client open across operations while the run that closes it is in progress."""
        self._close_after_operation = not owned


@dataclass(kw_only=True)
class VercelSandbox(AbstractCapability[AgentDepsT]):
    """Supply an isolated [Vercel Sandbox](https://vercel.com/docs/vercel-sandbox) as the run's workspace.

    A run with no reference creates a fresh sandbox on first use; pass a `WorkspaceRef` to attach
    to an existing one. Ending a run does not delete the sandbox: Vercel stops its session when
    `sandbox_timeout` runs out, saves its files, and keeps it until you delete it. Commands run
    under `sh -c` as the sandbox's default user.

    This capability supplies the workspace only. Compose it with capabilities that use it, such
    as `Coder`, `Shell`, and `FileSystem`. See
    [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.
    """

    client: SandboxClient | None = None
    """A `vercel.sandbox.SandboxClient` to share across runs on one event loop; you close it. When
    `None`, each run opens one that reads its credentials from the environment, and closes it when
    the run ends."""

    image: str | None = None
    """Image a newly created sandbox runs; Vercel's default, `vercel/sandbox/universal`, when `None`.
    An unknown image raises `WorkspaceUnavailableError` on first use."""

    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT
    """Seconds each session of a newly created sandbox runs before Vercel stops it and saves its files.

    The next operation resumes the sandbox in a new session. The default, 45 minutes, is the most
    Vercel's Hobby plan allows; Pro and Enterprise allow up to 86400.
    """

    allow_internet_access: bool = True
    """Whether a newly created sandbox may reach the internet."""

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; `None` uses the sandbox's own.
    Created on a new sandbox; an attached sandbox must already have it."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, also on an attached sandbox; a command's `env` is
    layered on top. Nothing is read from the host."""

    def __post_init__(self) -> None:
        if self.defer_loading:
            # Core picks the run's workspace from the always-on capabilities only.
            raise UserError(
                '`VercelSandbox` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.'
            )
        # Checked here rather than at the first workspace operation, so a bad value fails where it is written.
        check_working_dir(self.working_dir)
        check_integer('sandbox_timeout', self.sandbox_timeout)
        try:
            check_env_types(self.env)
        except TypeError:
            raise UserError('`env` keys and values must be strings.') from None

    def backend(self, ref: WorkspaceRef) -> VercelSandboxBackend:
        """Construct a lazy backend for an existing sandbox."""
        return VercelSandboxBackend(client=self.client, ref=ref, working_dir=self.working_dir, env=self.env)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Delete a sandbox by name without resuming it, with the snapshots no other sandbox uses.

        A sandbox that no longer exists is already gone, so destroying it returns quietly.
        """
        if ref.provider != 'vercel':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'vercel'")
        await destroy_sandbox(self.client, ref.id)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        if ref is not None and ref.provider != 'vercel':
            return None
        return _SuppliedBackend(
            run_id=ctx.run_id,
            client=self.client,
            ref=ref,
            image=self.image,
            allow_internet_access=self.allow_internet_access,
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
