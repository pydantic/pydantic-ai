"""Capability that supplies a Render sandbox to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from render import RenderAsync

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace_provider import check_working_dir

from ._backend import (
    DEFAULT_SANDBOX_TIMEOUT,
    DEFAULT_WORKING_DIR,
    RenderSandboxBackend,
    check_sandbox_timeout,
    terminate_sandbox,
)


@dataclass(kw_only=True)
class RenderSandbox(AbstractCapability[AgentDepsT]):
    """Supply an isolated [Render sandbox](https://render.com/docs/sandboxes) as the run's workspace.

    A run with no reference creates a fresh sandbox on first use. Pass a `WorkspaceRef` supplied by the
    application to attach to an environment managed elsewhere. Ending a run does not terminate the
    sandbox: store `result.workspace.ref` and call `destroy(ref)`, or let `sandbox_timeout` end it.

    This capability supplies execution only. Compose it with tools or capabilities that use the
    workspace, such as `Coder`, `Shell`, or `FileSystem`.
    """

    client: RenderAsync | None = None
    """A caller-owned `render.RenderAsync`, never closed for you.

    When omitted, each workspace operation opens a client from `RENDER_API_KEY` and `RENDER_WORKSPACE_ID`
    and closes it when the operation ends.
    """

    allow_internet_access: bool = True
    """Whether a newly created sandbox may reach the internet; without it, commands such as `pip install` fail."""

    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT
    """Seconds a newly created sandbox lives before Render terminates it, up to 86400."""

    working_dir: str = DEFAULT_WORKING_DIR
    """Absolute directory commands start in and relative paths resolve against.

    Created on a new sandbox; an attached sandbox must already have it.
    """

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, also on an attached sandbox; nothing is read from the host."""

    def __post_init__(self) -> None:
        if self.defer_loading:
            raise UserError(
                '`RenderSandbox` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.'
            )
        check_sandbox_timeout(self.sandbox_timeout)
        check_working_dir(self.working_dir)

    def backend(self, ref: WorkspaceRef) -> RenderSandboxBackend:
        """Attach lazily to an existing Render sandbox."""
        return RenderSandboxBackend(client=self.client, ref=ref, working_dir=self.working_dir, env=self.env)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Terminate a sandbox by ID without attaching; one already terminated or unknown returns quietly.

        Rejected credentials raise `WorkspaceUnavailableError`.
        """
        if ref.provider != 'render':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'render'")
        await terminate_sandbox(self.client, ref.id)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        del ctx
        if ref is not None and ref.provider != 'render':
            return None
        return RenderSandboxBackend(
            client=self.client,
            ref=ref,
            allow_internet_access=self.allow_internet_access,
            sandbox_timeout=self.sandbox_timeout,
            working_dir=self.working_dir,
            env=self.env,
        )
