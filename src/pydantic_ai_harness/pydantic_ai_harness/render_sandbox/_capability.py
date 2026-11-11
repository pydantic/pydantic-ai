"""Capability that supplies a Render sandbox to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from render import RenderAsync

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import Authority
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace import with_authority
from pydantic_ai_harness._workspace_provider import check_provider, check_working_dir, reject_defer_loading

from ._backend import (
    DEFAULT_SANDBOX_TIMEOUT,
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

    When omitted, credentials come from `RENDER_API_KEY` and `RENDER_WORKSPACE_ID`.
    """

    allow_internet_access: bool = True
    """Whether a newly created sandbox may reach the internet; without it, commands such as `pip install` fail."""

    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT
    """Seconds a newly created sandbox lives before Render terminates it, up to 86400."""

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against.

    Created on a new sandbox; an attached sandbox must already have it. `None` uses the sandbox's home directory.
    """

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, also on an attached sandbox; nothing is read from the host."""

    authority: Authority = 'operator'
    """The [authority][pydantic_ai.messages.Authority] of the sandbox's files: `'none'` for files nobody on the
    application's side wrote, such as a cloned third-party repository, and `'user'` for the user's own files.
    Anything below `'operator'` wraps the workspace in an [`AuthorityWorkspace`][pydantic_ai.workspaces.AuthorityWorkspace],
    so capabilities that send its files to the model send them with no more authority than that."""

    path_authority: Mapping[str, Authority] | None = None
    """Levels for some directories and files, overriding `authority` for the paths under them; the longest matching
    path wins. Paths are sandbox paths: relative ones resolve against the working directory, and `~` isn't expanded.
    See [`Workspace.authority_of`][pydantic_ai.workspaces.Workspace.authority_of]."""

    def __post_init__(self) -> None:
        reject_defer_loading('RenderSandbox', self.defer_loading)
        check_sandbox_timeout(self.sandbox_timeout)
        check_working_dir(self.working_dir)

    @classmethod
    def from_spec(
        cls,
        *,
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
        allow_internet_access: bool = True,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> RenderSandbox[Any]:
        """Build from an agent spec, naming the fields a spec can express.

        Core reads this signature to generate the spec's JSON schema. Inherited, core reads `__init__`, and
        `client`'s `RenderAsync` hint has no JSON schema, so only the bare `'RenderSandbox'` is published.
        """
        return cls(
            id=id,
            description=description,
            defer_loading=defer_loading,
            allow_internet_access=allow_internet_access,
            sandbox_timeout=sandbox_timeout,
            working_dir=working_dir,
            env=env,
        )

    def backend(self, ref: WorkspaceRef) -> RenderSandboxBackend:
        """Attach lazily to an existing Render sandbox."""
        return RenderSandboxBackend(client=self.client, ref=ref, working_dir=self.working_dir, env=self.env)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Terminate the sandbox identified by `ref`; an already terminated or unknown one returns quietly.

        Rejected credentials raise `WorkspaceUnavailableError`.
        """
        check_provider(ref, 'render')
        await terminate_sandbox(self.client, ref.id)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """The backend for this run, in an `AuthorityWorkspace` when `authority` or `path_authority` asks for one."""
        return with_authority(self._run_backend(ctx, ref=ref), self.authority, self.path_authority)

    def _run_backend(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Supply a new sandbox workspace or attach to a recognized Render reference."""
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
