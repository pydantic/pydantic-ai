"""Capability that supplies an E2B sandbox to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

from pydantic_ai_harness._workspace_provider import check_integer, check_working_dir
from pydantic_ai_harness.e2b_sandbox import _backend
from pydantic_ai_harness.e2b_sandbox._backend import DEFAULT_SANDBOX_TIMEOUT, E2BSandboxBackend


@dataclass(kw_only=True)
class E2BSandbox(AbstractCapability[AgentDepsT]):
    """Supply an isolated [E2B](https://e2b.dev) sandbox as the run's workspace.

    A run with no reference creates a fresh sandbox. Pass a `WorkspaceRef` supplied by the
    application to attach to an environment managed elsewhere.

    This capability supplies execution only. Compose it with tools or
    capabilities that use the workspace, such as `Coder`, `Shell`, or `FileSystem`.
    Shell commands run under `sh -c` in the sandbox's login shell environment.
    """

    template: str | None = None
    """E2B template name or ID for a newly created sandbox; E2B's default when `None`.

    An unknown template raises `WorkspaceUnavailableError` on first use.
    """

    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT
    """Total lifetime of the sandbox in seconds, applied on create and on attach.

    When it runs out, E2B pauses the sandbox and attaching resumes it. The default, 1 hour, is
    the most E2B's Hobby plan allows; Pro plans allow up to 86400.
    """

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; `None` uses the sandbox's own."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, also on an attached workspace; nothing is read from the host."""

    allow_internet_access: bool = True
    """Whether a newly created workspace may reach the internet."""

    def __post_init__(self) -> None:
        if self.defer_loading is True:
            raise UserError(
                '`defer_loading` is not supported on `E2BSandbox`: a run never takes its workspace '
                'from a deferred capability, so the sandbox would never be used.'
            )
        check_integer('sandbox_timeout', self.sandbox_timeout)
        check_working_dir(self.working_dir)

    def backend(self, ref: WorkspaceRef) -> E2BSandboxBackend:
        """Attach lazily to an existing E2B sandbox."""
        if ref.provider != 'e2b':
            raise ValueError(f'Expected an E2B workspace ref, got {ref.provider!r}')
        return E2BSandboxBackend(
            ref=ref, sandbox_timeout=self.sandbox_timeout, working_dir=self.working_dir, env=self.env
        )

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Kill a sandbox by ID, including paused sandboxes, without attaching."""
        if ref.provider != 'e2b':
            raise ValueError(f'Expected an E2B workspace ref, got {ref.provider!r}')
        # The SDK's ID-only DELETE works for paused sandboxes; connect would resume and bill them.
        await _backend.e2b.AsyncSandbox.kill(ref.id)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        del ctx
        if ref is not None and ref.provider != 'e2b':
            return None
        return E2BSandboxBackend(
            ref=ref,
            template=self.template,
            sandbox_timeout=self.sandbox_timeout,
            working_dir=self.working_dir,
            env=self.env,
            allow_internet_access=self.allow_internet_access,
        )
