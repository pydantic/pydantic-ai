"""Capability that supplies an E2B sandbox to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import Authority
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace import with_authority
from pydantic_ai_harness._workspace_provider import (
    check_env_types,
    check_integer,
    check_path_authority,
    check_provider,
    check_working_dir,
    reject_defer_loading,
)
from pydantic_ai_harness.e2b_sandbox import _backend
from pydantic_ai_harness.e2b_sandbox._backend import DEFAULT_SANDBOX_TIMEOUT, E2BSandboxBackend

if TYPE_CHECKING:
    import e2b


@dataclass(kw_only=True)
class E2BSandbox(AbstractCapability[AgentDepsT]):
    """Supply an isolated [E2B](https://e2b.dev) sandbox as the run's workspace.

    A run with no reference creates a fresh sandbox. Pass a `WorkspaceRef` supplied by the
    application to attach to an environment managed elsewhere.

    This capability supplies execution only. Compose it with tools or
    capabilities that use the workspace, such as `Coder`, `Shell`, or `FileSystem`.
    Shell commands run under `sh -c` in the sandbox's login shell environment.
    """

    client: e2b.E2B | None = None
    """An `e2b.E2B` to create, attach and destroy with, for its API key and other connection settings;
    `E2B_API_KEY` and the rest of the environment when `None`. A spec cannot set it.

    To use different credentials for each user, supply the workspace from a capability of your own, which
    can build the client from the run's `deps` and pass it to `E2BSandboxBackend`.
    """

    template: str | None = None
    """E2B template name or ID for a newly created sandbox; E2B's default when `None`.

    An unknown template raises `WorkspaceUnavailableError` on first use.
    """

    allow_internet_access: bool = True
    """Whether a newly created workspace may reach the internet."""

    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT
    """Total lifetime of the sandbox in seconds, applied on create and on attach.

    When it runs out, E2B pauses the sandbox and attaching resumes it. The default, 1 hour, is
    the most E2B's Hobby plan allows; Pro plans allow up to 86400.
    """

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; `None` uses the sandbox's own."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, also on an attached workspace; nothing is read from the host.

    The agent's commands can read them, and a new sandbox keeps them for every process it runs.
    """

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
        reject_defer_loading('E2BSandbox', self.defer_loading)
        check_integer('sandbox_timeout', self.sandbox_timeout)
        check_working_dir(self.working_dir)
        check_path_authority(self.path_authority)
        try:
            check_env_types(self.env)
        except TypeError:
            raise UserError('`env` keys and values must be strings.') from None

    @classmethod
    def from_spec(
        cls,
        *,
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
        template: str | None = None,
        allow_internet_access: bool = True,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        authority: Authority = 'operator',
        path_authority: Mapping[str, Authority] | None = None,
    ) -> E2BSandbox[Any]:
        """Build from an agent spec, naming the fields a spec can express.

        Core reads this signature to generate the spec's JSON schema. Inherited, core falls back to the
        variadic base `from_spec`, because `client`'s `e2b.E2B` hint is a `TYPE_CHECKING` import, and
        publishes only the bare `'E2BSandbox'`.
        """
        return cls(
            id=id,
            description=description,
            defer_loading=defer_loading,
            template=template,
            allow_internet_access=allow_internet_access,
            sandbox_timeout=sandbox_timeout,
            working_dir=working_dir,
            env=env,
            authority=authority,
            path_authority=path_authority,
        )

    def backend(self, ref: WorkspaceRef) -> E2BSandboxBackend:
        """Attach lazily to an existing E2B sandbox."""
        return E2BSandboxBackend(
            client=self.client,
            ref=ref,
            sandbox_timeout=self.sandbox_timeout,
            working_dir=self.working_dir,
            env=self.env,
        )

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Kill a sandbox by ID, including paused sandboxes, without attaching."""
        check_provider(ref, 'e2b')
        await _backend.kill_sandbox(ref.id, client=self.client)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """The backend for this run, in an `AuthorityWorkspace` when `authority` or `path_authority` asks for one."""
        return with_authority(self._run_backend(ctx, ref=ref), self.authority, self.path_authority)

    def _run_backend(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        del ctx
        if ref is not None and ref.provider != 'e2b':
            return None
        return E2BSandboxBackend(
            client=self.client,
            ref=ref,
            template=self.template,
            allow_internet_access=self.allow_internet_access,
            sandbox_timeout=self.sandbox_timeout,
            working_dir=self.working_dir,
            env=self.env,
        )
