"""Capability that supplies a Runta sandbox to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import Authority
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace import with_authority
from pydantic_ai_harness._workspace_provider import (
    check_integer,
    check_path_authority,
    check_provider,
    check_working_dir,
    reject_defer_loading,
)
from pydantic_ai_harness.runta_sandbox._backend import RuntaSandboxBackend, destroy_runtime


@dataclass(kw_only=True)
class RuntaSandbox(AbstractCapability[AgentDepsT]):
    """Run the agent's workspace in a persistent [Runta](https://runta.com/docs/) runtime, a cloud microVM.

    A run with no reference creates a fresh runtime on first use; pass a `WorkspaceRef` to attach
    to an existing one. Ending a run does not delete the runtime: it keeps running, and billing, until
    you delete it. Commands run under `sh -c` in the runtime's own shell environment.

    This capability supplies the workspace only. Compose it with capabilities that use it, such
    as `Coder`, `Shell`, and `FileSystem`. See
    [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.
    """

    token: str | None = field(default=None, repr=False)
    """A Runta API key. When omitted, each operation reads `RUNTA_TOKEN`."""

    endpoint: str | None = None
    """The Runta API endpoint. When omitted, `RUNTA_ENDPOINT`, then `https://api.runta.com`."""

    image: str | None = None
    """Image for a newly created runtime; `None` uses Runta's default. An unknown image fails on first use."""

    vcpus: int | None = None
    """vCPUs for a newly created runtime; `None` uses Runta's default."""

    memory_mib: int | None = None
    """Memory in MiB for a newly created runtime; `None` uses Runta's default."""

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; `None` uses the runtime's default.
    Created on a new runtime; an attached runtime must already have it."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, on top of the runtime's own; a command's `env` is layered on top."""

    authority: Authority = 'operator'
    """The [authority][pydantic_ai.messages.Authority] of the sandbox's files: `'none'` for files nobody on the
    application's side wrote, such as a cloned third-party repository, and `'user'` for the user's own files.
    Anything below `'operator'` wraps the workspace in an [`AuthorityWorkspace`][pydantic_ai.workspaces.AuthorityWorkspace]."""

    path_authority: Mapping[str, Authority] | None = None
    """Levels for some directories and files, overriding `authority` for the paths under them; the longest matching
    path wins. Paths are sandbox paths: relative ones resolve against the working directory, and `~` isn't expanded.
    See [`Workspace.authority_of`][pydantic_ai.workspaces.Workspace.authority_of]."""

    def __post_init__(self) -> None:
        reject_defer_loading('RuntaSandbox', self.defer_loading)
        # Checked here rather than at the first workspace operation, so a bad value fails where it is written.
        check_integer('vcpus', self.vcpus, optional=True)
        check_integer('memory_mib', self.memory_mib, optional=True)
        check_working_dir(self.working_dir)
        check_path_authority(self.path_authority)

    @classmethod
    def from_spec(
        cls,
        *,
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
        endpoint: str | None = None,
        image: str | None = None,
        vcpus: int | None = None,
        memory_mib: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        authority: Authority = 'operator',
        path_authority: Mapping[str, Authority] | None = None,
    ) -> RuntaSandbox[Any]:
        """Build from an agent spec, naming the fields a spec can express.

        `token` is left out: a spec is configuration, and the API key comes from `RUNTA_TOKEN`.
        """
        return cls(
            id=id,
            description=description,
            defer_loading=defer_loading,
            endpoint=endpoint,
            image=image,
            vcpus=vcpus,
            memory_mib=memory_mib,
            working_dir=working_dir,
            env=env,
            authority=authority,
            path_authority=path_authority,
        )

    def backend(self, ref: WorkspaceRef | None = None) -> RuntaSandboxBackend:
        """Construct a lazy backend: for the runtime `ref` names, or for a new one."""
        return RuntaSandboxBackend(
            ref=ref,
            token=self.token,
            endpoint=self.endpoint,
            image=self.image,
            vcpus=self.vcpus,
            memory_mib=self.memory_mib,
            working_dir=self.working_dir,
            env=self.env,
        )

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Delete a runtime by its UUID without attaching or waking it.

        A runtime that no longer exists is already gone, so destroying it returns quietly.
        """
        check_provider(ref, 'runta')
        await destroy_runtime(ref.id, token=self.token, endpoint=self.endpoint)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """The backend for this run, in an `AuthorityWorkspace` when `authority` or `path_authority` asks for one."""
        if ref is not None and ref.provider != 'runta':
            return None
        # No I/O here: the backend attaches or creates on first use.
        return with_authority(self.backend(ref), self.authority, self.path_authority)
