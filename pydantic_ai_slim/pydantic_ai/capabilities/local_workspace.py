from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from pathlib import Path

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.messages import Authority
from pydantic_ai.workspaces import (
    AuthorityWorkspace,
    LocalWorkspaceBackend,
    ReadOnlyWorkspace,
    Workspace,
    WorkspaceBackend,
    WorkspaceRef,
)
from pydantic_ai.workspaces.workspace import merge_path_authority

from .abstract import AbstractCapability


@dataclass
class LocalWorkspace(AbstractCapability[AgentDepsT]):
    """Gives runs a [workspace](../workspace.md) on this machine: host subprocesses and the host filesystem.

    This isolates nothing: tools reach anywhere on the host this process can. Use it for trusted local
    work, and a container- or VM-based workspace for untrusted code.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import LocalWorkspace

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[LocalWorkspace('~/project')])
    ```

    It declines a ref for any other directory, so a ref in message history can't point it elsewhere on the host.
    """

    working_dir: str | Path
    """Where commands start and relative paths resolve; `~` is expanded and `'.'` is today's directory."""

    _: KW_ONLY

    read_only: bool = False
    """Whether to wrap the workspace in a [`ReadOnlyWorkspace`][pydantic_ai.workspaces.ReadOnlyWorkspace]."""

    authority: Authority = 'operator'
    """The [authority][pydantic_ai.messages.Authority] of the directory's files: `'user'` for the user's own files,
    `'none'` for files nobody on the application's side wrote. Anything below `'operator'` wraps the workspace in an
    [`AuthorityWorkspace`][pydantic_ai.workspaces.AuthorityWorkspace]."""

    path_authority: Mapping[str, Authority] | None = None
    """Levels for some directories and files, overriding `authority` for the paths under them; the longest matching
    path wins. Relative paths resolve against `working_dir`, and `~` is expanded. A path outside `working_dir`, such
    as `'~/.config/my-app/AGENTS.md'`, is part of the workspace only when it's listed here; any other is at `'none'`.
    See [`Workspace.authority_of`][pydantic_ai.workspaces.Workspace.authority_of]."""

    env: Mapping[str, str] | None = None
    """Environment variables for every command, on top of `PATH`, `HOME`, `LANG`, `LC_ALL` and `LC_CTYPE`.

    Nothing else from this process's environment reaches commands; don't pass `os.environ` (secrets).
    """

    id: str | None = 'local_workspace'
    """Fixed, so a later `LocalWorkspace` replaces an earlier one whole; pass distinct ids to keep both."""

    def __post_init__(self) -> None:
        # Pin a relative `working_dir` to today's directory, and surface an unusable platform where the
        # capability is written, not on the first run.
        self.working_dir = LocalWorkspaceBackend(self.working_dir).ref.id
        if self.path_authority:
            # Two spellings of one path, like `~/x` and `/home/me/x`, keep the lower level.
            self.path_authority = merge_path_authority(
                (os.path.expanduser(path), level) for path, level in self.path_authority.items()
            )

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        # The later configuration replaces the earlier one whole, so an `env` (and its secrets), `read_only`,
        # `authority` or `path_authority` stated only on the replaced one never carries over.
        return capabilities[-1]

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        backend = LocalWorkspaceBackend(self.working_dir, env=self.env)
        if ref is not None and ref != backend.ref:
            # A ref from history never redirects host access away from the configured backend.
            return None
        workspace: WorkspaceBackend = backend
        if self.read_only:
            workspace = ReadOnlyWorkspace(Workspace(workspace))
        if self.authority != 'operator' or self.path_authority:
            workspace = AuthorityWorkspace(
                Workspace(workspace), authority=self.authority, path_authority=self.path_authority
            )
        return workspace
