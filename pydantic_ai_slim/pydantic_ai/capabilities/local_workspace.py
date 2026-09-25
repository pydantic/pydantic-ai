from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from pathlib import Path

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.workspaces import LocalWorkspaceBackend, ReadOnlyWorkspace, Workspace, WorkspaceBackend, WorkspaceRef

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

    env: Mapping[str, str] | None = None
    """Environment variables for every command, on top of `PATH` and `HOME`; don't pass `os.environ` (secrets)."""

    id: str | None = 'local_workspace'
    """Fixed, so a later `LocalWorkspace` replaces an earlier one whole; pass distinct ids to keep both."""

    def __post_init__(self) -> None:
        # Pin a relative `working_dir` to today's directory, and surface an unusable platform where the
        # capability is written, not on the first run.
        self.working_dir = LocalWorkspaceBackend(self.working_dir).ref.id

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        # The later configuration replaces the earlier one whole, so
        # an `env` (and its secrets) or `read_only` stated only on the replaced one never carries over.
        return capabilities[-1]

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        backend = LocalWorkspaceBackend(self.working_dir, env=self.env)
        if ref is not None and ref != backend.ref:
            # Another provider's environment, or a local directory other than the configured one:
            # a ref from message history must never redirect the agent to an arbitrary host directory.
            return None
        return ReadOnlyWorkspace(Workspace(backend)) if self.read_only else backend
