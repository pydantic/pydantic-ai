from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass

from pydantic_ai.capabilities import WrapperCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import Workspace, WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness.bubblewrap_sandbox._workspace import BubblewrapWorkspace


@dataclass
class BubblewrapSandbox(WrapperCapability[AgentDepsT]):
    """Runs the commands of the wrapped capability's workspace in a bubblewrap sandbox, on that workspace's host.

    See [`BubblewrapWorkspace`][pydantic_ai_harness.bubblewrap_sandbox.BubblewrapWorkspace] for what the sandbox allows.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import BubblewrapSandbox, SSHWorkspace

    agent = Agent(
        'anthropic:claude-opus-5-5',
        capabilities=[BubblewrapSandbox(SSHWorkspace('dev@build-box', working_dir='/srv/app'))],
    )
    ```
    """

    _: KW_ONLY

    network: bool = False
    """Whether commands share the host's network; without it, a seccomp filter also blocks every socket connection."""

    bwrap_args: Sequence[str] = ()
    """Extra `bwrap` arguments, placed after the defaults so they can override them."""

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        backend = super().get_workspace(ctx, ref=ref)
        if backend is None:
            return None
        workspace = backend if isinstance(backend, Workspace) else Workspace(backend)
        return BubblewrapWorkspace(workspace, network=self.network, bwrap_args=self.bwrap_args)
