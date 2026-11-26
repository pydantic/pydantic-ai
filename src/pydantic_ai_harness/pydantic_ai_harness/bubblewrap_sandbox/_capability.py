from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass

from pydantic_ai.capabilities import WrapperCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import Workspace, WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness.bubblewrap_sandbox._workspace import BubblewrapWorkspace


@dataclass
class BubblewrappedWorkspace(WrapperCapability[AgentDepsT]):
    """Runs the commands of the wrapped capability's workspace in a bubblewrap wrapper, on that workspace's host.

    !!! warning "Best effort, not a security boundary"
        This wrapper is a best-effort convenience that keeps a trusted agent from accidentally changing files outside
        its working directory. It is **not a security boundary** and **not a sandbox for untrusted code or untrusted
        agents**: commands share the host kernel and run as the host user. For untrusted agents, use a VM or
        microVM-backed sandbox such as [`E2BSandbox`][pydantic_ai_harness.e2b_sandbox.E2BSandbox],
        [`ModalSandbox`][pydantic_ai_harness.modal_sandbox.ModalSandbox],
        [`DaytonaSandbox`][pydantic_ai_harness.daytona_sandbox.DaytonaSandbox],
        [`BoxLiteSandbox`][pydantic_ai_harness.boxlite_sandbox.BoxLiteSandbox], or
        [`SmolSandbox`][pydantic_ai_harness.smol_sandbox.SmolSandbox].

        Do **not** set the working directory to your home directory (or a directory containing it): files there such as
        shell startup files, or configuration under a non-default `$XDG_CONFIG_HOME`, can be changed by the agent and
        run later outside the wrapper (only the default startup locations are mounted read-only).

    See [`BubblewrapWorkspace`][pydantic_ai_harness.bubblewrap_sandbox.BubblewrapWorkspace] for what the wrapper allows.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import BubblewrappedWorkspace, SSHWorkspace

    agent = Agent(
        'anthropic:claude-opus-5-5',
        capabilities=[BubblewrappedWorkspace(SSHWorkspace('dev@build-box', working_dir='/srv/app'))],
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
