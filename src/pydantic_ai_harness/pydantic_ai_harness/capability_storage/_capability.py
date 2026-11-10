"""`CapabilityStorage` -- keep the files harness capabilities store for themselves outside the run's workspace."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import Any

from pydantic_ai.capabilities import AbstractCapability, WrapperCapability
from pydantic_ai.capabilities.abstract import leaf_capabilities
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import Workspace, WorkspaceBackend
from pydantic_ai_harness._workspace import backend_workspace


@dataclass
class CapabilityStorage(AbstractCapability[AgentDepsT]):
    """Keep the files harness capabilities store for themselves in a workspace of their own.

    Without it, a capability such as `ToolOutputLimits` keeps its files in the run's workspace,
    under `.pydantic-ai-harness/` in the working directory. With it, that directory is created in
    `workspace` instead: a location the model's file tools cannot reach, one that stays
    writable when the run's workspace is read-only, or one per tenant when the capability is passed
    per run with `agent.run(..., capabilities=[CapabilityStorage(...)])`.

    It contributes no tools, instructions, or hooks. Capabilities that keep their own files look it
    up among the run's capabilities. A store given to a capability explicitly, such as
    `WorkspaceStore(workspace=...)`, takes precedence.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.workspaces import LocalWorkspaceBackend
        from pydantic_ai_harness import CapabilityStorage, ToolOutputLimits

        agent = Agent(
            'anthropic:claude-fable-5',
            capabilities=[
                CapabilityStorage(LocalWorkspaceBackend('/var/lib/myapp/agent-files')),
                ToolOutputLimits(),
            ],
        )
        ```
    """

    workspace: WorkspaceBackend
    """The backend that holds the files, such as `LocalWorkspaceBackend('/var/lib/myapp/agent-files')`.

    A backend, not a workspace capability like `LocalWorkspace`. Its I/O runs in-process: a durable
    engine does not route it through its workflow machinery.
    """

    _: KW_ONLY
    # A fixed id, so a `CapabilityStorage` passed to a run replaces the agent's.
    id: str | None = 'capability_storage'

    _storage: Workspace = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._storage = backend_workspace(self.workspace, 'CapabilityStorage')


def storage_workspace(ctx: RunContext[Any]) -> Workspace | None:
    """The workspace of the run's `CapabilityStorage`, or `None` when the run has none."""
    # `root_capability` rather than `capabilities`, which a durable unit's context does not carry.
    if ctx.root_capability is None:
        return None
    storage: Workspace | None = None
    for capability in leaf_capabilities(ctx.root_capability):
        # One returned from a `CapabilityFunc` or given `prefix_tools()` is registered as its wrapper.
        while isinstance(capability, WrapperCapability):
            capability = capability.wrapped
        if isinstance(capability, CapabilityStorage):
            storage = capability._storage  # pyright: ignore[reportPrivateUsage]
    return storage
