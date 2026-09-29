"""Discovery protocol for capabilities that expose general workspace file tools."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend


@runtime_checkable
class ProvidesFileTools(Protocol):
    """A capability that gives the model tools for reading workspace files.

    Consumers use `can_read` before replacing a capability-specific reader with the general
    file tools. The check must apply the same path boundary and access rules as those tools.
    """

    async def can_read(self, path: str, *, workspace: WorkspaceBackend) -> bool:
        """Whether this provider's model-facing tools can read `path` in `workspace`."""
        ...  # pragma: no cover


async def file_tools_provider(ctx: RunContext[AgentDepsT], paths: str | Sequence[str]) -> ProvidesFileTools | None:
    """Return the first active file-tools provider that can read every requested path."""
    requested = (paths,) if isinstance(paths, str) else paths
    for capability_id, capability in ctx.capabilities.items():
        if capability_id not in ctx.active_capability_ids or not isinstance(capability, ProvidesFileTools):
            continue
        for path in requested:
            if not await capability.can_read(path, workspace=ctx.workspace):
                break
        else:
            return capability
    return None
