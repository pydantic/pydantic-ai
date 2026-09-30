"""Discovery protocol for capabilities that expose general workspace file tools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend

FILE_READ_OVERHEAD_CHARS = 512
"""Conservative room for a provider's model-facing read header and continuation hint."""


@dataclass(frozen=True)
class FileToolsInfo:
    """The model-facing tools and read limit exposed by a file-tools provider."""

    read_tool: str
    """Tool that reads one file."""

    path_arg: str = 'path'
    """Argument that receives the workspace path."""

    list_tools: frozenset[str] = frozenset()
    """Tools that can discover files below a directory."""

    max_read_chars: int | None = None
    """Maximum characters returned by one read, or `None` when unlimited."""


@runtime_checkable
class ProvidesFileTools(Protocol):
    """A capability that gives the model tools for reading workspace files.

    Consumers use `can_read` before replacing a capability-specific reader with the general
    file tools. The check must apply the same path boundary and access rules as those tools.
    """

    async def can_read(self, path: str, *, workspace: WorkspaceBackend) -> bool:
        """Whether this provider's model-facing tools can read `path` in `workspace`."""
        ...  # pragma: no cover

    async def can_read_tree(self, path: str, *, workspace: WorkspaceBackend) -> bool:
        """Whether the provider can read every file below `path`."""
        ...  # pragma: no cover

    def file_tools(self) -> FileToolsInfo:
        """Describe the provider's model-facing file tools."""
        ...  # pragma: no cover


async def file_tools_provider(
    ctx: RunContext[AgentDepsT],
    paths: str | Sequence[str],
    *,
    tool_names: set[str] | None = None,
    require_listing: bool = False,
    require_tree: bool = False,
    min_read_chars: int | None = None,
) -> tuple[ProvidesFileTools, FileToolsInfo] | None:
    """Return the first active provider whose offered tools can read every requested path."""
    requested = (paths,) if isinstance(paths, str) else paths
    for capability_id, capability in ctx.capabilities.items():
        if capability_id not in ctx.active_capability_ids or not isinstance(capability, ProvidesFileTools):
            continue
        info = capability.file_tools()
        if not info.read_tool or (tool_names is not None and info.read_tool not in tool_names):
            continue
        if require_listing and (
            not info.list_tools or (tool_names is not None and info.list_tools.isdisjoint(tool_names))
        ):
            continue
        if min_read_chars is not None and info.max_read_chars is not None and info.max_read_chars < min_read_chars:
            continue
        for path in requested:
            can_read = capability.can_read_tree if require_tree else capability.can_read
            if not await can_read(path, workspace=ctx.workspace):
                break
        else:
            return capability, info
    return None
