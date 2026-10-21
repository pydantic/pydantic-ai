"""Toolset exposing the asset-inventory tool."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from pydantic_ai.messages import ToolKind
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness._warn import SET_WORKING_DIR_ON_THE_WORKSPACE, warn_argument_ignored
from pydantic_ai_harness.repo_context._inventory import AgentContextInventory, scan_assets


class RepoContextToolset(FunctionToolset[AgentDepsT]):
    """Exposes a single tool that reports where the repo's CE assets live."""

    def __init__(
        self,
        asset_roots: Sequence[str],
        tool_name: str,
        *,
        workspace_dir: Path | None = None,
        id: str | None = None,
        unless_tool_kind: ToolKind | None = None,
    ) -> None:
        """Create the toolset.

        Args:
            asset_roots: Root directories to scan, relative to the working directory.
            tool_name: Name of the inventory tool.
            workspace_dir: Deprecated and ignored.
            id: The toolset's id.
            unless_tool_kind: A kind the inventory tool steps aside for, such as `FileSystem`'s `READ_FILE_KIND`.
        """
        super().__init__(id=id)
        if workspace_dir is not None:
            warn_argument_ignored('RepoContextToolset', 'workspace_dir', SET_WORKING_DIR_ON_THE_WORKSPACE, stacklevel=3)
        self._asset_roots = asset_roots
        self._unless_tool_kind = unless_tool_kind
        self.add_function(
            self.inventory_agent_context,
            name=tool_name,
            prepare=self._step_aside if unless_tool_kind is not None else None,
        )

    async def _step_aside(self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition:
        return replace(tool_def, unless_tool_kind=self._unless_tool_kind)

    async def inventory_agent_context(self, ctx: RunContext[AgentDepsT]) -> AgentContextInventory:
        """Report where this repo's coding-assistant setup lives.

        Returns the locations of instruction dirs (`.claude`, `.agents`,
        `.codex`, `.grok`) and, within each, the `skills/`, `agents/`, and
        `settings.json` (hooks) it contains. This locates assets for inspection;
        it does not parse their contents.
        """
        return await scan_assets(ctx.workspace, Path(await ctx.workspace.working_dir()), self._asset_roots)
