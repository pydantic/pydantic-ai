"""Toolset exposing the asset-inventory tool."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.repo_context._inventory import AgentContextInventory, scan_assets


class RepoContextToolset(FunctionToolset[AgentDepsT]):
    """Exposes a single tool that reports where the repo's coding-assistant assets live."""

    def __init__(self, asset_roots: Sequence[str], tool_name: str, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._asset_roots = asset_roots
        self.add_function(self.inventory_agent_context, name=tool_name)

    async def inventory_agent_context(self, ctx: RunContext[AgentDepsT]) -> AgentContextInventory:
        """Report where this repo's coding-assistant setup lives.

        Returns the locations of instruction dirs (`.claude`, `.agents`,
        `.codex`, `.grok`) and, within each, the `skills/`, `agents/`, and
        `settings.json` (hooks) it contains. This locates assets for inspection;
        it does not parse their contents.
        """
        return await scan_assets(ctx.workspace, Path(await ctx.workspace.working_dir()), self._asset_roots)
