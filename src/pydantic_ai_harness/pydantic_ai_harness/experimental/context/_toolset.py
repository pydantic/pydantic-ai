"""Toolset exposing the asset-inventory tool."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.experimental.context._inventory import AgentContextInventory, scan_assets


class RepoContextToolset(FunctionToolset[AgentDepsT]):
    """Exposes a single tool that reports where the repo's CE assets live."""

    def __init__(self, workspace_dir: Path, asset_roots: Sequence[str], tool_name: str) -> None:
        super().__init__()
        self._workspace_dir = workspace_dir
        self._asset_roots = asset_roots
        self.add_function(self.inventory_agent_context, name=tool_name)

    async def inventory_agent_context(self) -> AgentContextInventory:
        """Report where this repo's coding-assistant setup lives.

        Returns the locations of instruction dirs (`.claude`, `.agents`,
        `.codex`, `.grok`) and, within each, the `skills/`, `agents/`, and
        `settings.json` (hooks) it contains. This locates assets so you can read
        and translate them; it does not parse their contents.
        """
        return scan_assets(self._workspace_dir, self._asset_roots)
