from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .._run_context import AgentDepsT, RunContext
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass
class PrefixedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prefixes the names of the tools it contains."""

    prefix: str

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {
            new_name: replace(
                tool,
                toolset=self,
                tool_def=replace(tool.tool_def, name=new_name),
            )
            for name, tool in (await super().get_tools(ctx)).items()
            if (new_name := self._prefixed_tool_name(name))
        }

    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any]) -> Any:
        original_name = self._unprefixed_tool_name(name)
        ctx = replace(ctx, tool_name=original_name)
        return await super().call_tool(ctx, original_name, tool_args)

    def _prefixed_tool_name(self, tool_name: str) -> str:
        return f'{self.prefix}_{tool_name}'

    def _unprefixed_tool_name(self, tool_name: str) -> str:
        full_prefix = f'{self.prefix}_'
        if not tool_name.startswith(full_prefix):
            raise ValueError(f"Tool name '{tool_name}' does not start with prefix '{full_prefix}'")
        return tool_name[len(full_prefix) :]
