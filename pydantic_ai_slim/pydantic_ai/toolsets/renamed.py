from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .._run_context import AgentDepsT, RunContext
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass
class RenamedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that renames the tools it contains using a dictionary mapping new names to original names.

    See [toolset docs](../toolsets.md#renaming-tools) for more information.
    """

    name_map: dict[str, str]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        original_to_new_name_map = {v: k for k, v in self.name_map.items()}
        original_tools = await super().get_tools(ctx)
        tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        for original_name, tool in original_tools.items():
            new_name = original_to_new_name_map.get(original_name, None)
            if new_name:
                tools[new_name] = replace(
                    tool,
                    toolset=self,
                    tool_def=replace(tool.tool_def, name=new_name),
                )
            else:
                tools[original_name] = tool
        return tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        original_name = self.name_map.get(name, name)
        ctx = replace(ctx, tool_name=original_name)
        tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name))
        return await super().call_tool(original_name, tool_args, ctx, tool)
