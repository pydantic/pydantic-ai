from __future__ import annotations

from dataclasses import dataclass, replace

from .._run_context import AgentDepsT, RunContext
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass
class LazyToolset(WrapperToolset[AgentDepsT]):
    """A toolset that marks tools as lazy, hiding them from the model until discovered via tool search.

    See [toolset docs](../toolsets.md#lazy-tools) for more information.
    """

    tool_names: list[str] | None = None
    """Optional list of tool names to mark as lazy. If `None`, all tools are marked as lazy."""

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await super().get_tools(ctx)
        result: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in tools.items():
            if self.tool_names is None or name in self.tool_names:
                result[name] = replace(tool, tool_def=replace(tool.tool_def, lazy=True))
            else:
                result[name] = tool
        return result
