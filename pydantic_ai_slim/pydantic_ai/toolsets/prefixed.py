from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolSelector, matches_tool_selector
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass(init=False)
class PrefixedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prefixes the names of the tools it contains.

    See [toolset docs](../toolsets.md#prefixing-tool-names) for more information.
    """

    prefix: str
    tools: ToolSelector[AgentDepsT]

    def __init__(
        self,
        wrapped: Any,  # AbstractToolset[AgentDepsT] — Any to avoid circular import
        prefix: str,
        *,
        tools: ToolSelector[AgentDepsT] = 'all',
    ) -> None:
        self.wrapped = wrapped
        self.prefix = prefix
        self.tools = tools

    @property
    def tool_name_conflict_hint(self) -> str:
        return 'Change the `prefix` attribute to avoid name conflicts.'

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        result: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in (await super().get_tools(ctx)).items():
            if await matches_tool_selector(self.tools, ctx, tool.tool_def):
                new_name = f'{self.prefix}_{name}'
                result[new_name] = replace(
                    tool,
                    toolset=self,
                    tool_def=replace(tool.tool_def, name=new_name),
                )
            else:
                result[name] = tool
        return result

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        prefix_with_sep = self.prefix + '_'
        if name.startswith(prefix_with_sep):
            original_name = name.removeprefix(prefix_with_sep)
            ctx = replace(ctx, tool_name=original_name)
            tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name))
            return await super().call_tool(original_name, tool_args, ctx, tool)
        return await super().call_tool(name, tool_args, ctx, tool)
