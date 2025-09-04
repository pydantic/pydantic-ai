from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .._run_context import AgentDepsT, RunContext
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass
class PrefixedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prefixes the names of the tools it contains.

    See [toolset docs](../toolsets.md#prefixing-tool-names) for more information.
    """

    prefix: str

    @property
    def tool_name_conflict_hint(self) -> str:
        return 'Change the `prefix` attribute to avoid name conflicts.'

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {
            new_name: replace(
                tool,
                toolset=self,
                tool_def=replace(tool.tool_def, name=new_name),
            )
            for name, tool in (await super().get_tools(ctx)).items()
            if (new_name := f'{self.prefix}_{name}')
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        original_name = name.removeprefix(self.prefix + '_')
        ctx = replace(ctx, tool_name=original_name)
        tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name))
        return await super().call_tool(original_name, tool_args, ctx, tool)
