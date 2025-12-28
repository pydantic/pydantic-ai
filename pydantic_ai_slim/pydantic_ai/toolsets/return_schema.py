from __future__ import annotations

from dataclasses import dataclass, replace

from .._run_context import AgentDepsT, RunContext
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass
class ReturnSchemaToolset(WrapperToolset[AgentDepsT]):
    """A toolset that adds the return schema to the tool description if it is present.

    See [toolset docs](../toolsets.md#return-schema-toolset) for more information.
    """

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        original_tools = await super().get_tools(ctx)

        return {
            name: replace(
                original_tools[name],
                tool_def=replace(
                    tool.tool_def,
                    description='\n\n'.join(
                        [tool.tool_def.description or '', 'Return schema:', str(tool.tool_def.return_schema)]
                    ),
                ),
            )
            for name, tool in original_tools.items()
            if tool.tool_def.return_schema
        }
