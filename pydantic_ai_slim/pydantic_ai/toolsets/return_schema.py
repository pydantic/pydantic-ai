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

    include_return_schema: bool = True
    """Whether to include the return schema in the tool description."""

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        original_tools = await super().get_tools(ctx)

        def _build_description(tool: ToolsetTool[AgentDepsT]) -> str:
            base_desc = tool.tool_def.description or ''
            if tool.tool_def.return_schema is not None:
                # TODO: This should be overrideable by PromptConfig when that lands
                return '\n\n'.join([base_desc, 'Return schema:', str(tool.tool_def.return_schema)])
            return base_desc

        return {
            name: replace(
                original_tools[name],
                tool_def=replace(
                    tool.tool_def,
                    description=_build_description(tool),
                ),
            )
            for name, tool in original_tools.items()
            if self.include_return_schema or tool.tool_def.return_schema  # Either the toolset flags it for us or the tool itself does
        }
