from __future__ import annotations

from dataclasses import dataclass, replace

from .._run_context import AgentDepsT, RunContext
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass
class ReturnSchemaToolset(WrapperToolset[AgentDepsT]):
    """A toolset that adds the return schema to the tool description if it is present.

    See [toolset docs](../toolsets.md#return-schema-toolset) for more information.
    # Non existient, will add once the API is approved of
    """

    include_return_schema: bool = False
    """Whether to include the return schema in the tool description."""

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        original_tools = await super().get_tools(ctx)

        def _build_description(tool: ToolsetTool[AgentDepsT]) -> str:
            base_desc = tool.tool_def.description or ''
            if (self.include_return_schema or tool.include_return_schema) and tool.tool_def.return_schema is not None:
                # TODO: This should be overrideable by PromptConfig when that lands
                return '\n\n'.join([base_desc, 'Return schema:', str(tool.tool_def.return_schema)])
            return base_desc

        # All tools pass through, only descriptions are conditionally modified
        return {
            name: replace(
                original_tools[name],
                tool_def=replace(
                    tool.tool_def,
                    description=_build_description(tool),
                ),
            )
            for name, tool in original_tools.items()
        }
