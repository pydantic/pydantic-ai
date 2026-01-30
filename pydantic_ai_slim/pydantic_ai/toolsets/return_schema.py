from __future__ import annotations

from dataclasses import dataclass, replace

from .._run_context import AgentDepsT, RunContext
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass
class ReturnSchemaToolset(WrapperToolset[AgentDepsT]):
    """A toolset that gates whether return schemas are included in tool definitions.

    When include_return_schema is enabled (at toolset or tool level), the return_schema
    field is preserved on ToolDefinition. The model layer then decides how to present it
    (natively via API field, or as fallback text in description).

    When not enabled, return_schema is cleared to None so the model layer ignores it.

    The return schema can be enabled at two levels:
    - Toolset-level: Set `include_return_schema=True` when creating this wrapper
    - Tool-level: Individual tools can opt-in via their `include_return_schema` flag

    See [toolset docs](../toolsets.md#return-schema-toolset) for more information.
    """

    include_return_schema: bool = True
    """Whether to include the return schema in tool definitions.

    Defaults to True since the purpose of this wrapper is to include return schemas.
    Individual tools can still opt-in independently via their own `include_return_schema` flag.
    """

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        original_tools = await super().get_tools(ctx)

        def _gate_return_schema(tool: ToolsetTool[AgentDepsT]) -> ToolsetTool[AgentDepsT]:
            opted_in = (self.include_return_schema or tool.include_return_schema) and tool.tool_def.return_schema is not None
            if opted_in:
                # Keep return_schema as-is; model layer will decide how to present it
                return tool
            else:
                # Clear return_schema so model layer ignores it
                return replace(tool, tool_def=replace(tool.tool_def, return_schema=None))

        return {name: _gate_return_schema(tool) for name, tool in original_tools.items()}
