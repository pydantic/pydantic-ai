from __future__ import annotations

import json
from dataclasses import dataclass, replace

from .._run_context import AgentDepsT, RunContext
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass
class ReturnSchemaToolset(WrapperToolset[AgentDepsT]):
    """A toolset that adds the return schema to the tool description if it is present.

    This wrapper toolset inspects each tool's `return_schema` and, when enabled, appends
    a JSON representation of the schema to the tool's description. This helps LLMs understand
    what data a tool returns, enabling better planning for multi-step operations and tool chaining.

    The return schema can be enabled at two levels:
    - Toolset-level: Set `include_return_schema=True` when creating this wrapper
    - Tool-level: Individual tools can opt-in via their `include_return_schema` flag

    See [toolset docs](../toolsets.md#return-schema-toolset) for more information.
    """

    include_return_schema: bool = True
    """Whether to include the return schema in the tool description.
    
    Defaults to True since the purpose of this wrapper is to include return schemas.
    Individual tools can still opt-in independently via their own `include_return_schema` flag.
    """

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        original_tools = await super().get_tools(ctx)

        def _build_description(tool: ToolsetTool[AgentDepsT]) -> str | None:
            if (self.include_return_schema or tool.include_return_schema) and tool.tool_def.return_schema is not None:
                # TODO: This should be overrideable by PromptConfig when that lands
                base_desc = tool.tool_def.description or ''
                return '\n\n'.join([base_desc, 'Return schema:', json.dumps(tool.tool_def.return_schema, indent=2)])
            # Preserve the original description (including None)
            return tool.tool_def.description

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
