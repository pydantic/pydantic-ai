from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from pydantic_ai.prompt_config import ToolConfig

from .._run_context import AgentDepsT, RunContext
from ..exceptions import UserError
from ..tools import ToolsPrepareFunc
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass
class PreparedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prepares the tools it contains using a prepare function that takes the agent context and the original tool definitions.

    See [toolset docs](../toolsets.md#preparing-tool-definitions) for more information.
    """

    prepare_func: ToolsPrepareFunc[AgentDepsT]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        original_tools = await super().get_tools(ctx)
        original_tool_defs = [tool.tool_def for tool in original_tools.values()]
        prepared_tool_defs_by_name = {
            tool_def.name: tool_def for tool_def in (await self.prepare_func(ctx, original_tool_defs) or [])
        }

        if len(prepared_tool_defs_by_name.keys() - original_tools.keys()) > 0:
            raise UserError(
                'Prepare function cannot add or rename tools. Use `FunctionToolset.add_function()` or `RenamedToolset` instead.'
            )

        return {
            name: replace(original_tools[name], tool_def=tool_def)
            for name, tool_def in prepared_tool_defs_by_name.items()
        }


@dataclass
class ToolConfigPreparedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prepares the tools it contains using a ToolConfig."""

    tool_config: dict[str, ToolConfig]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        original_tools = await super().get_tools(ctx)
        tool_config = self.tool_config

        for tool_name in original_tools:
            if tool_name not in tool_config:
                continue
            tool_info = tool_config[tool_name]
            original_tool_def = original_tools[tool_name].tool_def

            parameters_json_schema = copy.deepcopy(original_tool_def.parameters_json_schema)

            if tool_arg_descriptions := tool_info.tool_args_descriptions:
                for arg_name, description in tool_arg_descriptions.items():
                    nodes = arg_name.split('.')
                    total_nodes = len(nodes)

                    current_schema = parameters_json_schema

                    for index_of_node, node in enumerate(nodes):
                        if 'properties' not in current_schema or node not in current_schema['properties']:
                            ref = current_schema.get('$ref', None)
                            if ref:
                                current_schema = parameters_json_schema['$defs'][ref.split('/')[-1]]
                            else:  # Nothing to do here we have to bail out
                                break
                        if index_of_node == total_nodes - 1:
                            # Last node - update the description
                            current_schema['properties'][node]['description'] = description
                        else:
                            # Navigate deeper into nested structure
                            current_schema = current_schema['properties'][node]

            updated_tool_def = replace(
                original_tool_def,
                parameters_json_schema=parameters_json_schema,
                **{
                    k: v
                    for k, v in {
                        'name': tool_info.name,
                        'description': tool_info.tool_description,
                        'strict': tool_info.strict,
                    }.items()
                    if v is not None
                },
            )

            original_tools[tool_name] = replace(original_tools[tool_name], tool_def=updated_tool_def)

        return original_tools
