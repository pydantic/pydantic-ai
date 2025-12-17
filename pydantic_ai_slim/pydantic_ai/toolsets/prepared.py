from __future__ import annotations

from copy import deepcopy
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
    """A toolset that prepares the tools it contains using a ToolConfig.

    See [toolset docs](../toolsets.md#preparing-tool-definitions) for more information.
    """

    tool_config: ToolConfig

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        original_tools = await super().get_tools(ctx)
        return await self._get_tools_from_tool_config(original_tools)

    async def _get_tools_from_tool_config(
        self,
        original_tools: dict[str, ToolsetTool[AgentDepsT]],
    ) -> dict[str, ToolsetTool[AgentDepsT]]:
        tool_descriptions = self.tool_config.tool_descriptions

        for tool_name, description in tool_descriptions.items():
            if tool_name in original_tools:
                original_tool = original_tools[tool_name]
                updated_tool_def = replace(original_tool.tool_def, description=description)
                original_tools[tool_name] = replace(original_tool, tool_def=updated_tool_def)

        for tool_name in list(original_tools.keys()):
            tool_args = self.tool_config.tool_args_descriptions.get(tool_name, {})
            if not tool_args:
                continue

            original_tool = original_tools[tool_name]
            parameter_defs = deepcopy(original_tool.tool_def.parameters_json_schema)

            for param_name, param_schema in parameter_defs.get('properties', {}).items():
                if param_name in tool_args:
                    param_schema['description'] = tool_args[param_name]

            updated_tool_def = replace(original_tool.tool_def, parameters_json_schema=parameter_defs)
            original_tools[tool_name] = replace(original_tool, tool_def=updated_tool_def)

        return original_tools
