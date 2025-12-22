from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai._tool_arg_descriptions import ToolArgDescriptions
from pydantic_ai.prompt_config import ToolConfig

from .._run_context import AgentDepsT, RunContext
from ..exceptions import UserError
from ..tools import ToolDefinition, ToolsPrepareFunc
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

    async def get_all_tool_definitions(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        # We want all the tool definitions, but updated
        # I should apply prepare_func to each tool definition and return the updated list
        # I need to ensure no tool is removed with this step though
        return await super().get_all_tool_definitions(ctx)

    @staticmethod
    def create_tool_config_prepare_func(tool_config: dict[str, ToolConfig]) -> ToolsPrepareFunc[AgentDepsT]:
        async def prepare_func(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            # Given a ctx, which I am not sure is useful for our case, and tool_defs let us update these tool_defs using ToolConfig and return the updated list
            updated_tool_defs: list[ToolDefinition] = []

            for tool_def in tool_defs:
                # get_tool_config is it exists for this current_tool
                tool_name = tool_def.name
                if tool_name not in tool_config:
                    updated_tool_defs.append(tool_def)
                    continue  # Nothing to be done here, no configuration_present

                config = tool_config[tool_name]
                parameters_json_schema = tool_def.parameters_json_schema

                if config.parameters_descriptions:
                    parameters_json_schema = ToolArgDescriptions.update_in_json_schema(
                        parameters_json_schema, config.parameters_descriptions, tool_name
                    )

                updated_tool_def = replace(
                    tool_def,
                    parameters_json_schema=parameters_json_schema,
                    **{
                        k: v
                        for k, v in {
                            # 'name': config.name,
                            # Not changing the name part here, will delegate this work to RenamedToolset
                            'description': config.description,
                            'strict': config.strict,
                        }.items()
                        if v is not None
                    },
                )

                updated_tool_defs.append(updated_tool_def)

            return updated_tool_defs

        return prepare_func


def tool_config_prepare_func(tool_config: dict[str, ToolConfig]):
    """Create prepare_func using tool_config to be used with PreparedToolset."""

    async def prepare_func(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        # Given a ctx, which I am not sure is useful for our case, and tool_defs let us update these tool_defs using ToolConfig and return the updated list
        updated_tool_defs: list[ToolDefinition] = []

        for tool_def in tool_defs:
            # get_tool_config is it exists for this current_tool
            tool_name = tool_def.name
            if tool_name not in tool_config:
                updated_tool_defs.append(tool_def)
                continue  # Nothing to be done here, no configuration_present

            config = tool_config[tool_name]
            parameters_json_schema = tool_def.parameters_json_schema

            if config.parameters_descriptions:
                parameters_json_schema = ToolArgDescriptions.update_in_json_schema(
                    parameters_json_schema, config.parameters_descriptions, tool_name
                )

            updated_tool_def = replace(
                tool_def,
                parameters_json_schema=parameters_json_schema,
                **{
                    k: v
                    for k, v in {
                        # 'name': config.name,
                        # Not changing the name part here, will delegate this work to RenamedToolset
                        'description': config.description,
                        'strict': config.strict,
                    }.items()
                    if v is not None
                },
            )

            updated_tool_defs.append(updated_tool_def)

        return updated_tool_defs

    return prepare_func
