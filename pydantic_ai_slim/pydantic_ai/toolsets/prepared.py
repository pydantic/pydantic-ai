from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

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

        # Start with a shallow copy to avoid mutating the parent's dict
        result_tools = dict(original_tools)

        # Iterate tool_config - skip tools that don't exist in this toolset
        # (tool_config may be shared across multiple toolsets, e.g. function tools + output tools)
        for tool_name, config in self.tool_config.items():
            if tool_name not in original_tools:
                continue

            tool = original_tools[tool_name]
            original_tool_def = tool.tool_def
            parameters_json_schema = copy.deepcopy(original_tool_def.parameters_json_schema)

            if config.parameters_descriptions:
                self._update_arg_descriptions(parameters_json_schema, config.parameters_descriptions, tool_name)

            updated_tool_def = replace(
                original_tool_def,
                parameters_json_schema=parameters_json_schema,
                **{
                    k: v
                    for k, v in {
                        'name': config.name,
                        'description': config.description,
                        'strict': config.strict,
                    }.items()
                    if v is not None
                },
            )

            updated_tool = replace(tool, tool_def=updated_tool_def)

            # Handle renaming: remove old key if renamed, then add with final name
            final_tool_name = config.name if config.name is not None else tool_name
            if final_tool_name != tool_name:
                del result_tools[tool_name]
            result_tools[final_tool_name] = updated_tool

        return result_tools

    def _update_arg_descriptions(
        self,
        schema: dict[str, Any],
        arg_descriptions: dict[str, str],
        tool_name: str,
    ) -> None:
        """Update descriptions for argument paths in the JSON schema (modifies schema in place)."""
        defs = schema.get(_DEFS, {})

        for arg_path, description in arg_descriptions.items():
            current = schema
            parts = arg_path.split('.')

            for i, part in enumerate(parts):
                # Resolve $ref if present.
                # We inline the definition to avoid modifying shared definitions in $defs
                # and to handle chained references (A -> B -> C).
                visited_refs: set[str] = set()
                while (ref := current.get(_REF)) and isinstance(ref, str) and ref.startswith(_REF_PREFIX):
                    if ref in visited_refs:
                        raise UserError(f"Circular reference detected in schema at '{arg_path}': {ref}")
                    visited_refs.add(ref)

                    def_name = ref[len(_REF_PREFIX) :]
                    if def_name not in defs:
                        raise UserError(f"Invalid path '{arg_path}' for tool '{tool_name}': undefined $ref '{ref}'.")

                    # Inline the definition: replace 'current' contents with the definition's contents
                    # This ensures we don't mutate the shared definition in $defs
                    # "sender": { "$ref": "#/$defs/User" }
                    # "receiver": { "$ref": "#/$defs/User" }
                    # We write to the 'sender' key itself and not the $defs key because we don't want to mutate the shared definition which is also used by receiver.
                    #   "sender": {
                    #     "description": "I CHANGED THIS!",
                    #     # ... copy of other User fields ...
                    # }

                    target_def = defs[def_name]
                    current.pop(_REF)
                    current.update(copy.deepcopy(target_def))

                props = current.get(_PROPERTIES, {})
                if part not in props:
                    available = ', '.join(f"'{p}'" for p in props)
                    raise UserError(
                        f"Invalid path '{arg_path}' for tool '{tool_name}': "
                        f"'{part}' not found. Available properties: {available}"
                    )

                if i == len(parts) - 1:
                    props[part][_DESCRIPTION] = description
                else:
                    current = props[part]


# JSON Schema keys used for traversal
_PROPERTIES = 'properties'
_DEFS = '$defs'
_REF = '$ref'
_REF_PREFIX = '#/$defs/'
_DESCRIPTION = 'description'
