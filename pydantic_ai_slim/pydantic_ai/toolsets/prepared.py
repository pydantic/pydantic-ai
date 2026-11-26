from __future__ import annotations

import inspect
from dataclasses import dataclass, replace

from .. import _utils
from .._run_context import AgentDepsT, RunContext, without_tool_name_path
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
        result = self.prepare_func(without_tool_name_path(ctx), original_tool_defs)
        if inspect.isawaitable(result):
            result = await result
        prepared_tool_defs_by_name = {
            tool_def.name: tool_def for tool_def in _utils.check_tools_prepare_func_result(result, self.prepare_func)
        }

        if len(prepared_tool_defs_by_name.keys() - original_tools.keys()) > 0:
            raise UserError(
                'Prepare function cannot add or rename tools. Use `FunctionToolset.add_function()` or `RenamedToolset` instead.'
            )

        return {
            name: replace(original_tools[name], tool_def=_with_original_owner(tool_def, original_tools[name].tool_def))
            for name, tool_def in prepared_tool_defs_by_name.items()
        }


def _with_original_owner(tool_def: ToolDefinition, original: ToolDefinition) -> ToolDefinition:
    """`tool_def`, still owned by the capability that contributed the tool.

    The owning capability is fixed for the run, and the tool's content is held to its authority. A prepare function
    that builds a new `ToolDefinition` instead of `replace()`-ing the one it was given would otherwise drop it.
    """
    if tool_def.capability_id == original.capability_id:
        return tool_def
    return replace(tool_def, capability_id=original.capability_id)
