from __future__ import annotations

import inspect
from dataclasses import dataclass, replace

from pydantic_graph.util import get_callable_name

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
        result = self.prepare_func(ctx, original_tool_defs)
        if inspect.isawaitable(result):
            result = await result
        if result is None:
            raise TypeError(
                f'prepare callback {get_callable_name(self.prepare_func)!r} returned `None`; '
                'return `[]` to hide all tool definitions explicitly, or `tool_defs` to pass them through unchanged.'
            )
        prepared_tool_defs_by_name = {tool_def.name: tool_def for tool_def in result}

        if len(prepared_tool_defs_by_name.keys() - original_tools.keys()) > 0:
            raise UserError(
                'Prepare function cannot add or rename tools. Use `FunctionToolset.add_function()` or `RenamedToolset` instead.'
            )

        return {
            name: replace(original_tools[name], tool_def=tool_def)
            for name, tool_def in prepared_tool_defs_by_name.items()
        }
