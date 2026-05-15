from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from typing import Any

from .._run_context import AgentDepsT, RunContext
from ..exceptions import UserError
from ..tools import ToolSelector, ToolsPrepareFunc, matches_tool_selector
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass(init=False)
class PreparedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prepares the tools it contains using a prepare function that takes the agent context and the original tool definitions.

    See [toolset docs](../toolsets.md#preparing-tool-definitions) for more information.
    """

    prepare_func: ToolsPrepareFunc[AgentDepsT]
    tools: ToolSelector[AgentDepsT]

    def __init__(
        self,
        wrapped: Any,  # AbstractToolset[AgentDepsT] — Any to avoid circular import
        prepare_func: ToolsPrepareFunc[AgentDepsT],
        *,
        tools: ToolSelector[AgentDepsT] = 'all',
    ) -> None:
        self.wrapped = wrapped
        self.prepare_func = prepare_func
        self.tools = tools

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        original_tools = await super().get_tools(ctx)

        if self.tools == 'all':
            # Fast path: prepare all tools (original behavior)
            matching_tools = original_tools
            passthrough_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        else:
            matching_tools = {}
            passthrough_tools = {}
            for name, tool in original_tools.items():
                if await matches_tool_selector(self.tools, ctx, tool.tool_def):
                    matching_tools[name] = tool
                else:
                    passthrough_tools[name] = tool

        matching_tool_defs = [tool.tool_def for tool in matching_tools.values()]
        result = self.prepare_func(ctx, matching_tool_defs)
        if inspect.isawaitable(result):
            result = await result
        prepared_tool_defs_by_name = {tool_def.name: tool_def for tool_def in (result or [])}

        if len(prepared_tool_defs_by_name.keys() - matching_tools.keys()) > 0:
            raise UserError(
                'Prepare function cannot add or rename tools. Use `FunctionToolset.add_function()` or `RenamedToolset` instead.'
            )

        prepared = {
            name: replace(matching_tools[name], tool_def=tool_def)
            for name, tool_def in prepared_tool_defs_by_name.items()
        }
        return {
            name: prepared[name] if name in prepared else passthrough_tools[name]
            for name in original_tools
            if name in prepared or name in passthrough_tools
        }
