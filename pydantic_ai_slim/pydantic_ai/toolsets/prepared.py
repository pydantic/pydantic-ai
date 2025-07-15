from __future__ import annotations

from dataclasses import dataclass

from .._run_context import AgentDepsT, RunContext
from ..exceptions import UserError
from ..tools import ToolDefinition, ToolsPrepareFunc
from .wrapper import WrapperToolset


@dataclass
class PreparedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prepares the tools it contains using a prepare function."""

    prepare_func: ToolsPrepareFunc[AgentDepsT]

    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        original_tool_defs = await super().list_tool_defs(ctx)
        prepared_tool_defs = await self.prepare_func(ctx, original_tool_defs) or []

        original_tool_names = {tool_def.name for tool_def in original_tool_defs}
        prepared_tool_names = {tool_def.name for tool_def in prepared_tool_defs}
        if len(prepared_tool_names - original_tool_names) > 0:
            raise UserError(
                'Prepare function cannot add or rename tools. Use `FunctionToolset.add_function()` or `RenamedToolset` instead.'
            )

        return prepared_tool_defs
