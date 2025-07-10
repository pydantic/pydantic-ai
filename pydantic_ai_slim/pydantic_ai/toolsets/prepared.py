from __future__ import annotations

from dataclasses import dataclass

from .._run_context import AgentDepsT, RunContext
from ..exceptions import UserError
from ..tools import ToolsPrepareFunc
from ._abstract import AbstractToolset
from ._tool_defs import ToolDefsToolset
from .wrapper import WrapperToolset


@dataclass
class PreparedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prepares the tools it contains using a prepare function."""

    prepare_func: ToolsPrepareFunc[AgentDepsT]

    async def _rewrap_for_run(
        self, wrapped: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
    ) -> WrapperToolset[AgentDepsT]:
        original_tool_defs = wrapped.tool_defs
        prepared_tool_defs = await self.prepare_func(ctx, original_tool_defs) or []

        original_tool_names = {tool_def.name for tool_def in original_tool_defs}
        prepared_tool_names = {tool_def.name for tool_def in prepared_tool_defs}
        if len(prepared_tool_names - original_tool_names) > 0:
            raise UserError('Prepare function is not allowed to change tool names or add new tools.')

        return ToolDefsToolset(wrapped, prepared_tool_defs)
