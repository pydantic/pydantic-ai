from __future__ import annotations

from dataclasses import dataclass

from .._run_context import AgentDepsT, RunContext
from ..exceptions import UserError
from ..tools import ToolsPrepareFunc
from .run import RunToolset
from .wrapper import WrapperToolset


@dataclass
class PreparedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prepares the tools it contains using a prepare function."""

    prepare_func: ToolsPrepareFunc[AgentDepsT]

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        wrapped_for_run = await self.wrapped.prepare_for_run(ctx)
        original_tool_defs = wrapped_for_run.tool_defs
        prepared_tool_defs = await self.prepare_func(ctx, original_tool_defs) or []

        original_tool_names = {tool_def.name for tool_def in original_tool_defs}
        prepared_tool_names = {tool_def.name for tool_def in prepared_tool_defs}
        if len(prepared_tool_names - original_tool_names) > 0:
            raise UserError('Prepare function is not allowed to change tool names or add new tools.')

        prepared_for_run = PreparedToolset(wrapped_for_run, self.prepare_func)
        return RunToolset(prepared_for_run, ctx, prepared_tool_defs)
