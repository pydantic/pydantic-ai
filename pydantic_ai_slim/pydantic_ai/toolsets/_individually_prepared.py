from __future__ import annotations

from dataclasses import dataclass, replace

from pydantic_ai.toolsets import AbstractToolset

from .._run_context import AgentDepsT, RunContext
from ..exceptions import UserError
from ..tools import (
    ToolDefinition,
    ToolPrepareFunc,
)
from ._mapped import MappedToolset
from .wrapper import WrapperToolset


@dataclass
class IndividuallyPreparedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prepares the tools it contains using a per-tool prepare function."""

    prepare_func: ToolPrepareFunc[AgentDepsT]

    async def _rewrap_for_run(
        self, wrapped: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
    ) -> WrapperToolset[AgentDepsT]:
        tool_defs: dict[str, ToolDefinition] = {}
        name_map: dict[str, str] = {}
        for original_tool_def in wrapped.tool_defs:
            original_name = original_tool_def.name

            run_context = replace(ctx, tool_name=original_name, retry=ctx.retries.get(original_name, 0))
            tool_def = await self.prepare_func(run_context, original_tool_def)
            if not tool_def:
                continue

            new_name = tool_def.name
            if new_name in tool_defs:
                if new_name != original_name:
                    raise UserError(f'Renaming tool {original_name!r} to {new_name!r} conflicts with existing tool.')
                else:
                    raise UserError(f'Tool name conflicts with previously renamed tool: {new_name!r}.')
            name_map[new_name] = original_name

            tool_defs[new_name] = tool_def

        return MappedToolset(wrapped, list(tool_defs.values()), name_map)
