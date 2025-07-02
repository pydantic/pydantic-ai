from __future__ import annotations

from dataclasses import dataclass

from .._run_context import AgentDepsT, RunContext
from ..exceptions import UserError
from ..tools import (
    ToolDefinition,
    ToolPrepareFunc,
)
from ._mapped import MappedToolset
from ._run import RunToolset
from .wrapper import WrapperToolset


@dataclass
class IndividuallyPreparedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prepares the tools it contains using a per-tool prepare function."""

    prepare_func: ToolPrepareFunc[AgentDepsT]

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        wrapped_for_run = await self.wrapped.prepare_for_run(ctx)

        tool_defs: dict[str, ToolDefinition] = {}
        name_map: dict[str, str] = {}
        for original_tool_def in wrapped_for_run.tool_defs:
            original_name = original_tool_def.name
            tool_def = await self.prepare_func(ctx, original_tool_def)
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

        mapped_for_run = await MappedToolset(wrapped_for_run, list(tool_defs.values()), name_map).prepare_for_run(ctx)
        return RunToolset(mapped_for_run, ctx, original=self)
