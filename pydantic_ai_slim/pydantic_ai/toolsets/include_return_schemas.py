from __future__ import annotations

from dataclasses import dataclass, field, replace

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition, ToolSelector, ToolsPrepareFunc, matches_tool_selector
from .abstract import AbstractToolset
from .prepared import PreparedToolset


@dataclass(init=False)
class IncludeReturnSchemasToolset(PreparedToolset[AgentDepsT]):
    """A toolset that sets `include_return_schema=True` on selected tools.

    See [toolset docs](../toolsets.md) for more information.
    """

    prepare_func: ToolsPrepareFunc[AgentDepsT] = field(init=False, repr=False)

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        tools: ToolSelector[AgentDepsT] = 'all',
    ) -> None:
        selector = tools

        async def _include(ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [
                replace(td, include_return_schema=True)
                if td.include_return_schema is None and await matches_tool_selector(selector, ctx, td)
                else td
                for td in tool_defs
            ]

        super().__init__(wrapped=wrapped, prepare_func=_include)
