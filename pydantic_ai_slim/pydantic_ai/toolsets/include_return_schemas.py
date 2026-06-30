from __future__ import annotations

from dataclasses import dataclass, field, replace

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition, ToolsPrepareFunc
from .abstract import AbstractToolset
from .prepared import PreparedToolset


@dataclass(init=False)
class IncludeReturnSchemasToolset(PreparedToolset[AgentDepsT]):
    """A toolset that sets `include_return_schema=True` on all its tools.

    See [toolset docs](../toolsets.md) for more information.
    """

    prepare_func: ToolsPrepareFunc[AgentDepsT] = field(init=False, repr=False)

    def __init__(self, wrapped: AbstractToolset[AgentDepsT]) -> None:
        async def _include(ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [
                replace(td, include_return_schema=True) if td.include_return_schema is None else td for td in tool_defs
            ]

        super().__init__(wrapped=wrapped, prepare_func=_include)
