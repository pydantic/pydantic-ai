from __future__ import annotations

from dataclasses import dataclass, field, replace

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition, ToolsPrepareFunc
from .abstract import AbstractToolset
from .prepared import PreparedToolset


@dataclass(init=False)
class DeferredLoadingToolset(PreparedToolset[AgentDepsT]):
    """A toolset that marks tools for deferred loading, hiding them from the model until discovered via tool search.

    See [toolset docs](../toolsets.md#deferred-loading) for more information.
    """

    prepare_func: ToolsPrepareFunc[AgentDepsT] = field(init=False, repr=False)
    tool_names: frozenset[str] | None = None
    """Optional set of tool names to mark for deferred loading. If `None`, all tools are marked for deferred loading."""

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        tool_names: frozenset[str] | None = None,
    ):
        self.tool_names = tool_names

        async def _mark_deferred(_ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [
                replace(td, defer_loading=True) if (tool_names is None or td.name in tool_names) else td
                for td in tool_defs
            ]

        self.wrapped = wrapped
        self.prepare_func = _mark_deferred
