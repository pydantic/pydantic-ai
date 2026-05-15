from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition, ToolSelector, ToolsPrepareFunc, matches_tool_selector
from .abstract import AbstractToolset
from .prepared import PreparedToolset


@dataclass(init=False)
class SetMetadataToolset(PreparedToolset[AgentDepsT]):
    """A toolset that merges metadata key-value pairs onto selected tools.

    See [toolset docs](../toolsets.md) for more information.
    """

    prepare_func: ToolsPrepareFunc[AgentDepsT] = field(init=False, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        metadata: dict[str, Any],
        *,
        tools: ToolSelector[AgentDepsT] = 'all',
    ) -> None:
        self.metadata = metadata
        selector = tools

        async def _set_metadata(ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [
                replace(td, metadata={**(td.metadata or {}), **self.metadata})
                if await matches_tool_selector(selector, ctx, td)
                else td
                for td in tool_defs
            ]

        super().__init__(wrapped=wrapped, prepare_func=_set_metadata)
