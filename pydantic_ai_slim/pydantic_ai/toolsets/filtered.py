from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .wrapper import WrapperToolset


@dataclass(init=False)
class FilteredToolset(WrapperToolset[AgentDepsT]):
    """A toolset that filters the tools it contains using a filter function."""

    filter_func: Callable[[RunContext[AgentDepsT], ToolDefinition], bool]

    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        return [tool_def for tool_def in await super().list_tool_defs(ctx) if self.filter_func(ctx, tool_def)]
