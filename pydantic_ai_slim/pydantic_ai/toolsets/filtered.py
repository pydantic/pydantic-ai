from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from . import AbstractToolset
from .individually_prepared import IndividuallyPreparedToolset


@dataclass(init=False)
class FilteredToolset(IndividuallyPreparedToolset[AgentDepsT]):
    """A toolset that filters the tools it contains using a filter function."""

    def __init__(
        self,
        toolset: AbstractToolset[AgentDepsT],
        filter_func: Callable[[RunContext[AgentDepsT], ToolDefinition], bool],
    ):
        async def filter_tool_def(ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition | None:
            return tool_def if filter_func(ctx, tool_def) else None

        super().__init__(toolset, filter_tool_def)
