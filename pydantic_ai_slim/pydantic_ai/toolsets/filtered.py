from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass
class FilteredToolset(WrapperToolset[AgentDepsT]):
    """A toolset that filters the tools it contains using a filter function that takes the agent context and the tool definition.

    Both sync and async filter functions are accepted.

    See [toolset docs](../toolsets.md#filtering-tools) for more information.
    """

    filter_func: Callable[[RunContext[AgentDepsT], ToolDefinition], bool | Awaitable[bool]]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        result: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in (await super().get_tools(ctx)).items():
            match = self.filter_func(ctx, tool.tool_def)
            if inspect.isawaitable(match):
                match = await match
            if match:
                result[name] = tool
        return result
