from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition, ToolSelector, matches_tool_selector
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass(init=False)
class FilteredToolset(WrapperToolset[AgentDepsT]):
    """A toolset that filters the tools it contains using a [`ToolSelector`][pydantic_ai.tools.ToolSelector].

    Both sync and async filter functions are accepted as part of the `ToolSelector` type.

    See [toolset docs](../toolsets.md#filtering-tools) for more information.
    """

    tools: ToolSelector[AgentDepsT]

    def __init__(
        self,
        wrapped: Any,  # AbstractToolset[AgentDepsT] — Any to avoid circular import
        tools: ToolSelector[AgentDepsT] | None = None,
        *,
        filter_func: Callable[[RunContext[AgentDepsT], ToolDefinition], bool | Awaitable[bool]] | None = None,
    ) -> None:
        if tools is not None and filter_func is not None:
            raise TypeError("Cannot specify both 'tools' and 'filter_func'.")
        if filter_func is not None:
            warnings.warn(
                "'filter_func' is deprecated, use 'tools' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.tools = filter_func
        elif tools is not None:
            self.tools = tools
        else:
            raise TypeError("Either 'tools' or 'filter_func' must be specified.")
        self.wrapped = wrapped

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        result: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in (await super().get_tools(ctx)).items():
            if await matches_tool_selector(self.tools, ctx, tool.tool_def):
                result[name] = tool
        return result
