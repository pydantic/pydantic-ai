from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable

from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from ..exceptions import UserError
from .abstract import AbstractToolset, ToolsetTool


@dataclass
class _CombinedToolsetTool(ToolsetTool[AgentDepsT]):
    """A tool definition for a combined toolset tools that keeps track of the source toolset and tool."""

    source_toolset: AbstractToolset[AgentDepsT]
    source_tool: ToolsetTool[AgentDepsT]


@dataclass
class CombinedToolset(AbstractToolset[AgentDepsT]):
    """A toolset that combines multiple toolsets.

    See [toolset docs](../toolsets.md#combining-toolsets) for more information.
    """

    toolsets: Sequence[AbstractToolset[AgentDepsT]]

    @asynccontextmanager
    async def setup(self) -> AsyncGenerator[Self, Any]:
        async with AsyncExitStack() as exit_stack:
            try:
                for toolset in self.toolsets:
                    await exit_stack.enter_async_context(toolset.setup())
            except Exception as e:
                await exit_stack.aclose()
                raise e

            yield self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        toolsets_tools = await asyncio.gather(*(toolset.get_tools(ctx) for toolset in self.toolsets))
        all_tools: dict[str, ToolsetTool[AgentDepsT]] = {}

        for toolset, tools in zip(self.toolsets, toolsets_tools):
            for name, tool in tools.items():
                if existing_tools := all_tools.get(name):
                    raise UserError(
                        f'{toolset.name} defines a tool whose name conflicts with existing tool from {existing_tools.toolset.name}: {name!r}. {toolset.tool_name_conflict_hint}'
                    )

                all_tools[name] = _CombinedToolsetTool(
                    toolset=tool.toolset,
                    tool_def=tool.tool_def,
                    max_retries=tool.max_retries,
                    args_validator=tool.args_validator,
                    source_toolset=toolset,
                    source_tool=tool,
                )
        return all_tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        assert isinstance(tool, _CombinedToolsetTool)
        return await tool.source_toolset.call_tool(name, tool_args, ctx, tool.source_tool)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        for toolset in self.toolsets:
            toolset.apply(visitor)
