from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Callable

from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from .._utils import get_async_lock
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

    _enter_lock: asyncio.Lock = field(compare=False, init=False)
    _entered_count: int = field(init=False)
    _exit_stack: AsyncExitStack | None = field(init=False)

    def __post_init__(self):
        self._enter_lock = get_async_lock()
        self._entered_count = 0
        self._exit_stack = None

    async def __aenter__(self) -> Self:
        async with self._enter_lock:
            if self._entered_count == 0:
                async with AsyncExitStack() as exit_stack:
                    for toolset in self.toolsets:
                        await exit_stack.enter_async_context(toolset)
                    self._exit_stack = exit_stack.pop_all()
            self._entered_count += 1
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        async with self._enter_lock:
            self._entered_count -= 1
            if self._entered_count == 0 and self._exit_stack is not None:
                await self._exit_stack.aclose()
                self._exit_stack = None

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
