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
class CombinedToolset(AbstractToolset[AgentDepsT]):
    """A toolset that combines multiple toolsets."""

    toolsets: Sequence[AbstractToolset[AgentDepsT]]

    _enter_lock: asyncio.Lock = field(compare=False, init=False)
    _entered_count: int = field(init=False)
    _exit_stack: AsyncExitStack | None = field(init=False)

    def __post_init__(self):
        self._enter_lock = get_async_lock()
        self._entered_count = 0
        self._exit_stack = None

    @property
    def name(self) -> str:
        return 'Toolset'  # pragma: no cover

    async def __aenter__(self) -> Self:
        async with self._enter_lock:
            if self._entered_count == 0:
                self._exit_stack = AsyncExitStack()
                for toolset in self.toolsets:
                    await self._exit_stack.enter_async_context(toolset)
            self._entered_count += 1
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        async with self._enter_lock:
            self._entered_count -= 1
            if self._entered_count == 0 and self._exit_stack is not None:
                await self._exit_stack.aclose()
                self._exit_stack = None

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        toolsets = await asyncio.gather(*(toolset.for_run_step(ctx) for toolset in self.toolsets))
        tools, tool_toolsets = await self._get_tools(ctx, toolsets)
        return _CachedCombinedToolset(toolsets, tools, tool_toolsets)

    @staticmethod
    async def _get_tools(
        ctx: RunContext[AgentDepsT], toolsets: Sequence[AbstractToolset[AgentDepsT]]
    ) -> tuple[dict[str, ToolsetTool[AgentDepsT]], dict[str, AbstractToolset[AgentDepsT]]]:
        toolsets_tools = await asyncio.gather(*(toolset.get_tools(ctx) for toolset in toolsets))
        combined_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        tool_toolsets: dict[str, AbstractToolset[AgentDepsT]] = {}

        for toolset, tools in zip(toolsets, toolsets_tools):
            for name, tool in tools.items():
                try:
                    existing_tool = combined_tools[name]
                    raise UserError(
                        f'{toolset.name} defines a tool whose name conflicts with existing tool from {existing_tool.toolset.name}: {name!r}. {toolset.tool_name_conflict_hint}'
                    )
                except KeyError:
                    combined_tools[name] = tool
                    tool_toolsets[name] = toolset
        return combined_tools, tool_toolsets

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools, _ = await self._get_tools(ctx, self.toolsets)
        return tools

    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any]) -> Any:
        raise NotImplementedError('CombinedToolset cannot be used to directly call tools')

    def accept(self, visitor: Callable[[AbstractToolset[AgentDepsT]], Any]) -> Any:
        for toolset in self.toolsets:
            toolset.accept(visitor)


@dataclass
class _CachedCombinedToolset(CombinedToolset[AgentDepsT]):
    tools: dict[str, ToolsetTool[AgentDepsT]]
    tool_toolsets: dict[str, AbstractToolset[AgentDepsT]]

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return self.tools

    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any]) -> Any:
        return await self.tool_toolsets[name].call_tool(ctx, name, tool_args)
