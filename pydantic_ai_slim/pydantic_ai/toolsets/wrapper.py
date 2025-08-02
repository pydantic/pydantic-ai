from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable

from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from .abstract import AbstractToolset, ToolsetTool


@dataclass
class WrapperToolset(AbstractToolset[AgentDepsT]):
    """A toolset that wraps another toolset and delegates to it.

    See [toolset docs](../toolsets.md#wrapping-a-toolset) for more information.
    """

    wrapped: AbstractToolset[AgentDepsT]

    @asynccontextmanager
    async def setup(self) -> AsyncGenerator[Self, Any]:
        async with self.wrapped.setup():
            yield self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return await self.wrapped.get_tools(ctx)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        self.wrapped.apply(visitor)
