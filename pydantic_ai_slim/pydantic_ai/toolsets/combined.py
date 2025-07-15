from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic_core import SchemaValidator
from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from .._utils import get_async_lock
from ..tools import ToolDefinition
from .abstract import AbstractToolset


@dataclass
class CombinedToolset(AbstractToolset[AgentDepsT]):
    """A toolset that combines multiple toolsets."""

    toolsets: list[AbstractToolset[AgentDepsT]]

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

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return CombinedToolset(await asyncio.gather(*(toolset.for_run(ctx) for toolset in self.toolsets)))

    async def list_tool_defs(self, ctx: RunContext[AgentDepsT]) -> list[ToolDefinition]:
        tool_defs = await asyncio.gather(*(toolset.list_tool_defs(ctx) for toolset in self.toolsets))
        return [tool_def for toolset_tool_defs in tool_defs for tool_def in toolset_tool_defs]

    def max_retries_for_tool(self, name: str) -> int:
        return self._toolset_for_tool_name(name).max_retries_for_tool(name)

    def get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return self._toolset_for_tool_name(name).get_tool_args_validator(ctx, name)

    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any]) -> Any:
        return await self._toolset_for_tool_name(name).call_tool(ctx, name, tool_args)

    def accept(self, visitor: Callable[[AbstractToolset[AgentDepsT]], Any]) -> Any:
        for toolset in self.toolsets:
            toolset.accept(visitor)

    def _toolset_for_tool_name(self, name: str) -> AbstractToolset[AgentDepsT]:
        try:
            tool = next(
                toolset for toolset in self.toolsets if any(tool_def.name == name for tool_def in toolset.tool_defs)
            )
        except KeyError as e:
            raise ValueError(f'Tool {name!r} not found in any toolset') from e
