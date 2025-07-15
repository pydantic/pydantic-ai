from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic_core import SchemaValidator
from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from .._utils import get_async_lock
from ..exceptions import UserError
from ..tools import ToolDefinition
from ._run import RunToolset
from .abstract import AbstractToolset


@dataclass(init=False)
class CombinedToolset(AbstractToolset[AgentDepsT]):
    """A toolset that combines multiple toolsets."""

    toolsets: list[AbstractToolset[AgentDepsT]]
    _toolset_per_tool_name: dict[str, AbstractToolset[AgentDepsT]]

    _enter_lock: asyncio.Lock = field(compare=False)
    _entered_count: int
    _exit_stack: AsyncExitStack | None

    def __init__(self, toolsets: Sequence[AbstractToolset[AgentDepsT]]):
        self._enter_lock = get_async_lock()
        self._entered_count = 0
        self._exit_stack = None

        self.toolsets = list(toolsets)

        self._toolset_per_tool_name = {}
        for toolset in self.toolsets:
            for name in toolset.tool_names:
                try:
                    existing_toolset = self._toolset_per_tool_name[name]
                    raise UserError(
                        f'{toolset.name} defines a tool whose name conflicts with existing tool from {existing_toolset.name}: {name!r}. {toolset._tool_name_conflict_hint}'
                    )
                except KeyError:
                    pass
                self._toolset_per_tool_name[name] = toolset

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

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        toolsets_for_run = await asyncio.gather(*(toolset.prepare_for_run(ctx) for toolset in self.toolsets))
        combined_for_run = CombinedToolset(toolsets_for_run)
        return RunToolset(combined_for_run, ctx, original=self)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return [tool_def for toolset in self.toolsets for tool_def in toolset.tool_defs]

    @property
    def tool_names(self) -> list[str]:
        return list(self._toolset_per_tool_name.keys())

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
            return self._toolset_per_tool_name[name]
        except KeyError as e:
            raise ValueError(f'Tool {name!r} not found in any toolset') from e
