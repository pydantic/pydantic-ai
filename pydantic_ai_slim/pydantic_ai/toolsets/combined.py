from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from pydantic_core import SchemaValidator
from typing_extensions import Self

from .._run_context import AgentDepsT, RunContext
from ..exceptions import UserError
from ..tools import ToolDefinition
from . import AbstractToolset
from ._run import RunToolset


@dataclass(init=False)
class CombinedToolset(AbstractToolset[AgentDepsT]):
    """A toolset that combines multiple toolsets."""

    toolsets: list[AbstractToolset[AgentDepsT]]
    _toolset_per_tool_name: dict[str, AbstractToolset[AgentDepsT]]
    _exit_stack: AsyncExitStack | None
    _running_count: int

    def __init__(self, toolsets: Sequence[AbstractToolset[AgentDepsT]]):
        self._exit_stack = None
        self._running_count = 0
        self.toolsets = list(toolsets)

        self._toolset_per_tool_name = {}
        for toolset in self.toolsets:
            for name in toolset.tool_names:
                try:
                    existing_toolset = self._toolset_per_tool_name[name]
                    raise UserError(
                        f'{toolset.name} defines a tool whose name conflicts with existing tool from {existing_toolset.name}: {name!r}. {toolset.tool_name_conflict_hint}'
                    )
                except KeyError:
                    pass
                self._toolset_per_tool_name[name] = toolset

    async def __aenter__(self) -> Self:
        if self._running_count == 0:
            self._exit_stack = AsyncExitStack()
            for toolset in self.toolsets:
                await self._exit_stack.enter_async_context(toolset)
        self._running_count += 1
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        self._running_count -= 1
        if self._running_count <= 0 and self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        return None

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        toolsets_for_run = await asyncio.gather(*[toolset.prepare_for_run(ctx) for toolset in self.toolsets])
        combined_for_run = CombinedToolset(toolsets_for_run)
        return RunToolset(combined_for_run, ctx)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return [tool_def for toolset in self.toolsets for tool_def in toolset.tool_defs]

    @property
    def tool_names(self) -> list[str]:
        return list(self._toolset_per_tool_name.keys())

    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return self._toolset_for_tool_name(name)._get_tool_args_validator(ctx, name)

    def validate_tool_args(
        self, ctx: RunContext[AgentDepsT], name: str, args: str | dict[str, Any] | None, allow_partial: bool = False
    ) -> dict[str, Any]:
        return self._toolset_for_tool_name(name).validate_tool_args(ctx, name, args, allow_partial)

    def _max_retries_for_tool(self, name: str) -> int:
        return self._toolset_for_tool_name(name)._max_retries_for_tool(name)

    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        return await self._toolset_for_tool_name(name).call_tool(ctx, name, tool_args, *args, **kwargs)

    def _toolset_for_tool_name(self, name: str) -> AbstractToolset[AgentDepsT]:
        try:
            return self._toolset_per_tool_name[name]
        except KeyError as e:
            raise ValueError(f'Tool {name!r} not found in any toolset') from e
