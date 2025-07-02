from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_core import SchemaValidator

from .._run_context import AgentDepsT, RunContext
from ..tools import (
    ToolDefinition,
)
from . import AbstractToolset
from ._run import RunToolset
from .wrapper import WrapperToolset


@dataclass(init=False)
class MappedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that maps renamed tool names to original tool names. Used by `IndividuallyPreparedToolset` as the prepare function may rename a tool."""

    name_map: dict[str, str]
    _tool_defs: list[ToolDefinition]

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        tool_defs: list[ToolDefinition],
        name_map: dict[str, str],
    ):
        super().__init__(wrapped)
        self._tool_defs = tool_defs
        self.name_map = name_map

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        wrapped_for_run = await self.wrapped.prepare_for_run(ctx)
        mapped_for_run = MappedToolset(wrapped_for_run, self._tool_defs, self.name_map)
        return RunToolset(mapped_for_run, ctx)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return self._tool_defs

    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return super()._get_tool_args_validator(ctx, self._map_name(name))

    def _max_retries_for_tool(self, name: str) -> int:
        return super()._max_retries_for_tool(self._map_name(name))

    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        return await super().call_tool(ctx, self._map_name(name), tool_args, *args, **kwargs)

    def _map_name(self, name: str) -> str:
        return self.name_map.get(name, name)
