from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pydantic_core import SchemaValidator

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from ._abstract import AbstractToolset
from ._tool_defs import ToolDefsToolset
from .wrapper import WrapperToolset


@dataclass(init=False)
class MappedToolset(ToolDefsToolset[AgentDepsT]):
    """A toolset that maps renamed tool names to original tool names. Used by `IndividuallyPreparedToolset` as the prepare function may rename a tool."""

    name_map: dict[str, str]

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        tool_defs: list[ToolDefinition],
        name_map: dict[str, str] | None = None,
    ):
        super().__init__(wrapped, tool_defs)
        self.name_map = name_map or {}

    async def _rewrap_for_run(
        self, wrapped: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
    ) -> WrapperToolset[AgentDepsT]:
        return MappedToolset(wrapped, self._tool_defs, self.name_map)

    def _max_retries_for_tool(self, name: str) -> int:
        return super()._max_retries_for_tool(self._map_name(name))

    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        original_name = self._map_name(name)
        ctx = replace(ctx, tool_name=original_name)
        return super()._get_tool_args_validator(ctx, original_name)

    async def _call_tool(self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any]) -> Any:
        original_name = self._map_name(name)
        ctx = replace(ctx, tool_name=original_name)
        return await super()._call_tool(ctx, self._map_name(name), tool_args)

    def _map_name(self, name: str) -> str:
        return self.name_map.get(name, name)
